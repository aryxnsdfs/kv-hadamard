#!/usr/bin/env python3
"""
Ternary-V — Phase 2 Stage B (the contribution).

Replaces KIVI's INT4 VALUE cache with TERNARY values {-1, 0, +1} x scale
(~1.58 bits), leaving KIVI's INT4 KEY cache untouched. This isolates the effect
of ternary value quantization, measured against the working KIVI INT4 baseline
on the same model/harness.

TWO-STEP METHODOLOGY (quality before kernel):
  B1 (this file): SIMULATED ternary-V — values are ternary-fake-quantized
      (quantize -> dequantize) and stored as fp16, used via plain matmul. This
      answers the research question FIRST: does ternary V preserve quality
      (perplexity)? No CUDA kernel needed, runs today on the existing harness.
      NOTE: because the simulated cache is stored fp16, B1 does NOT show the
      memory win in `measure_kv_memory` — that is expected. B1 is a QUALITY
      experiment.
  B2 (later, only if B1 holds quality): a packed ~1.58-bit CUDA kernel for the
      real memory saving. Big CUDA effort; not worth it until B1 says quality
      survives.

The K path is byte-for-byte KIVI (INT4 packed kernel). Only the V path differs.
Activated by monkeypatching each attention module's `forward` (patch_ternary_v),
so KIVI's class and the harness are untouched.
"""

import math
import types
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

# KIVI's own building blocks — reused for the (unchanged) K path.
from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb


# ----------------------------------------------------------------------
# Ternary fake-quantization (the method, simulated in fp16).
# ----------------------------------------------------------------------
def ternary_fake_quant(v: torch.Tensor, group_size: int) -> torch.Tensor:
    """Per-group symmetric ternary quant along the last dim, returned dequantized
    to v.dtype (fp16). Each contiguous group of `group_size` elements gets one
    scale; values map to {-1, 0, +1} * scale by absmax rounding.

        scale = max(|x|) over the group
        q     = round(x / scale)  in {-1, 0, +1}
        x_hat = q * scale

    absmax rounding is the simplest defensible ternary scheme. A threshold
    variant (TWN-style, t = 0.7*mean|x|) is a natural knob to try in B2.
    """
    shape = v.shape
    D = shape[-1]
    assert D % group_size == 0, f"head_dim {D} not divisible by group_size {group_size}"
    g = D // group_size
    x = v.reshape(*shape[:-1], g, group_size)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    q = torch.round(x / scale).clamp_(-1, 1)
    x_hat = (q * scale).reshape(shape)
    return x_hat.to(v.dtype)


# ----------------------------------------------------------------------
# Hadamard + int2 fake-quant (QuaRot/SpinQuant-style scalar route).
# Rotate V by a normalized Hadamard H along head_dim -> spreads outliers evenly
# -> a coarse uniform 2-bit (4-level, asymmetric) grid now fits with little loss
# -> rotate back after (H is symmetric orthonormal, so H^-1 = H). Scalar + int =
# hardware-friendly (the reviewer-proof baseline). 2 bits/elem vs KIVI's 4 = 2x.
# ----------------------------------------------------------------------
_HAD_CACHE = {}


def _hadamard(D, device, dtype):
    key = (D, device, dtype)
    H = _HAD_CACHE.get(key)
    if H is None:
        assert (D & (D - 1)) == 0, f"head_dim {D} must be power of 2 for Hadamard"
        H = torch.ones(1, 1)
        while H.shape[0] < D:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        H = (H / math.sqrt(D)).to(device=device, dtype=dtype)  # orthonormal, symmetric
        _HAD_CACHE[key] = H
    return H


def _asym_uniform_quant(x, group_size, bits):
    """Per-group asymmetric uniform quant (zero-point + scale), all 2^bits levels
    used. Mirrors KIVI's own value scheme (scale + min)."""
    shape = x.shape
    D = shape[-1]
    g = D // group_size
    xg = x.reshape(*shape[:-1], g, group_size)
    mn = xg.amin(dim=-1, keepdim=True)
    mx = xg.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / (2 ** bits - 1)).clamp(min=1e-8)
    q = torch.round((xg - mn) / scale).clamp_(0, 2 ** bits - 1)
    return (q * scale + mn).reshape(shape)


def hadamard_int_fake_quant(v, group_size, bits=2):
    D = v.shape[-1]
    H = _hadamard(D, v.device, v.dtype)
    vr = v.to(torch.float32) @ H.to(torch.float32)          # rotate (fp32 for the 128-term sum)
    vr_hat = _asym_uniform_quant(vr, group_size, bits)
    vhat = vr_hat @ H.to(torch.float32)                     # H symmetric orthonormal -> rotate back
    return vhat.to(v.dtype)


def hadamard_ternary_fake_quant(v, group_size):
    """Rotate -> ternary {-1,0,+1}*scale -> rotate back. Tests whether the
    outlier-spreading rotation rescues the 1.58-bit scheme the way it rescues
    int2 (sub-2-bit headline if it does)."""
    D = v.shape[-1]
    H = _hadamard(D, v.device, v.dtype)
    vr = (v.to(torch.float32) @ H.to(torch.float32)).to(v.dtype)
    vr_hat = ternary_fake_quant(vr, group_size)
    vhat = vr_hat.to(torch.float32) @ H.to(torch.float32)
    return vhat.to(v.dtype)


# Swappable value quantizer so the SAME cache path can test any V-compression
# method (ternary, Hadamard-int2, ...) with identical measurement.
_VALUE_QUANTIZER = ternary_fake_quant


def set_value_quantizer(fn):
    """fn(v, group_size) -> dequantized fp16 V. None resets to ternary."""
    global _VALUE_QUANTIZER
    _VALUE_QUANTIZER = fn if fn is not None else ternary_fake_quant


# ----------------------------------------------------------------------
# Patched attention forward: KIVI K path (INT4) + ternary V path (fp16 sim).
# Mirrors LlamaAttention_KIVI.forward; only the V handling is changed.
# In the cache tuple, slot 4 (value_states_quant) holds the ternary-fp16 OLD
# values; value_scale/value_mn (slots 6,7) are unused (None) for V.
# ----------------------------------------------------------------------
def ternary_v_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[-1]
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # ---- unpack cache ----
        key_states_quant_trans = past_key_value[0]
        key_states_full = past_key_value[1]
        key_scale_trans = past_key_value[2]
        key_mn_trans = past_key_value[3]
        value_states_ternary = past_key_value[4]   # ternary-fp16 OLD values (our change)
        value_states_full = past_key_value[5]      # recent fp16 residual
        # slots 6,7 unused for ternary V

        # ===== K attention scores — UNCHANGED from KIVI (INT4 packed kernel) =====
        if key_states_quant_trans is not None:
            att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans,
                                               key_scale_trans, key_mn_trans, self.k_bits)
        else:
            att_qkquant = None
        if key_states_full is not None:
            key_states_full = torch.cat([key_states_full, key_states], dim=2)
        else:
            key_states_full = key_states
        att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))
        if att_qkquant is not None:
            attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
        else:
            attn_weights = att_qkfull / math.sqrt(self.head_dim)

        if key_states_full.shape[-2] == self.residual_length:
            assert self.residual_length % self.group_size == 0
            key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = \
                triton_quantize_and_pack_along_last_dim(
                    key_states_full.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
            key_states_full = None
            if key_states_quant_trans is not None:
                key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
            else:
                key_states_quant_trans = key_states_quant_trans_new
                key_scale_trans = key_scale_trans_new
                key_mn_trans = key_mn_trans_new

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # ===== V attention — TERNARY (our change): fp16 matmul, no packed kernel =====
        # v_res: the fp16 shield for V, DECOUPLED from K's residual_length so we
        # can stress-test V quality (shrink v_res -> more/all recent V is ternary)
        # while K stays byte-identical to stock KIVI (group_size | residual_length).
        v_res = getattr(self, "v_residual_length", self.residual_length)
        value_states_full = torch.cat([value_states_full, value_states], dim=2)
        value_full_length = value_states_full.shape[-2]
        if value_states_ternary is None:
            attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))
        else:
            n_old = value_states_ternary.shape[-2]
            attn_output = torch.matmul(
                attn_weights[:, :, :, :n_old],
                repeat_kv(value_states_ternary, self.num_key_value_groups))
            attn_output += torch.matmul(
                attn_weights[:, :, :, n_old:],
                repeat_kv(value_states_full, self.num_key_value_groups))
        attn_output = attn_output.transpose(1, 2).contiguous()

        # evict oldest residual token -> ternary store (fake-quant to fp16)
        if value_full_length > v_res:
            assert value_full_length == v_res + 1
            evict = _VALUE_QUANTIZER(value_states_full[:, :, :1, :].contiguous(), self.group_size)
            value_states_full = value_states_full[:, :, 1:, :].contiguous()
            if value_states_ternary is not None:
                value_states_ternary = torch.cat([value_states_ternary, evict], dim=2)
            else:
                value_states_ternary = evict

    else:
        # ===== PREFILL ===== flash over full fp16 (same as KIVI)
        attn_output = self._flash_attention_forward(
            query_states.transpose(1, 2), key_states.transpose(1, 2),
            value_states.transpose(1, 2), None, q_len, dropout=0.0)

        # K cache: KIVI INT4 (unchanged)
        if key_states.shape[-2] % self.residual_length != 0:
            if key_states.shape[-2] < self.residual_length:
                key_states_quant = None
                key_states_full = key_states
            else:
                key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
        else:
            key_states_quant = key_states
            key_states_full = None
        if key_states_quant is not None:
            key_states_quant_trans, key_scale_trans, key_mn_trans = \
                triton_quantize_and_pack_along_last_dim(
                    key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
        else:
            key_states_quant_trans = key_scale_trans = key_mn_trans = None

        # V cache: TERNARY (our change) — old tokens fake-quantized to fp16.
        # v_res is V's fp16 shield (decoupled from K). v_res<=0 -> ALL V ternary
        # (guarded: `[:, :, :-0, :]` would be an EMPTY slice, a Python -0 footgun).
        v_res = getattr(self, "v_residual_length", self.residual_length)
        if v_res <= 0:
            value_states_ternary = _VALUE_QUANTIZER(value_states.contiguous(), self.group_size)
            value_states_full = value_states[:, :, :0, :].contiguous()   # empty shield
        elif value_states.shape[-2] <= v_res:
            value_states_ternary = None
            value_states_full = value_states
        else:
            old = value_states[:, :, :-v_res, :].contiguous()
            value_states_full = value_states[:, :, -v_res:, :].contiguous()
            value_states_ternary = _VALUE_QUANTIZER(old, self.group_size)

    # cache tuple: V scale/mn slots (6,7) are None for ternary
    past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans,
                      value_states_ternary, value_states_full, None, None, kv_seq_len) if use_cache else None

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def patch_ternary_v(model, v_residual_length=None):
    """Swap every attention module's forward for the ternary-V version.
    Returns the count patched. KIVI attention modules already carry every attr
    the patched forward needs (group_size, k_bits, residual_length, rotary_emb,
    _flash_attention_forward).

    v_residual_length: fp16 shield for the V cache, decoupled from K's
    residual_length. None -> use each module's residual_length (=stock behavior).
    Lower it (e.g. 8, 0) to ternary-quantize more/all recent V tokens — the
    quality stress test. K path is untouched regardless."""
    n = 0
    for module in model.modules():
        # KIVI attention modules expose these attrs + _flash_attention_forward
        if hasattr(module, "k_bits") and hasattr(module, "_flash_attention_forward"):
            module.forward = types.MethodType(ternary_v_forward, module)
            if v_residual_length is not None:
                module.v_residual_length = v_residual_length
            n += 1
    return n
