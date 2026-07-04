#!/usr/bin/env python3
"""
B2 — REAL packed 2-bit Hadamard-INT2 value cache (no more fp16 simulation).

B1/robust_sweep proved the QUALITY claim with fake-quant (quantize->dequantize,
stored fp16): Hadamard-INT2 V matches KIVI INT4 V within ~1-2% across two
models. But fake-quant stores fp16, so the MEMORY win was arithmetic, not
measurement. This file closes that gap:

  - V codes are stored as REAL packed 2-bit (4 codes per uint8 byte) in the
    cache, in Hadamard-rotated space, with per-group fp16 scale/min.
  - Dequantize-on-read with plain torch ops (unpack -> dequant -> matmul).
    No custom CUDA kernel; slower than a fused kernel would be, but every
    byte in the cache is real, so `measure_kv_memory` reports true storage.
  - Rotation trick: attention is computed IN rotated space and rotated back
    once per step —  A @ (Vr H) == (A @ Vr) H  because H is linear. So the
    stored cache never needs un-rotating token-by-token.

V bytes/token/head (head_dim 128, group 32):
  fp16: 256   KIVI int4: 64 + 16 scale/mn = 80   packed int2: 32 + 16 = 48
  -> V cache 1.67x smaller than KIVI, 5.3x smaller than fp16 (K stays KIVI int4).

Run:  python b2_packed_int2.py          (loads kivi, measures both, compares)
"""
import datetime
import json
import math
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb

from ternary_v import _hadamard

# ----------------------------------------------------------------------
# 2-bit pack / unpack (4 codes per byte) + rotated-space quantization
# ----------------------------------------------------------------------
def pack_int2(q):
    """q: uint8 [..., D] with values 0..3, D % 4 == 0 -> uint8 [..., D/4]."""
    q = q.reshape(*q.shape[:-1], q.shape[-1] // 4, 4)
    return (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).contiguous()


def unpack_int2(p, D):
    """p: uint8 [..., D/4] -> uint8 [..., D]."""
    out = torch.stack([(p >> 0) & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], dim=-1)
    return out.reshape(*p.shape[:-1], D)


def quant_pack_rotated(v, group_size):
    """v: fp16 [b,h,n,D] -> (packed uint8 [b,h,n,D/4], scale fp16 [b,h,n,g],
    mn fp16 [b,h,n,g]) in Hadamard-rotated space. Same asymmetric-uniform math
    as the validated fake-quant (hadamard_int_fake_quant, bits=2)."""
    D = v.shape[-1]
    H = _hadamard(D, v.device, torch.float32)
    vr = v.to(torch.float32) @ H
    g = D // group_size
    x = vr.reshape(*vr.shape[:-1], g, group_size)
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / 3.0).clamp(min=1e-8)          # 4 levels -> /(2^2 - 1)
    q = torch.round((x - mn) / scale).clamp_(0, 3).to(torch.uint8)
    q = q.reshape(*vr.shape[:-1], D)
    return pack_int2(q), scale.squeeze(-1).to(torch.float16), mn.squeeze(-1).to(torch.float16)


def dequant_rotated(packed, scale, mn, group_size):
    """-> fp16 [b,h,n,D] STILL IN ROTATED SPACE (caller rotates the attention
    output back once — cheaper than un-rotating every cached token)."""
    D = packed.shape[-1] * 4
    q = unpack_int2(packed, D).to(torch.float16)
    g = D // group_size
    x = q.reshape(*q.shape[:-1], g, group_size)
    vr = x * scale.unsqueeze(-1) + mn.unsqueeze(-1)
    return vr.reshape(*q.shape[:-1], D)


# ----------------------------------------------------------------------
# Attention forward: KIVI INT4 K path (unchanged) + PACKED INT2 rotated V.
# Cache tuple: (k_quant_t, k_full, k_scale_t, k_mn_t,
#               v_packed, v_full, v_scale, v_mn, kv_seq_len)
# ----------------------------------------------------------------------
def packed_int2_v_forward(
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

    v_res = getattr(self, "v_residual_length", self.residual_length)
    D = self.head_dim
    Hrot = _hadamard(D, hidden_states.device, torch.float32)

    if past_key_value is not None:
        key_states_quant_trans = past_key_value[0]
        key_states_full = past_key_value[1]
        key_scale_trans = past_key_value[2]
        key_mn_trans = past_key_value[3]
        v_packed = past_key_value[4]      # uint8 [b,h,n,D/4] — REAL 2-bit storage
        value_states_full = past_key_value[5]
        v_scale = past_key_value[6]       # fp16 [b,h,n,g]
        v_mn = past_key_value[7]          # fp16 [b,h,n,g]

        # ===== K path — byte-for-byte KIVI INT4 (unchanged) =====
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

        # ===== V path — REAL packed 2-bit, computed in rotated space =====
        value_states_full = torch.cat([value_states_full, value_states], dim=2)
        value_full_length = value_states_full.shape[-2]
        if v_packed is None:
            attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))
        else:
            n_old = v_packed.shape[-2]
            vr_hat = dequant_rotated(v_packed, v_scale, v_mn, self.group_size)  # rotated fp16
            # A @ (Vr H) == (A @ Vr) H — rotate ONCE after the weighted sum
            out_rot = torch.matmul(attn_weights[:, :, :, :n_old],
                                   repeat_kv(vr_hat, self.num_key_value_groups))
            attn_output = (out_rot.to(torch.float32) @ Hrot).to(value_states_full.dtype)
            attn_output += torch.matmul(attn_weights[:, :, :, n_old:],
                                        repeat_kv(value_states_full, self.num_key_value_groups))
            del vr_hat
        attn_output = attn_output.transpose(1, 2).contiguous()

        # evict oldest residual token -> packed 2-bit store
        if value_full_length > v_res:
            assert value_full_length == v_res + 1
            p_new, s_new, m_new = quant_pack_rotated(
                value_states_full[:, :, :1, :].contiguous(), self.group_size)
            value_states_full = value_states_full[:, :, 1:, :].contiguous()
            if v_packed is not None:
                v_packed = torch.cat([v_packed, p_new], dim=2)
                v_scale = torch.cat([v_scale, s_new], dim=2)
                v_mn = torch.cat([v_mn, m_new], dim=2)
            else:
                v_packed, v_scale, v_mn = p_new, s_new, m_new

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

        # V cache: REAL packed 2-bit rotated (guard the -0 slice footgun)
        if v_res <= 0:
            v_packed, v_scale, v_mn = quant_pack_rotated(value_states.contiguous(), self.group_size)
            value_states_full = value_states[:, :, :0, :].contiguous()
        elif value_states.shape[-2] <= v_res:
            v_packed = v_scale = v_mn = None
            value_states_full = value_states
        else:
            old = value_states[:, :, :-v_res, :].contiguous()
            value_states_full = value_states[:, :, -v_res:, :].contiguous()
            v_packed, v_scale, v_mn = quant_pack_rotated(old, self.group_size)

    past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans,
                      v_packed, value_states_full, v_scale, v_mn, kv_seq_len) if use_cache else None

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def patch_packed_int2_v(model, v_residual_length=None):
    import types
    n = 0
    for module in model.modules():
        if hasattr(module, "k_bits") and hasattr(module, "_flash_attention_forward"):
            module.forward = types.MethodType(packed_int2_v_forward, module)
            if v_residual_length is not None:
                module.v_residual_length = v_residual_length
            n += 1
    return n


# ----------------------------------------------------------------------
# THE FINAL TEST: kivi vs packed-int2 on the same model load path —
# quality (cache-path PPL), REAL memory (allocator bytes), decode speed.
# ----------------------------------------------------------------------
def main():
    from phase2_kivi_harness import load_kivi, CONFIG
    from fp16_baseline_harness import free_cuda, measure_kv_memory, measure_decode_tps
    from probe_cache_ppl import get_passage, ppl_cache, PREFILL

    results = {"timestamp": datetime.datetime.now().isoformat()}

    print("=== loading kivi (NF4 weights + KIVI INT4 KV) ===", flush=True)
    model, tok, _ = load_kivi()
    ids = get_passage(tok)

    print("\n--- KIVI reference ---", flush=True)
    kivi_ppl = ppl_cache(model, ids)
    print(f"kivi PPL(cache,896)   = {kivi_ppl:.4f}", flush=True)
    free_cuda()
    kivi_kv = measure_kv_memory(model, tok, 2048)
    print(f"kivi KV MB @2048      = {kivi_kv['empirical_mb']:.1f}", flush=True)
    free_cuda()
    kivi_tps = measure_decode_tps(model, tok)
    print(f"kivi decode tok/s     = {kivi_tps['tokens_per_sec']:.1f}", flush=True)
    free_cuda()
    results["kivi"] = {"ppl_896": kivi_ppl, "kv_mb_2048": kivi_kv["empirical_mb"],
                       "decode_tps": kivi_tps["tokens_per_sec"]}

    print("\n--- PACKED Hadamard-INT2 V (B2, real 2-bit storage) ---", flush=True)
    n = patch_packed_int2_v(model)  # v_res defaults to residual_length=32
    print(f"patched {n} attention modules", flush=True)
    b2_ppl = ppl_cache(model, ids)
    print(f"b2 PPL(cache,896)     = {b2_ppl:.4f}   (fake-quant B1 was 3.6548)", flush=True)
    free_cuda()
    b2_kv = measure_kv_memory(model, tok, 2048)
    print(f"b2 KV MB @2048        = {b2_kv['empirical_mb']:.1f}", flush=True)
    free_cuda()
    b2_tps = measure_decode_tps(model, tok)
    print(f"b2 decode tok/s       = {b2_tps['tokens_per_sec']:.1f}", flush=True)
    free_cuda()
    results["b2_packed_int2"] = {"ppl_896": b2_ppl, "kv_mb_2048": b2_kv["empirical_mb"],
                                 "decode_tps": b2_tps["tokens_per_sec"]}

    print("\n" + "=" * 62)
    print(f"{'':<22}{'kivi int4':>12}{'b2 packed-int2':>16}")
    print("-" * 62)
    print(f"{'PPL (cache, 896)':<22}{kivi_ppl:>12.4f}{b2_ppl:>16.4f}")
    print(f"{'KV MB @2048 (real)':<22}{kivi_kv['empirical_mb']:>12.1f}{b2_kv['empirical_mb']:>16.1f}")
    print(f"{'decode tok/s':<22}{kivi_tps['tokens_per_sec']:>12.1f}{b2_tps['tokens_per_sec']:>16.1f}")
    print("=" * 62)
    print("Read: PPL equal => quality claim survives REAL packing.")
    print("      KV MB drop => the memory win, now MEASURED not simulated.")

    out = os.path.join(CONFIG["output_dir"],
                       f"b2_packed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
