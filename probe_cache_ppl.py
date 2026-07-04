#!/usr/bin/env python3
"""
Probe: does PPL actually SEE the V-quant? (the honest ruler check)

Background: phase2's measure_perplexity runs `model.model(ids, use_cache=False)`
— a single full forward with NO KV cache. The KIVI/ternary quant only happens on
the CACHE path (prefill-build + decode), so use_cache=False NEVER exercises it:
every scheme falls into the fp16 prefill branch. That is why fp16 == int4 ==
ternary PPL to 3 decimals — the metric didn't run the quant.

This probe measures next-token PPL TWO ways on the SAME passage, one model load:
  * no-cache : single forward, use_cache=False  (what phase2 did = fp16 V, BLIND)
  * cache    : prefill then decode one token at a time, use_cache=True
               (predictions attend over the QUANTIZED K/V cache = real)

Schemes (all share NF4 weights + stock KIVI int4 K; only V differs):
  kivi            : int4 V, residual 32     (the reference)
  ternary v_res=R : ternary V, fp16 shield of R recent tokens (R = 32, 8, 0)

If cache-PPL for ternary v_res=0 still ~= kivi cache-PPL, ternary V is genuinely
near-lossless even with NO fp16 shield -> strong claim. If it diverges, we found
the real quality boundary. Either way it's a number the old ruler could not see.

Run in the KIVI venv:  python probe_cache_ppl.py
"""
import math
import torch
import torch.nn.functional as F

import functools
from phase2_kivi_harness import load_kivi, CONFIG
from ternary_v import (patch_ternary_v, set_value_quantizer,
                       ternary_fake_quant, hadamard_int_fake_quant)

# ---- knobs (kept small so the token-by-token decode finishes in a few min) ----
N = 896          # total tokens in the scored passage
PREFILL = 128    # tokens prefilled before scoring begins
DEVICE = CONFIG["device"]


def get_passage(tokenizer):
    from datasets import load_dataset
    ds = load_dataset(CONFIG["wikitext_dataset"], CONFIG["wikitext_config"],
                      split=CONFIG["wikitext_split"])
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids
    return ids[:, :N].to(DEVICE)   # [1, N]


@torch.no_grad()
def ppl_no_cache(model, ids):
    """Single forward, use_cache=False — the phase2 method. Blind to the cache."""
    out = model(ids, use_cache=False)
    logits = out.logits[0, PREFILL - 1:N - 1, :].float()   # predicts tokens PREFILL..N-1
    tgts = ids[0, PREFILL:N]
    nll = F.cross_entropy(logits, tgts, reduction="sum").item()
    return math.exp(nll / (N - PREFILL))


@torch.no_grad()
def ppl_cache(model, ids):
    """Prefill then decode one token at a time — predictions attend over the
    QUANTIZED cache. This is what generation actually does; it exercises V-quant."""
    pos = torch.arange(0, PREFILL, device=DEVICE).unsqueeze(0)
    out = model(ids[:, :PREFILL], position_ids=pos, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :].float()   # predicts token at index PREFILL
    nll = 0.0
    for t in range(PREFILL, N):
        tgt = ids[0, t].unsqueeze(0)
        nll += F.cross_entropy(logits, tgt, reduction="sum").item()
        if t + 1 >= N:
            break
        pos = torch.tensor([[t]], device=DEVICE)
        out = model(ids[:, t:t + 1], past_key_values=past, position_ids=pos, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()   # predicts token t+1
    return math.exp(nll / (N - PREFILL - 1))


def set_v_res(model, r):
    n = 0
    for m in model.modules():
        if hasattr(m, "v_residual_length") or hasattr(m, "k_bits"):
            if hasattr(m, "k_bits") and hasattr(m, "_flash_attention_forward"):
                m.v_residual_length = r
                n += 1
    return n


def main():
    print(f"loading kivi (NF4 + int4 KV) on {CONFIG['model_name']} ...")
    model, tok, meta = load_kivi()
    ids = get_passage(tok)
    print(f"passage: N={N} tokens, prefill={PREFILL}, scoring {N-PREFILL} tokens\n")

    rows = []

    # --- kivi (int4 V), unpatched ---
    print("[kivi] no-cache (blind) ...")
    kivi_nc = ppl_no_cache(model, ids)
    print(f"       PPL = {kivi_nc:.4f}")
    print("[kivi] cache (real) ...")
    kivi_c = ppl_cache(model, ids)
    print(f"       PPL = {kivi_c:.4f}\n")
    rows.append(("kivi int4-V", "no-cache", kivi_nc))
    rows.append(("kivi int4-V", "cache", kivi_c))

    # --- patch once; forward is now the swappable-quantizer V path ---
    patch_ternary_v(model)
    tern_nc = ppl_no_cache(model, ids)   # should ~= kivi_nc (both blind = fp16 V)
    rows.append(("<any> V", "no-cache", tern_nc))

    # method -> quantizer fn(v, group_size). v_res=0 = NO fp16 shield (hardest).
    methods = [
        ("ternary (1.58b)", ternary_fake_quant),
        ("hadamard-int2 (2b)", functools.partial(hadamard_int_fake_quant, bits=2)),
    ]
    for label, qfn in methods:
        set_value_quantizer(qfn)
        for r in (32, 0):
            set_v_res(model, r)
            print(f"[{label} v_res={r}] cache (real) ...")
            c = ppl_cache(model, ids)
            print(f"       PPL = {c:.4f}")
            rows.append((f"{label} v_res={r}", "cache", c))

    print("\n" + "=" * 56)
    print(f"{'scheme':<22}{'method':<12}{'PPL':>10}")
    print("-" * 56)
    for name, method, ppl in rows:
        print(f"{name:<22}{method:<12}{ppl:>10.4f}")
    print("=" * 56)
    print("Read:")
    print(" - no-cache rows equal across schemes => PPL-as-run was BLIND to V.")
    print(" - kivi cache vs no-cache gap => quality cost int4-V hid from the old ruler.")
    print(" - ternary v_res=0 cache vs kivi cache => real ternary-V quality delta,")
    print("   with ZERO fp16 shield (the strongest stress test).")


if __name__ == "__main__":
    main()
