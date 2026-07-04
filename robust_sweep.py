#!/usr/bin/env python3
"""
Robustness sweep — is "hadamard-int2 V matches KIVI int4 V" real, or a fluke
of one passage / one length / one model?

Everything goes through the REAL cache path (prefill + token-by-token decode,
use_cache=True), i.e. the path generation actually uses — the corrected ruler
from probe_cache_ppl.py.

Axes:
  passages : 3 disjoint WikiText-103 slices (different text, same domain)
  lengths  : up to near the model's context cap (Llama-2: 4096, TinyLlama: 2048)
  schemes  : kivi int4-V (reference) | ternary | hadamard-int2 | hadamard-ternary
             (v_res=0 = NO fp16 shield = hardest condition, plus v_res=32 for
              the headline hadamard-int2 config)
  models   : Llama-2-7B (main), TinyLlama-1.1B (second llama-family point;
             may fail if KIVI's class rejects its GQA config — logged, not fatal)
  GSM8K    : kivi vs hadamard-int2 v_res=32, n=20, 7B only. generate() uses the
             cache, so this is an HONEST quality check (unlike the old PPL).

Run (KIVI venv, WSL):  python robust_sweep.py [--fast]
Writes baseline_results/robust_sweep_<ts>.json and prints a table at the end.
"""
import argparse
import datetime
import functools
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from phase2_kivi_harness import load_kivi, CONFIG, KIVI_CONFIG
from fp16_baseline_harness import free_cuda, measure_gsm8k
from ternary_v import (patch_ternary_v, set_value_quantizer, ternary_fake_quant,
                       hadamard_int_fake_quant, hadamard_ternary_fake_quant)

PREFILL = 128
DEVICE = CONFIG["device"]

SCHEMES = {
    # label                  quantizer fn                                       bits
    "ternary v0":            (ternary_fake_quant,                               1.58),
    "had-int2 v32":          (functools.partial(hadamard_int_fake_quant, bits=2), 2),
    "had-int2 v0":           (functools.partial(hadamard_int_fake_quant, bits=2), 2),
    "had-ternary v0":        (hadamard_ternary_fake_quant,                      1.58),
}
V_RES = {"ternary v0": 0, "had-int2 v32": 32, "had-int2 v0": 0, "had-ternary v0": 0}


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_stream(tokenizer):
    from datasets import load_dataset
    ds = load_dataset(CONFIG["wikitext_dataset"], CONFIG["wikitext_config"],
                      split=CONFIG["wikitext_split"])
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt").input_ids  # [1, ~280k]


@torch.no_grad()
def ppl_no_cache(model, ids):
    """Single forward, use_cache=False — the OLD (blind) method, kept as the
    per-passage reference so the cache numbers have an anchor."""
    N = ids.size(1)
    out = model(ids.to(DEVICE), use_cache=False)
    logits = out.logits[0, PREFILL - 1:N - 1, :].float()
    tgts = ids[0, PREFILL:N].to(DEVICE)
    nll = F.cross_entropy(logits, tgts, reduction="sum").item()
    del out, logits
    return math.exp(nll / (N - PREFILL))


@torch.no_grad()
def ppl_cache(model, ids):
    """Prefill + one-token-at-a-time decode over the quantized cache (real)."""
    ids = ids.to(DEVICE)
    N = ids.size(1)
    pos = torch.arange(0, PREFILL, device=DEVICE).unsqueeze(0)
    out = model(ids[:, :PREFILL], position_ids=pos, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :].float()
    nll, t0 = 0.0, time.time()
    for t in range(PREFILL, N):
        nll += F.cross_entropy(logits, ids[0, t].unsqueeze(0), reduction="sum").item()
        if t + 1 >= N:
            break
        out = model(ids[:, t:t + 1], past_key_values=past,
                    position_ids=torch.tensor([[t]], device=DEVICE), use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()
        if (t - PREFILL + 1) % 512 == 0:
            r = (t - PREFILL + 1) / (time.time() - t0)
            log(f"      ... {t - PREFILL + 1}/{N - PREFILL} tok ({r:.1f}/s)")
    del past, out
    return math.exp(nll / (N - PREFILL - 1))


def set_v_res(model, r):
    for m in model.modules():
        if hasattr(m, "k_bits") and hasattr(m, "_flash_attention_forward"):
            m.v_residual_length = r


def save_original_forwards(model):
    return {id(m): m.forward for m in model.modules()
            if hasattr(m, "k_bits") and hasattr(m, "_flash_attention_forward")}


def run_model(model_name, lengths, passage_offsets, results, do_gsm8k, fast):
    CONFIG["model_name"] = model_name
    log(f"=== MODEL {model_name} | lengths={lengths} | offsets={passage_offsets} ===")
    try:
        model, tok, meta = load_kivi()
    except Exception as e:
        log(f"!! load failed for {model_name}: {e}")
        results[model_name] = {"load_error": str(e)}
        return
    stream = get_stream(tok)
    passages = {}
    for pi, off in enumerate(passage_offsets):
        passages[f"P{pi}(@{off // 1000}k)"] = {
            N: stream[:, off:off + N] for N in lengths if off + N <= stream.size(1)
        }

    mres = {"scheme_bits": {k: v[1] for k, v in SCHEMES.items()}, "rows": []}
    results[model_name] = mres

    def add(scheme, passage, N, method, ppl):
        mres["rows"].append({"scheme": scheme, "passage": passage, "N": N,
                             "method": method, "ppl": round(ppl, 4)})
        log(f"  {scheme:<16} {passage:<10} N={N:<5} {method:<8} PPL={ppl:.4f}")

    # ---- 1. kivi int4-V (UNPATCHED forward) — reference, cache + no-cache ----
    for pname, byN in passages.items():
        for N, ids in byN.items():
            add("kivi int4 v32", pname, N, "no-cache", ppl_no_cache(model, ids)); free_cuda()
            add("kivi int4 v32", pname, N, "cache", ppl_cache(model, ids)); free_cuda()

    # honest GSM8K for kivi (generate() -> cache path) BEFORE patching
    if do_gsm8k:
        CONFIG["gsm8k_n"] = 10 if fast else 20
        log("  GSM8K kivi ...")
        g = measure_gsm8k(model, tok)
        mres["gsm8k_kivi"] = g["accuracy"]
        log(f"  GSM8K kivi acc={g['accuracy']:.3f}"); free_cuda()

    # ---- 2. patch once; sweep quantizer schemes over the same passages ----
    n = patch_ternary_v(model)
    log(f"  patched {n} attention modules")
    for scheme, (qfn, bits) in SCHEMES.items():
        set_value_quantizer(qfn)
        set_v_res(model, V_RES[scheme])
        for pname, byN in passages.items():
            for N, ids in byN.items():
                add(scheme, pname, N, "cache", ppl_cache(model, ids)); free_cuda()

    if do_gsm8k:
        set_value_quantizer(functools.partial(hadamard_int_fake_quant, bits=2))
        set_v_res(model, 32)
        log("  GSM8K had-int2 v32 ...")
        g = measure_gsm8k(model, tok)
        mres["gsm8k_had_int2_v32"] = g["accuracy"]
        log(f"  GSM8K had-int2 acc={g['accuracy']:.3f}"); free_cuda()

    del model
    free_cuda()


def print_table(results):
    for model_name, mres in results.items():
        if "rows" not in mres:
            print(f"\n{model_name}: LOAD FAILED: {mres.get('load_error')}")
            continue
        rows = mres["rows"]
        schemes = list(dict.fromkeys(r["scheme"] + "/" + r["method"] for r in rows))
        cells = {}
        for r in rows:
            cells[(r["scheme"] + "/" + r["method"], r["passage"], r["N"])] = r["ppl"]
        pns = sorted(set((r["passage"], r["N"]) for r in rows))
        print(f"\n=== {model_name} — PPL through the REAL cache path ===")
        hdr = f"{'scheme/method':<26}" + "".join(f"{p}@{n:<6}"[:14].ljust(14) for p, n in pns)
        print(hdr); print("-" * len(hdr))
        for s in schemes:
            line = f"{s:<26}"
            for p, n in pns:
                v = cells.get((s, p, n))
                line += (f"{v:<14.4f}" if v is not None else f"{'-':<14}")
            print(line)
        for k in ("gsm8k_kivi", "gsm8k_had_int2_v32"):
            if k in mres:
                print(f"{k}: {mres[k]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="1 passage, short lengths (smoke)")
    ap.add_argument("--skip-tinyllama", action="store_true")
    args = ap.parse_args()

    results = {}
    t0 = time.time()

    if args.fast:
        run_model("NousResearch/Llama-2-7b-hf", [896], [0], results,
                  do_gsm8k=False, fast=True)
    else:
        # Llama-2-7B: 3 passages x {896, 2048}; long 3584 on passage 0 only
        # (window cap 4096; 3584 + headroom stays legal).
        run_model("NousResearch/Llama-2-7b-hf", [896, 2048], [0, 60000, 120000],
                  results, do_gsm8k=True, fast=False)
        # long-context single-passage add-on, reuse the same JSON structure
        CONFIG["model_name"] = "NousResearch/Llama-2-7b-hf"
        log("=== long-context add-on (N=3584, P0) ===")
        model, tok, _ = load_kivi()
        stream = get_stream(tok)
        ids = stream[:, :3584]
        r = results["NousResearch/Llama-2-7b-hf"]
        r["rows"].append({"scheme": "kivi int4 v32", "passage": "P0(@0k)", "N": 3584,
                          "method": "cache", "ppl": round(ppl_cache(model, ids), 4)})
        log(f"  kivi long done"); free_cuda()
        patch_ternary_v(model)
        for scheme in ("had-int2 v0", "ternary v0"):
            set_value_quantizer(SCHEMES[scheme][0]); set_v_res(model, V_RES[scheme])
            r["rows"].append({"scheme": scheme, "passage": "P0(@0k)", "N": 3584,
                              "method": "cache", "ppl": round(ppl_cache(model, ids), 4)})
            log(f"  {scheme} long done"); free_cuda()
        del model; free_cuda()

        if not args.skip_tinyllama:
            # second model point — llama-family, GQA. May not load under KIVI.
            run_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", [896, 1792], [0, 60000],
                      results, do_gsm8k=False, fast=False)

    out = os.path.join(CONFIG["output_dir"],
                       f"robust_sweep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"saved -> {out} ({(time.time() - t0) / 60:.1f} min total)")
    print_table(results)


if __name__ == "__main__":
    main()
