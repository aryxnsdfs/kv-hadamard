#!/usr/bin/env python3
"""
Resume the robustness sweep after a WSL restart killed it mid-run.

Already captured (robust_sweep_run2.log, Llama-2-7B): full main sweep -
kivi/ternary/had-int2(v32,v0)/had-ternary x 3 passages x {896,2048}, plus
GSM8K for kivi and had-int2 v32 (both 0/20 - confirmed a real base-model
0-shot floor by reading transcripts, not a harness bug). Long-context add-on
got through kivi @3584 only before WSL died.

Missing, and all this script does:
  1. Long-context (N=3584, P0) for had-int2 v0 and ternary v0 (kivi already done)
  2. TinyLlama-1.1B second-model pass (896, 1792 x 2 offsets, all 4 schemes)

SAVES AFTER EVERY ROW (not just at the end) so a future WSL death loses at
most one measurement, not two hours.
"""
import datetime
import functools
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from phase2_kivi_harness import load_kivi, CONFIG
from fp16_baseline_harness import free_cuda
from ternary_v import (patch_ternary_v, set_value_quantizer, ternary_fake_quant,
                       hadamard_int_fake_quant, hadamard_ternary_fake_quant)

PREFILL = 128
DEVICE = CONFIG["device"]
OUT_PATH = os.path.join(CONFIG["output_dir"], "robust_sweep_resumed.json")


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            return json.load(f)
    return {"rows": [], "gsm8k": {}}


def save_row(state, row):
    state["rows"].append(row)
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(state, f, indent=2)
    log(f"  {row['model']:<20}{row['scheme']:<16}{row['passage']:<10}N={row['N']:<5}"
        f"{row['method']:<8}PPL={row['ppl']:.4f}  [saved]")


def already_done(state, model, scheme, passage, N, method):
    return any(r["model"] == model and r["scheme"] == scheme and r["passage"] == passage
              and r["N"] == N and r["method"] == method for r in state["rows"])


def get_stream(tokenizer):
    from datasets import load_dataset
    ds = load_dataset(CONFIG["wikitext_dataset"], CONFIG["wikitext_config"],
                      split=CONFIG["wikitext_split"])
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt").input_ids


@torch.no_grad()
def ppl_no_cache(model, ids):
    N = ids.size(1)
    out = model(ids.to(DEVICE), use_cache=False)
    logits = out.logits[0, PREFILL - 1:N - 1, :].float()
    tgts = ids[0, PREFILL:N].to(DEVICE)
    nll = F.cross_entropy(logits, tgts, reduction="sum").item()
    del out, logits
    return math.exp(nll / (N - PREFILL))


@torch.no_grad()
def ppl_cache(model, ids):
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


SCHEMES = {
    "ternary v0": (ternary_fake_quant, 0),
    "had-int2 v0": (functools.partial(hadamard_int_fake_quant, bits=2), 0),
}


def part1_long_context(state):
    """N=3584, P0, Llama-2-7B: kivi already have it from run2's log (3.??? -
    recorded manually below since the JSON output never got written for it).
    Only had-int2 v0 and ternary v0 remain."""
    model_name = "NousResearch/Llama-2-7b-hf"
    CONFIG["model_name"] = model_name

    # kivi's 3584 result was computed in the crashed run but never saved to
    # JSON (process died before _save). Recorded here as a known value so the
    # comparison table is complete; recompute is ~6min and cheap to verify.
    if not already_done(state, model_name, "kivi int4 v32", "P0(@0k)", 3584, "cache"):
        log("=== long-context: kivi @3584 (recomputing, wasn't saved before crash) ===")
        model, tok, _ = load_kivi()
        stream = get_stream(tok)
        ids = stream[:, :3584]
        ppl = ppl_cache(model, ids)
        save_row(state, {"model": model_name, "scheme": "kivi int4 v32",
                         "passage": "P0(@0k)", "N": 3584, "method": "cache", "ppl": ppl})
        free_cuda()
        n = patch_ternary_v(model)
        log(f"  patched {n} modules")
    else:
        log("=== long-context: kivi @3584 already done, loading model for the rest ===")
        model, tok, _ = load_kivi()
        stream = get_stream(tok)
        ids = stream[:, :3584]
        patch_ternary_v(model)

    for scheme, (qfn, vres) in SCHEMES.items():
        if already_done(state, model_name, scheme, "P0(@0k)", 3584, "cache"):
            log(f"  {scheme} @3584 already done, skip")
            continue
        set_value_quantizer(qfn)
        set_v_res(model, vres)
        log(f"=== long-context: {scheme} @3584 ===")
        ppl = ppl_cache(model, ids)
        save_row(state, {"model": model_name, "scheme": scheme, "passage": "P0(@0k)",
                         "N": 3584, "method": "cache", "ppl": ppl})
        free_cuda()

    del model
    free_cuda()


def part2_tinyllama(state):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    CONFIG["model_name"] = model_name
    lengths = [896, 1792]
    offsets = [0, 60000]

    try:
        model, tok, _ = load_kivi()
    except Exception as e:
        log(f"!! TinyLlama load failed under KIVI class: {e}")
        state["tinyllama_load_error"] = str(e)
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(state, f, indent=2)
        return

    stream = get_stream(tok)
    passages = {f"P{i}(@{off // 1000}k)": {N: stream[:, off:off + N] for N in lengths
                                            if off + N <= stream.size(1)}
               for i, off in enumerate(offsets)}

    # kivi reference (unpatched)
    for pname, byN in passages.items():
        for N, ids in byN.items():
            for method, fn in (("no-cache", ppl_no_cache), ("cache", ppl_cache)):
                if already_done(state, model_name, "kivi int4 v32", pname, N, method):
                    continue
                ppl = fn(model, ids)
                save_row(state, {"model": model_name, "scheme": "kivi int4 v32",
                                 "passage": pname, "N": N, "method": method, "ppl": ppl})
                free_cuda()

    n = patch_ternary_v(model)
    log(f"  patched {n} modules for TinyLlama")

    all_schemes = {
        "ternary v0": (ternary_fake_quant, 0),
        "had-int2 v32": (functools.partial(hadamard_int_fake_quant, bits=2), 32),
        "had-int2 v0": (functools.partial(hadamard_int_fake_quant, bits=2), 0),
        "had-ternary v0": (hadamard_ternary_fake_quant, 0),
    }
    for scheme, (qfn, vres) in all_schemes.items():
        set_value_quantizer(qfn)
        set_v_res(model, vres)
        for pname, byN in passages.items():
            for N, ids in byN.items():
                if already_done(state, model_name, scheme, pname, N, "cache"):
                    continue
                ppl = ppl_cache(model, ids)
                save_row(state, {"model": model_name, "scheme": scheme, "passage": pname,
                                 "N": N, "method": "cache", "ppl": ppl})
                free_cuda()

    del model
    free_cuda()


def main():
    state = load_state()
    log(f"resuming: {len(state['rows'])} rows already saved in {OUT_PATH}")
    part1_long_context(state)
    part2_tinyllama(state)
    log(f"DONE. {len(state['rows'])} total rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
