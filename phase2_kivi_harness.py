#!/usr/bin/env python3
"""
Phase 2 (pre-ternary) — three-baseline harness on Llama-2-7B.

Goal of THIS step: prove the whole pipeline runs end-to-end on the 7B model
that the real result lives on — model loads, KV cache quantizes, the Phase 1
harness measures it, and three baselines diff cleanly. NO ternary-V kernel yet;
that is Phase 2 proper. This step gets KIVI's STOCK 4-bit KV working on 7B-NF4
locally, plus the two reference baselines, all measurable on the same harness.

THREE BASELINES (selectable):
  fp16 : FP16 weights + FP16 KV    -> ~14GB weights, WON'T FIT a 12GB 3060.
                                      CLOUD-ONLY. This is the eventual headline
                                      reference; locally it is expected to fail
                                      to load and is logged as such.
  nf4  : NF4 weights + FP16 KV     -> local. Isolates the weight-quant quality
                                      hit from the KV contribution.
  kivi : NF4 weights + KIVI 4-bit KV -> local. The method (stock KIVI INT4 KV).

The Phase 1 harness is reused UNCHANGED (imported). Only CONFIG values are
overridden here and three loaders are added — the metric functions, JSON
schema, OOM handling and KV-isolation are identical, so the three runs diff
directly.

RUN (in the KIVI venv from setup_phase2.sh — transformers 4.37 supports BOTH
Llama-2 and KIVI, so all three baselines share one environment = clean parity):
    python3 phase2_kivi_harness.py --baseline nf4
    python3 phase2_kivi_harness.py --baseline kivi
    python3 phase2_kivi_harness.py --baseline fp16     # cloud / big-GPU only
    python3 phase2_kivi_harness.py --compare           # diff all three JSONs

READ PHASE2_README.md.
"""

import os
import json
import glob
import argparse
import datetime

import torch

# Reuse Phase 1 wholesale. Do not re-implement metrics.
from fp16_baseline_harness import (
    CONFIG, log, free_cuda, is_oom_error, _timestamp,
    measure_perplexity, measure_gsm8k, measure_decode_tps, measure_kv_memory,
)

# ----------------------------------------------------------------------
# CONFIG OVERRIDES — applied to the imported CONFIG dict so the Phase 1
# FILE is never edited. Switch the experiment to Llama-2-7B.
# ----------------------------------------------------------------------
# Ungated mirror of meta-llama/Llama-2-7b-hf (same weights/config). Swap to the
# Meta repo once you have gated access; numbers are equivalent.
CONFIG["model_name"] = "NousResearch/Llama-2-7b-hf"
# 7B KV is huge in FP16 (~2GB @2048 ... ~17GB @16384), so the long contexts are
# where NF4+FP16-KV OOMs and KIVI is meant to survive. Keep all four to record it.
CONFIG["context_lengths"] = [2048, 4096, 8192, 16384]
# GSM8K on 7B-NF4 is slow (~33s/problem, no flash) and a long run is exposed to
# transient WSL GPU faults. 40 is plenty for a quality-preservation GUARD (PPL
# is the primary quality metric). Same N applies to nf4 AND kivi -> comparable.
CONFIG["gsm8k_n"] = 40

# fp16 (Half). KIVI's CUDA gemv kernel ONLY accepts Half, not BFloat16
# ("expected scalar type Half but found BFloat16"), so KIVI mandates fp16.
# fp16 overflows -> nan only at 8192/16384, but those EXCEED Llama-2's 4096
# context window so their PPL is meaningless regardless of dtype (kept only for
# the KV-memory numbers). Within the valid range (<=4096) fp16 is clean
# (2048=4.99, 4096=4.89). So fp16 is both required by KIVI and correct where
# quality actually matters.
CONFIG["dtype"] = torch.float16

# Shrink the PPL eval token budget for 7B. Without a flash kernel, attention is
# math O(seq^2) on a 7B model -> 16384-ctx PPL over 100k tokens is ~45 min/run
# and exposed to transient WSL GPU faults. 30k tokens (~58 windows) gives a
# stable PPL while cutting runtime ~3.4x. nf4 AND kivi use the same budget, so
# the comparison stays valid. (Phase 1's 1B baseline used 100k; that's a
# different model/run and not directly compared to these 7B numbers.)
CONFIG["wikitext_max_eval_tokens"] = 30000

# KIVI 4-bit KV knobs (stock KIVI — NOT ternary).
# use_flash MUST be True: KIVI asserts it and has no non-flash path
# ("currently KIVI is only available for flash-attn"). Requires flash-attn
# built in the venv (see setup_phase2.sh step 7).
KIVI_CONFIG = {
    "k_bits": 4, "v_bits": 4, "group_size": 32, "residual_length": 32,
    "use_flash": True,
    "kivi_module": "models.llama_kivi",
    "kivi_class": "LlamaForCausalLM_KIVI",
}


# ----------------------------------------------------------------------
# Shared NF4 weight-quant config (bitsandbytes). Used by both nf4 and kivi.
# ----------------------------------------------------------------------
def _nf4_bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=CONFIG["dtype"],   # fp16 compute
        bnb_4bit_use_double_quant=True,
    )


def _common_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ----------------------------------------------------------------------
# THE THREE LOADERS. Each returns (model, tokenizer, meta_dict).
# ----------------------------------------------------------------------
def load_fp16():
    """FP16 weights + FP16 KV. ~14GB weights — WILL NOT FIT a 12GB 3060.
    Kept as a clearly-marked path; the eventual headline reference must be run
    on a bigger GPU / cloud. Locally this is expected to OOM at load."""
    from fp16_baseline_harness import _ATTN_IMPL
    from transformers import LlamaForCausalLM
    log.warning("fp16 baseline: ~14GB FP16 weights — expected to NOT FIT 12GB. "
                "This path is CLOUD-ONLY; running locally to record the failure.")
    model = LlamaForCausalLM.from_pretrained(
        CONFIG["model_name"], torch_dtype=CONFIG["dtype"],
        attn_implementation=_ATTN_IMPL,
    ).to(CONFIG["device"])
    model.eval()
    return model, _common_tokenizer(), {"weights": "fp16", "kv": "fp16"}


def load_nf4():
    """NF4 weights + FP16 KV. Local. Isolates weight-quant quality from the KV
    contribution. NOTE: bitsandbytes requires device_map and forbids .to()."""
    from fp16_baseline_harness import _ATTN_IMPL
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=_nf4_bnb_config(),
        torch_dtype=CONFIG["dtype"],
        device_map={"": 0},                 # bnb: do NOT call .to() afterwards
        attn_implementation=_ATTN_IMPL,
    )
    model.eval()
    return model, _common_tokenizer(), {"weights": "nf4", "kv": "fp16"}


def load_kivi():
    """NF4 weights + KIVI stock 4-bit KV. Local. The method (pre-ternary)."""
    import importlib
    from transformers import LlamaConfig
    try:
        kivi_mod = importlib.import_module(KIVI_CONFIG["kivi_module"])
        KiviClass = getattr(kivi_mod, KIVI_CONFIG["kivi_class"])
    except Exception as e:
        log.error(f"Could not import KIVI ({e}). See PHASE2_README.md.")
        raise

    config = LlamaConfig.from_pretrained(CONFIG["model_name"])
    config.k_bits = KIVI_CONFIG["k_bits"]
    config.v_bits = KIVI_CONFIG["v_bits"]
    config.group_size = KIVI_CONFIG["group_size"]
    config.residual_length = KIVI_CONFIG["residual_length"]
    config.use_flash = KIVI_CONFIG["use_flash"]

    # NF4 weights + KIVI KV. Stacking bnb NF4 onto KIVI's patched class is the
    # one combination most likely to need a tweak — if it errors, see README.
    model = KiviClass.from_pretrained(
        CONFIG["model_name"], config=config,
        quantization_config=_nf4_bnb_config(),
        torch_dtype=CONFIG["dtype"],
        device_map={"": 0},
    )
    model.eval()
    return model, _common_tokenizer(), {
        "weights": "nf4", "kv": f"kivi-int{KIVI_CONFIG['k_bits']}",
        "kivi_config": KIVI_CONFIG,
    }


def load_ternary():
    """NF4 weights + KIVI INT4 *key* cache + TERNARY *value* cache (the
    contribution). Loads exactly like `kivi`, then monkeypatches every
    attention module's forward to the ternary-V path (see ternary_v.py).
    B1 = QUALITY experiment: the simulated ternary V is stored fp16, so the KV
    *memory* number here will NOT show the win yet (that needs the B2 packed
    kernel). What matters here is PPL vs the `kivi` baseline."""
    from ternary_v import patch_ternary_v
    model, tok, meta = load_kivi()
    n = patch_ternary_v(model)
    log.info(f"patched {n} attention modules to ternary-V")
    meta = {"weights": "nf4", "kv": "kivi-int4-K + ternary-V (B1 sim, fp16 storage)"}
    return model, tok, meta


LOADERS = {"fp16": load_fp16, "nf4": load_nf4, "kivi": load_kivi,
           "ternary": load_ternary}


# ----------------------------------------------------------------------
# RUN ONE BASELINE — structurally identical to Phase 1 run_all().
# ----------------------------------------------------------------------
def run_baseline(mode):
    assert mode in LOADERS, f"unknown baseline {mode}"
    results = {
        "model_name": CONFIG["model_name"],
        "baseline": mode,
        "timestamp": datetime.datetime.now().isoformat(),
        "environment": {"gpu": "RTX 3060 12GB", "torch": torch.__version__,
                        "phase": "2-pre-ternary (three-baseline 7B)"},
        "config": {k: (str(v) if isinstance(v, torch.dtype) else v)
                   for k, v in CONFIG.items()},
        "per_context_length": {}, "gsm8k": None, "decode": None,
    }

    log.info(f"=== BASELINE '{mode}' on {CONFIG['model_name']} ===")
    try:
        model, tokenizer, meta = LOADERS[mode]()
        results["scheme"] = meta
    except Exception as e:
        if is_oom_error(e):
            log.warning(f"baseline '{mode}': OOM at load (expected for fp16 on 12GB)")
            results["load"] = "OOM"
        else:
            results["load"] = {"error": str(e)}
            log.exception(f"baseline '{mode}' failed to load")
        _save(results, mode)
        return results

    log.info(f"loaded: {results['scheme']}")

    log.info("[*] GSM8K ...")
    try:
        results["gsm8k"] = measure_gsm8k(model, tokenizer)
        log.info(f"    acc={results['gsm8k']['accuracy']:.3f}")
    except Exception as e:
        results["gsm8k"] = {"error": str(e)}; log.exception("gsm8k failed")
    free_cuda()

    log.info("[*] Decode tok/s ...")
    try:
        results["decode"] = measure_decode_tps(model, tokenizer)
        log.info(f"    {results['decode']['tokens_per_sec']:.1f} tok/s")
    except Exception as e:
        results["decode"] = {"error": str(e)}; log.exception("decode failed")
    free_cuda()

    for ctx in CONFIG["context_lengths"]:
        log.info(f"[*] ctx {ctx} ...")
        entry = {"perplexity": None, "kv_memory": None}
        try:
            entry["perplexity"] = measure_perplexity(model, tokenizer, ctx)
            log.info(f"    ppl={entry['perplexity']['perplexity']:.3f}")
        except Exception as e:
            entry["perplexity"] = "OOM" if is_oom_error(e) else {"error": str(e)}
            (log.warning if is_oom_error(e) else log.exception)(f"ppl @ {ctx}")
        free_cuda()
        try:
            entry["kv_memory"] = measure_kv_memory(model, tokenizer, ctx)
            log.info(f"    kv empirical={entry['kv_memory']['empirical_mb']:.1f}MB")
        except Exception as e:
            entry["kv_memory"] = "OOM" if is_oom_error(e) else {"error": str(e)}
            (log.warning if is_oom_error(e) else log.exception)(f"kv @ {ctx}")
        free_cuda()
        results["per_context_length"][str(ctx)] = entry

    del model
    free_cuda()
    _save(results, mode)
    return results


def _save(results, mode):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    path = os.path.join(CONFIG["output_dir"], f"phase2_{mode}_{_timestamp()}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    results["_results_path"] = path
    log.info(f"saved -> {path}")


# ----------------------------------------------------------------------
# COMPARE the three baselines side by side.
# ----------------------------------------------------------------------
def _latest(mode):
    c = sorted(glob.glob(os.path.join(CONFIG["output_dir"], f"phase2_{mode}_*.json")))
    return c[-1] if c else None


def _ppl(e):
    p = e.get("perplexity") if isinstance(e, dict) else None
    return p["perplexity"] if isinstance(p, dict) and "perplexity" in p else p


def _kv(e):
    k = e.get("kv_memory") if isinstance(e, dict) else None
    return k["empirical_mb"] if isinstance(k, dict) and "empirical_mb" in k else k


def compare_all():
    loaded = {}
    for m in ["fp16", "nf4", "kivi", "ternary"]:
        p = _latest(m)
        if p:
            with open(p) as f:
                loaded[m] = json.load(f)
    if not loaded:
        log.error("no phase2_*.json found yet. Run --baseline nf4/kivi first.")
        return

    print("\n" + "=" * 80)
    print(f"PHASE 2 (pre-ternary) — three baselines on {CONFIG['model_name']}")
    for m, r in loaded.items():
        print(f"  {m:5s}: {os.path.basename(r.get('_results_path',''))}  "
              f"scheme={r.get('scheme', r.get('load','?'))}")
    print("=" * 80)

    def g(r):
        x = r.get("gsm8k") or {}
        return f"{x['accuracy']:.3f}" if "accuracy" in x else "-"

    def d(r):
        x = r.get("decode") or {}
        return f"{x['tokens_per_sec']:.1f}" if "tokens_per_sec" in x else "-"

    print(f"{'metric':>14} | " + " ".join(f"{m:>10}" for m in loaded))
    print("-" * 80)
    print(f"{'GSM8K acc':>14} | " + " ".join(f"{g(loaded[m]):>10}" for m in loaded))
    print(f"{'decode tok/s':>14} | " + " ".join(f"{d(loaded[m]):>10}" for m in loaded))
    print("-" * 80)
    for ctx in CONFIG["context_lengths"]:
        c = str(ctx)
        def cell(m, fn):
            e = loaded[m].get("per_context_length", {}).get(c, {})
            v = fn(e)
            return f"{v:.3f}" if isinstance(v, float) and fn is _ppl else \
                   (f"{v:.0f}" if isinstance(v, float) else str(v))
        print(f"{'ppl @'+c:>14} | " + " ".join(f"{cell(m,_ppl):>10}" for m in loaded))
        print(f"{'KV MB @'+c:>14} | " + " ".join(f"{cell(m,_kv):>10}" for m in loaded))
    print("=" * 80)
    print("Read: nf4->kivi ppl change = quality cost of KV quant; "
          "nf4->kivi KV MB drop = the memory win; fp16 = cloud reference.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["fp16", "nf4", "kivi", "ternary", "all"],
                    help="which baseline to run")
    ap.add_argument("--compare", action="store_true",
                    help="diff the latest fp16/nf4/kivi JSONs")
    args = ap.parse_args()

    if args.baseline == "all":
        for m in ["nf4", "kivi", "ternary", "fp16"]:  # locals first, cloud-only fp16 last
            run_baseline(m)
    elif args.baseline:
        run_baseline(args.baseline)

    if args.compare or args.baseline == "all":
        compare_all()
