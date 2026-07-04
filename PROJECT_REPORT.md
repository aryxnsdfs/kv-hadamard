# KV-Cache Quantization Research — Progress Report

**Date:** 2026-06-28
**Hardware:** RTX 3060 12GB · WSL Ubuntu · CUDA 12.1
**Status:** Phase 1 complete · Phase 2 Stage A complete (pipeline proven) · Phase 2 Stage B (ternary-V, the contribution) not started

---

## 1. Objective

Investigate quantizing the **KV cache** of a transformer LLM to reduce GPU memory
during long-context inference, without degrading output quality or speed beyond
acceptable limits. The KV cache grows linearly with context length and becomes the
dominant VRAM cost at long contexts on large models — the bottleneck this work targets.

The project is staged: build a trustworthy measurement harness and reference baselines
first, then introduce the novel method (ternary value-cache quantization) and measure it
against those baselines. The discipline throughout: **never compare against a number you
cannot trust.**

---

## 2. What was built

### 2.1 Phase 1 — FP16 baseline harness (`fp16_baseline_harness.py`)
A single runnable harness measuring four metrics per context length, each an independent
function, with a JSON output schema and a summary table:

1. **Perplexity** — sliding-window over WikiText-103 (stride 512), memory-lean scoring.
2. **GSM8K accuracy** — greedy, exact-match on extracted final number, transcripts saved.
3. **Decode throughput** — steady-state tokens/sec, CUDA-event timed, prefill excluded.
4. **KV-cache memory** — computed two ways (analytic + empirical allocator delta),
   isolated from model weights.

Plus: OOM handling (a failed length is recorded, not a crash), CUDA cleanup between
lengths, timestamped JSON, and a Phase 2 stub.

### 2.2 Phase 2 — three-baseline harness (`phase2_kivi_harness.py`)
Reuses the Phase 1 metric functions **unchanged** (imported, not copied), overriding only
config and adding three model loaders so all baselines diff directly:

- `fp16` — FP16 weights + FP16 KV (cloud reference; won't fit 12GB locally).
- `nf4` — NF4 weights + FP16 KV (local; isolates weight-quant from KV quant).
- `kivi` — NF4 weights + KIVI 4-bit KV (local; the stock-KIVI method).

Plus `setup_phase2.sh` (one-venv environment build) and `PHASE2_README.md`.

---

## 3. Results

### 3.1 Phase 1 — Llama-3.2-1B, FP16 (validated)
| ctx | Perplexity | KV cache (MB) |
|---|---|---|
| 2048 | 8.78 | 64 |
| 4096 | 8.45 | 128 |
| 8192 | 8.23 | 256 |
| 16384 | 8.43 | 512 |

GSM8K 3/100 · decode 66 tok/s.

**Cross-validated**: identical perplexity (to 3 decimals) on two independent PyTorch
builds (native Windows vs WSL/Linux), confirming the numbers are real, not artifacts.
**Key finding:** on a 1B model the KV cache is tiny (≤512 MB) — *not* the memory
bottleneck. The memory-win story belongs to a larger model, motivating the move to 7B in
Phase 2. The 1B work served its purpose: validating the instrument on a cheap model.

### 3.2 Phase 2 — Llama-2-7B, NF4 weights, FP16 vs KIVI 4-bit KV
| ctx | PPL (nf4) | PPL (kivi) | KV nf4 (MB) | KV kivi (MB) | KV saved |
|---|---|---|---|---|---|
| 2048 | 4.894 | **4.894** | 1024 | **328** | **68%** |
| 4096 | 4.722 | **4.722** | 2048 | **647** | **68%** |
| 8192 | — | — | 4096 | 1287 | 69% |
| 16384 | — | — | 8192 | 2569 | 69% |

GSM8K (N=40): nf4 0.025, kivi 0.000 (1-problem difference = noise at this N).
Decode: nf4 20.5 → kivi 11.5 tok/s.

**Headline findings:**
- **Quality preserved exactly** — perplexity identical to 3 decimals at all valid
  contexts. KIVI's 4-bit KV is effectively lossless here. (KV memory dropped, so the
  quantization is genuinely active, not a no-op.)
- **~68% KV memory reduction** — ~3.1×, not the naive 4×, because KIVI keeps the last 32
  tokens in FP16 and stores per-group scales/zeros. Consistent with the KIVI paper.
- **~1.8× decode slowdown** (20.5 → 11.5 tok/s) — the cost of quant/dequant; the
  speed-for-memory trade.

**Important caveat:** Llama-2-7B has a 4096-token context window. Perplexity at 8192/16384
is meaningless (the model is out of trained range) and is reported only for the
KV-**memory** numbers, which remain valid at all lengths.

---

## 4. The engineering journey (what made it hard, and what we learned)

Every blocker below was a real environment/compatibility issue, surfaced and fixed one at
a time. All fixes are now permanent in the code/scripts.

**Phase 1**
- Gated model (`meta-llama/Llama-3.2-1B`) → ungated mirror (`unsloth/...`).
- Stale HF dataset IDs → namespaced (`openai/gsm8k`, `Salesforce/wikitext`).
- Three separate "fake OOM" bugs that would have invalidated the baseline:
  1. fp32 cross-entropy over the full window × 128k vocab — fixed with chunked scoring.
  2. lm_head projecting all positions when only ~512 are scored — fixed by projecting
     only the needed positions.
  3. **The real one:** grouped-query attention silently falling back to SDPA's O(seq²)
     *math* kernel (no flash on Windows; efficient kernel rejects unequal Q/KV heads) —
     fixed with a custom attention that expands K/V heads so the efficient kernel engages.
  Without these, "FP16 OOMs at 8k" would have been recorded as a baseline — and later
  "our method beats it" would have been beating our own bug, not FP16.

**Phase 2**
- KIVI's `requirements.txt` over-constrained (pins unavailable torch) → install pins
  manually.
- CUDA 12.1 `nvcc` rejects gcc > 12 → install and point to gcc-12.
- Kernel + flash-attn builds need `--no-build-isolation` (they import torch at build time).
- flash-attn source build OOM-killed → used a **prebuilt wheel** (no compilation).
- **KIVI's code needs transformers 4.44.2**, not the 4.36.2 its requirements claim
  (uses the newer `LlamaRotaryEmbedding(config=)` API).
- **KIVI's CUDA kernel only accepts fp16 (Half)**, not bf16 → dtype fixed to fp16.
- KIVI mandates flash attention (no non-flash path).
- Environment hygiene: must activate the venv every shell; `PYTHONNOUSERSITE=1` to stop a
  user-site transformers-5.x from shadowing the venv; reduced GSM8K N and eval-token budget
  to keep 7B runs short and robust against transient WSL GPU faults.

**Methodological wins**
- Phase 1 metric functions run **unchanged** across: 1B and 7B; FP16 and NF4; transformers
  5.0 and 4.44; Windows and WSL. The harness is genuinely model/dtype/version agnostic.
- All baselines compared at the **same dtype, same transformers, same env** — so any
  difference is attributable to the weight/KV scheme alone.

---

## 5. Current status

- ✅ **Phase 1**: trustworthy FP16 baseline harness + validated 1B numbers.
- ✅ **Phase 2 Stage A**: full pipeline proven on 7B — NF4 load, KIVI 4-bit KV, all four
  metrics, three baselines diffing cleanly. KIVI reference established: lossless quality,
  ~68% KV memory saved, ~1.8× decode cost.
- ⬜ **Phase 2 Stage B (the contribution)**: ternary value-cache quantization — not started.

---

## 6. Next step — ternary-V

Replace KIVI's INT4 **value** cache with **ternary-V**, run the same harness, and diff
against the `kivi` baseline above. Because the baseline is a working KIVI on the identical
model/env, any change in perplexity, KV memory, or decode speed is unambiguously the effect
of ternary quantization — the cleanest possible attribution of the result.

This is a kernel/quantization change (KIVI's value-quant path), not a config tweak, and is
the actual novel contribution of the project.

---

## 7. Artifacts
- `fp16_baseline_harness.py` — Phase 1 harness (model/dtype/version agnostic).
- `phase2_kivi_harness.py` — Phase 2 three-baseline harness.
- `setup_phase2.sh` — one-venv Phase 2 environment build.
- `PHASE2_README.md` — Phase 2 run guide + decisions.
- `baseline_results/*.json` — all measured runs (timestamped) + GSM8K transcripts + logs.
