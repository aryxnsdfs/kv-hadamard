# Phase 2 (pre-ternary) — Three Baselines on Llama-2-7B

This step proves the **whole pipeline runs on 7B** — the model your real
result lives on — and produces three directly-comparable baselines. **No
ternary-V kernel yet.** We get KIVI's *stock* INT4 KV working on 7B-NF4 first,
so that when ternary lands later, any change in the numbers is unambiguously
your contribution, measured against a working KIVI baseline on the same model.

## The three baselines
| name | weights | KV cache | runs where | purpose |
|---|---|---|---|---|
| `fp16` | FP16 (~14GB) | FP16 | **cloud / big GPU only** | eventual headline reference; won't fit 12GB |
| `nf4`  | NF4 (~3.5GB) | FP16 | local | isolates weight-quant quality hit from the KV contribution |
| `kivi` | NF4 | KIVI INT4 | local | the method (stock KIVI, pre-ternary) |

Why these three: `nf4 -> kivi` PPL change = the quality cost of quantizing the
KV cache (your contribution's effect). `nf4 -> kivi` KV-MB drop = the memory
win. `fp16` is the absolute reference you can only measure on a bigger GPU.

## Why Llama-2-7B (not Llama-3.2-1B)
1. KV cache is negligible on 1B — the memory win only shows at 7B, where the
   FP16 KV cache is ~2GB @2048 ... ~17GB @16384 and actually dominates VRAM.
2. **Llama-2 is supported by KIVI's pinned transformers**, so KIVI works out of
   the box with zero version-downgrade fight (Llama-3.2 needed transformers
   ≥4.45, which KIVI doesn't support — that fight is avoided entirely).
3. The 1B work was not wasted: it validated the harness (PPL loop correct,
   memory-honest, OOM handling real). Instrument proven on a cheap model before
   pointing it at the expensive one.

7B FP16 is ~14GB and **will not fit a 12GB 3060**, so locally we run **NF4
weights**. That is why the two local baselines are NF4, and `fp16` is recorded
as cloud-only.

## One venv, perfect parity
All three baselines run in **one venv** (`~/kvenv_phase2`) built by
`setup_phase2.sh`: KIVI's pinned transformers supports Llama-2 *and*
bitsandbytes NF4 works on it. fp16/nf4/kivi therefore share identical
library/kernels — the diff is attributable to the weight/KV scheme alone.

## Run
```bash
# in WSL
bash setup_phase2.sh
source ~/kvenv_phase2/bin/activate
cd /mnt/c/Users/aryan/Downloads/cuda

python3 phase2_kivi_harness.py --baseline nf4     # local
python3 phase2_kivi_harness.py --baseline kivi    # local
python3 phase2_kivi_harness.py --baseline fp16    # expected: OOM at load locally
python3 phase2_kivi_harness.py --compare          # side-by-side table
# or all at once:
python3 phase2_kivi_harness.py --baseline all
```
Outputs: `baseline_results/phase2_<mode>_<ts>.json` + a comparison table.

## Harness changes
**Zero changes to `fp16_baseline_harness.py`.** `phase2_kivi_harness.py`
imports the Phase 1 metric functions unchanged and only:
- overrides `CONFIG["model_name"]` / `context_lengths` in memory (file untouched),
- adds the three loaders (`load_fp16` / `load_nf4` / `load_kivi`).
The PPL loop, GSM8K, decode, KV-isolation, OOM handling and JSON schema are
identical across all three — that is what makes them diffable.

## Known risk points (can't be verified without the GPU)
1. **`kivi` = NF4 weights + KIVI patched class.** Stacking bitsandbytes NF4 onto
   KIVI's `from_pretrained` is the combination most likely to need a tweak
   (KIVI may not thread `quantization_config` through). If it errors, options:
   run `kivi` with FP16 weights on cloud, or patch KIVI's `from_pretrained` to
   accept the bnb config. Tell me the traceback and I'll wire it.
2. **flash-attn build** on WSL can fail. If so, set
   `KIVI_CONFIG["use_flash"] = False` in `phase2_kivi_harness.py` (KIVI has a
   non-flash path) to unblock, slower but correct.
3. **bitsandbytes + `measure_kv_memory`.** NF4 weights sit resident before the
   prefill, so the alloc-delta still isolates KV correctly (the floor just
   moves down). No change needed.
4. **Long contexts OOM on `nf4`** (FP16 KV) at 8192/16384 — that's expected and
   honest; KIVI is what should survive them. The harness logs `OOM` and continues.

## What success looks like
Comparison table where, per context length:
- `nf4 -> kivi` PPL change is small (quality preserved),
- `nf4 -> kivi` KV MB drops ~4x (the memory win),
- `kivi` completes 8192/16384 where `nf4` OOMs.

Once all three diff cleanly, **then** we swap KIVI's INT4-V for ternary-V —
that's Phase 2 proper, and the cleanest possible attribution of your result.

## GSM8K note
Llama-2-7B base GSM8K accuracy is modest (older non-instruct model). You are
measuring *preservation* (does NF4/KIVI hold the FP16 number), not absolute
capability. For a stronger reasoning signal later: Llama-2-7B-chat or
Mistral-7B — but get the base pipeline working first.
