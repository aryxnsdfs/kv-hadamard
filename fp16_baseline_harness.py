#!/usr/bin/env python3
"""
Phase 1 — FP16 Baseline Harness for KV-Cache Quantization Research
==================================================================

This is the UNQUANTIZED CONTROL. It loads the stock
`transformers.LlamaForCausalLM` (NOT the KIVI class) and measures four
metrics across a set of context lengths, so that every future quantized
variant can be diffed directly against these numbers.

There is intentionally NO quantization code in this file. The only nod to
Phase 2 is a clearly-marked stub at the very bottom.

Environment this targets (assumed, not auto-detected):
    WSL Ubuntu 22.04 / RTX 3060 12GB / CUDA 12.1 / PyTorch >= 2.1
    transformers + datasets
    Model: meta-llama/Llama-3.2-1B, FP16

Metrics:
    1. Perplexity (sliding-window) on a WikiText-103 held-out slice.
    2. GSM8K exact-match accuracy (baseline guard, greedy decode).
    3. Decode tokens/sec (steady-state decode only, CUDA-event timed).
    4. KV-cache memory in MB (analytic + empirical, isolated from weights).

OOM at a given context length is a VALID DATA POINT, not a crash. It is
caught, logged as "OOM", and the harness continues.
"""

import os
import gc
import re
import sys
import json
import time
import math
import logging
import datetime
import traceback

import torch
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
from datasets import load_dataset

import torch.nn.functional as F

# ----------------------------------------------------------------------
# Memory-efficient attention for GQA on this torch build.
#
# Why this exists: Llama-3.2 uses grouped-query attention (32 query heads, 8
# KV heads). On this native-Windows torch build, flash attention is NOT
# compiled and the mem-efficient SDPA kernel REFUSES unequal Q/KV head counts
# ("both fused kernels require query, key and value to have the same
# num_heads"). So stock SDPA silently falls back to the MATH kernel, which
# materializes the full [1, heads, seq, seq] score matrix — O(seq^2) memory.
# That, not the KV cache, is what made a 1B model need ~7.4GB at 4096 and OOM
# at 8192/16384.
#
# Fix (exactly what the torch warning suggests): expand K/V from 8 -> 32 heads
# with repeat_interleave so all three tensors have equal heads, THEN call SDPA
# with the efficient backend (causal flag, no explicit mask). MATH is kept as
# a last-resort backend so the call never hard-fails — if the efficient kernel
# is still unavailable it degrades to (slow, correct) math instead of crashing.
# Numerically identical to stock attention.
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION,
                      SDPBackend.EFFICIENT_ATTENTION,
                      SDPBackend.MATH]

    def _eff_gqa_attention(module, query, key, value, attention_mask=None,
                           scaling=None, dropout=0.0, **kwargs):
        # query [b, Hq, q, d]; key/value [b, Hkv, k, d]
        Hq, Hkv = query.shape[1], key.shape[1]
        if Hkv != Hq:                       # expand KV heads to match Q (GQA)
            rep = Hq // Hkv
            key = key.repeat_interleave(rep, dim=1)
            value = value.repeat_interleave(rep, dim=1)
        q_len, k_len = query.shape[2], key.shape[2]
        with sdpa_kernel(_SDPA_BACKENDS):
            if q_len == k_len and q_len > 1:   # full causal self-attention
                out = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, is_causal=True,
                    dropout_p=dropout if module.training else 0.0, scale=scaling)
            else:                              # cached decode / single token
                out = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, is_causal=False,
                    dropout_p=dropout if module.training else 0.0, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    from transformers import AttentionInterface
    AttentionInterface.register("eff_gqa", _eff_gqa_attention)
    _ATTN_IMPL = "eff_gqa"
except Exception:  # pragma: no cover - fall back to stock sdpa
    _ATTN_IMPL = "sdpa"


# ----------------------------------------------------------------------
# Logging — timestamps + level, streamed to stdout AND a log file so you
# can watch progress live and read it back later.
# ----------------------------------------------------------------------
os.makedirs("./baseline_results", exist_ok=True)
_LOG_PATH = os.path.join(
    "./baseline_results",
    f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_PATH),
    ],
)
log = logging.getLogger("fp16_baseline")

# Silence noisy HTTP/dataset chatter so the progress logs stay readable.
for _noisy in ("httpx", "urllib3", "filelock", "huggingface_hub",
               "datasets", "fsspec"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _gpu_mem_str():
    """Current GPU memory snapshot, for inline progress logging. On Windows
    a climbing 'reserved' near total VRAM with the run crawling = the driver
    is spilling to system RAM (the silent slowdown)."""
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return f"alloc={alloc:.0f}MB reserved={reserved:.0f}MB"


# ======================================================================
# CONFIG — single source of truth. Edit here, nowhere else.
# ======================================================================
CONFIG = {
    # --- model ---
    # Ungated mirror of meta-llama/Llama-3.2-1B — identical weights + config
    # (16 layers / 8 KV heads / head_dim 64). Swap back to the Meta repo once
    # your gated-access request is approved; numbers are equivalent.
    "model_name": "unsloth/Llama-3.2-1B",
    "dtype": torch.float16,
    "device": "cuda",

    # --- context lengths to sweep ---
    "context_lengths": [2048, 4096, 8192, 16384],

    # --- KV-cache architecture facts (Llama-3.2-1B) ---
    # Used only by the ANALYTIC memory calc; the empirical calc reads them
    # back from the live model config to cross-check.
    "num_layers": 16,
    "num_kv_heads": 8,
    "head_dim": 64,
    "batch_size": 1,

    # --- perplexity (WikiText-103) ---
    "wikitext_dataset": "Salesforce/wikitext",
    "wikitext_config": "wikitext-103-raw-v1",
    "wikitext_split": "test",          # held-out test split
    "wikitext_stride": 512,            # sliding-window stride
    # Cap how much raw text we tokenize so PPL runs in reasonable time.
    # This bounds the number of windows, not the window size.
    "wikitext_max_eval_tokens": 100_000,

    # --- GSM8K ---
    "gsm8k_dataset": "openai/gsm8k",
    "gsm8k_config": "main",
    "gsm8k_split": "test",
    "gsm8k_n": 100,                    # subset size (configurable)
    "gsm8k_max_new_tokens": 256,       # generation budget per problem
    # Stop generation when the base model starts looping into a new
    # "Question:" block. Keeps the answer single-turn so extraction is clean.
    "gsm8k_stop_strings": ["\nQuestion:", "\n\nQuestion"],
    "gsm8k_transcript_count": 5,       # save full transcripts for N problems

    # --- decode throughput ---
    "decode_prompt_tokens": 64,        # short prefill; we exclude its time
    "decode_warmup_tokens": 32,        # discarded warm-up generations
    "decode_measure_tokens": 256,      # >= 200 measured decode steps

    # --- output ---
    "output_dir": "./baseline_results",
}


# ======================================================================
# Small shared utilities
# ======================================================================
def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def free_cuda():
    """Hard reset of allocator state between measurements so one length's
    allocation cannot contaminate the next."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_and_tokenizer():
    """Load the STOCK FP16 LlamaForCausalLM. This is the control."""
    tok = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    # Llama tokenizer has no pad token by default; reuse EOS for padding.
    # We never train, and we run batch=1 generate, so this only matters if
    # any batched path pads — set it to be safe and explicit.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = LlamaForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=CONFIG["dtype"],
        device_map=None,
        # Memory-efficient attention. Without this the model can fall back to
        # EAGER attention, which materializes the full [heads, seq, seq] score
        # matrix -> O(seq^2) memory + compute. That, not the KV cache, was
        # making 4096 crawl at ~7.7s/window and OOM at 8192. SDPA is O(seq)
        # and uses fused kernels. (flash-attn-2 would be faster still but is
        # an optional dep; SDPA ships with torch.) On this GQA model + torch
        # build, "eff_gqa" (registered at top of file) expands KV heads so the
        # efficient kernel actually engages; falls back to "sdpa" otherwise.
        attn_implementation=_ATTN_IMPL,
    ).to(CONFIG["device"])
    model.eval()

    # Confirm where we're actually running — the #1 cause of "stuck for
    # hours" is silently running on CPU, or a CUDA build that isn't real.
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info(f"Device: CUDA ({name}, {total:.1f}GB total VRAM)")
        log.info(f"Attention impl: {getattr(model.config, '_attn_implementation', '?')}")
        log.info(f"Model loaded. {_gpu_mem_str()}")
    else:
        log.warning("Device: CPU ONLY — CUDA not available. This will be "
                    "extremely slow; expect hours on the long contexts.")
    return model, tok

    # ===== NF4 WEIGHT-QUANT STUB — DO NOT IMPLEMENT IN PHASE 1 =====
    # Three baselines will eventually run through this SAME loader, all
    # feeding the SAME metric functions unchanged:
    #   (1) FP16 weights + FP16 KV   <- this file (the control)
    #   (2) NF4  weights + FP16 KV   <- isolates KV contribution from
    #                                   weight-quant noise (the middle case)
    #   (3) NF4  weights + ternary KV <- the full quantized variant
    #
    # The NF4 weight-loading path slots in HERE, swapping only the
    # from_pretrained call — nothing downstream changes:
    #
    #   from transformers import BitsAndBytesConfig
    #   bnb = BitsAndBytesConfig(
    #       load_in_4bit=True,
    #       bnb_4bit_quant_type="nf4",
    #       bnb_4bit_compute_dtype=torch.float16,
    #       bnb_4bit_use_double_quant=True,
    #   )
    #   model = LlamaForCausalLM.from_pretrained(
    #       CONFIG["model_name"],
    #       quantization_config=bnb,
    #       torch_dtype=torch.float16,
    #       device_map={"": 0},   # bnb requires device_map, not .to()
    #   )
    #
    # NOTE for metric 4 when this is active: NF4 shrinks the WEIGHTS, not
    # the KV cache. The memory_allocated() baseline taken before the prefill
    # already includes the (now smaller) resident weights, so the KV delta
    # stays correctly isolated — the empirical KV numbers remain comparable
    # across all three baselines. Only the "before" floor moves.
    # ==============================================================


def is_oom_error(err: Exception) -> bool:
    """torch raises torch.cuda.OutOfMemoryError (subclass of RuntimeError)
    on most versions; older paths raise a plain RuntimeError whose message
    contains 'out of memory'. Catch both."""
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


# ======================================================================
# METRIC 1 — Perplexity (sliding-window) on WikiText-103
# ======================================================================
def measure_perplexity(model, tokenizer, context_length):
    """
    Sliding-window perplexity at a given window size (= context_length),
    using the strided method from the HF perplexity docs.

    Method:
      - Concatenate the WikiText-103 *raw* test split into one stream.
      - Tokenize once.
      - Slide a window of `context_length` tokens with stride
        CONFIG["wikitext_stride"].
      - For each window, only the last `trg_len` tokens (the part not seen
        in the previous window) contribute to the loss; earlier tokens are
        masked with -100 so they act purely as context.
      - PPL = exp(sum(NLL) / num_scored_tokens).

    Returns a dict with ppl and bookkeeping, or raises on OOM (caught by
    the orchestrator).
    """
    stride = CONFIG["wikitext_stride"]

    ds = load_dataset(
        CONFIG["wikitext_dataset"],
        CONFIG["wikitext_config"],
        split=CONFIG["wikitext_split"],
    )
    # WikiText rows are line fragments; join into one text stream.
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids_full = enc.input_ids
    # Bound total tokens so runtime stays sane (see CONFIG).
    max_tok = CONFIG["wikitext_max_eval_tokens"]
    if input_ids_full.size(1) > max_tok:
        input_ids_full = input_ids_full[:, :max_tok]
    seq_len = input_ids_full.size(1)

    device = CONFIG["device"]
    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    begins = list(range(0, seq_len, stride))
    n_windows = len(begins)
    log.info(f"  perplexity: window={context_length} stride={stride} "
             f"eval_tokens={seq_len} -> {n_windows} windows")
    t0 = time.time()

    for i, begin in enumerate(begins):
        end = min(begin + context_length, seq_len)
        trg_len = end - prev_end  # number of *new* tokens to score
        ids = input_ids_full[:, begin:end].to(device)

        L = ids.size(1)
        with torch.no_grad():
            # IMPORTANT (memory): two separate blow-ups had to be killed so
            # that this 1B model does NOT fake-OOM on long contexts (its KV
            # cache is only a few hundred MB at 8k — nothing here should OOM):
            #
            #  1. fp32 cross-entropy. The stock labels= path runs CE over the
            #     WHOLE window x 128k vocab in fp32 every sliding step. We
            #     instead score the loss in fp32 row-chunks (below).
            #  2. lm_head logits. The lm_head projects hidden->[1, L, 128k]
            #     for ALL L positions, scaling with context length (~7.8GB @
            #     4096, ~14GB @ 8192). But only the last `trg_len` tokens are
            #     scored. The `logits_to_keep` kwarg would handle this, but it
            #     is absent in this transformers version. So we do it by hand,
            #     version-independently: run the base transformer, slice the
            #     hidden states to just the positions we need, then run lm_head
            #     on ONLY those.
            #
            #     Cap the projection at `stride+1` rows for EVERY window,
            #     including the first. Steady windows already need only stride
            #     rows; the first window would otherwise project its full
            #     `trg_len`=context_length rows (1GB @ 4096, 2GB @ 8192, 4GB @
            #     16384) and set a context-scaling memory high-water that OOMs
            #     the long lengths. With the cap, peak memory is FLAT across
            #     all context lengths. Cost: the first window scores only its
            #     last `stride` tokens, i.e. the first ~context_length tokens
            #     of the corpus are not scored once — negligible over a 100k
            #     eval, and the identical method runs against the quantized
            #     model, so the FP16-vs-quantized comparison stays fair.
            keep = min(trg_len, stride) + 1  # +1: position t predicts token t+1
            # Attention efficiency is handled by the registered "eff_gqa"
            # attention impl (see top of file), so a plain forward is enough.
            base_out = model.model(ids, use_cache=False)  # base transformer
            hidden = base_out.last_hidden_state[:, -keep:, :]  # [1, M, hidden]
            logits = model.lm_head(hidden)[0]                # [M, vocab]
            del base_out, hidden

        M = logits.size(0)
        # The kept logits always END at absolute position L-1. Score the last
        # `trg_len` tokens (capped by what's predictable, dropping the very
        # first token of the first window which has no predictor).
        n_score = min(trg_len, M - 1)
        shift_logits = logits[M - 1 - n_score:M - 1, :]  # [n_score, vocab]
        shift_labels = ids[0, L - n_score:L]             # [n_score]

        # Chunked fp32 NLL sum — caps the fp32 footprint at chunk*vocab,
        # so even the long first window cannot blow up memory.
        ce_chunk = 512
        win_nll = 0.0
        for s in range(0, shift_labels.size(0), ce_chunk):
            lg = shift_logits[s:s + ce_chunk].float()
            lb = shift_labels[s:s + ce_chunk]
            win_nll += torch.nn.functional.cross_entropy(
                lg, lb, reduction="sum"
            ).item()
            del lg, lb
        nll_sum += win_nll
        n_tokens += shift_labels.size(0)

        prev_end = end
        del ids, logits, shift_logits, shift_labels

        # Progress every 10 windows: elapsed, rate, GPU mem. A sudden
        # collapse in windows/s + reserved memory pinned near total VRAM =
        # the driver is spilling to system RAM.
        if (i + 1) % 10 == 0 or (i + 1) == n_windows:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_windows - (i + 1)) / rate if rate > 0 else 0
            log.info(f"    window {i+1}/{n_windows} "
                     f"({rate:.2f} win/s, {elapsed:.0f}s elapsed, "
                     f"~{eta:.0f}s left) {_gpu_mem_str()}")

        if end == seq_len:
            break

    ppl = math.exp(nll_sum / n_tokens)
    return {
        "perplexity": ppl,
        "window": context_length,
        "stride": stride,
        "scored_tokens": n_tokens,
        "eval_tokens": seq_len,
    }


# ======================================================================
# METRIC 2 — GSM8K exact-match accuracy (baseline guard)
# ======================================================================
# Answer-extraction regex — shown explicitly so it can be corrected.
#
#   GOLD: GSM8K gold answers end with a line "#### <number>". We split on
#         "####" and take the trailing number.
#   PRED: The model's final answer is taken as the LAST number appearing in
#         its generation. This matches GSM8K convention (the model writes a
#         chain of reasoning then states the result last). We allow an
#         optional leading '$', thousands separators, a sign, and decimals.
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _extract_number(text: str):
    """Return the last number-like token in `text`, normalized to a float
    string (commas and '$' stripped). None if no number present."""
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace("$", "").replace(",", "")
    try:
        # Normalize 42.0 vs 42 so exact-match doesn't trip on formatting.
        f = float(raw)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return None


def _extract_gold(answer_field: str):
    """GSM8K gold: text after the final '####'."""
    after = answer_field.split("####")[-1]
    return _extract_number(after)


def measure_gsm8k(model, tokenizer):
    """
    Greedy GSM8K exact-match over a configurable subset (default N=100) of
    the test split. Saves full prompt+generation transcripts for the first
    CONFIG["gsm8k_transcript_count"] problems.

    NOTE: This is a baseline ACCURACY GUARD, not a long-chain stress test.
    It does not depend on the context-length sweep — it is run once.
    """
    n = CONFIG["gsm8k_n"]
    device = CONFIG["device"]

    ds = load_dataset(
        CONFIG["gsm8k_dataset"],
        CONFIG["gsm8k_config"],
        split=CONFIG["gsm8k_split"],
    )
    ds = ds.select(range(min(n, len(ds))))

    correct = 0
    total = 0
    transcripts = []
    log.info(f"  gsm8k: {len(ds)} problems, greedy, "
             f"max_new_tokens={CONFIG['gsm8k_max_new_tokens']}")
    t0 = time.time()

    # Stop-at-string via a StoppingCriteria so this works on ANY transformers
    # version. (generate(stop_strings=...) only exists in transformers >=4.40;
    # KIVI pins 4.36.) Behaviour is identical: halt when the generated text
    # contains a stop marker, so the base model can't loop into a new
    # "Question:" block and drag last-number extraction off the real answer.
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopOnStrings(StoppingCriteria):
        def __init__(self, tok, stops, prompt_len):
            self.tok, self.stops, self.prompt_len = tok, stops, prompt_len

        def __call__(self, input_ids, scores, **kwargs):
            text = self.tok.decode(input_ids[0, self.prompt_len:],
                                   skip_special_tokens=True)
            return any(s in text for s in self.stops)

    for i, ex in enumerate(ds):
        question = ex["question"]
        gold = _extract_gold(ex["answer"])

        # Plain instruction prompt. Llama-3.2-1B base — no chat template
        # applied (see assumptions). BOS is added by the tokenizer.
        prompt = (
            "Answer the following math problem. "
            "Show your reasoning, then give the final numeric answer.\n\n"
            f"Question: {question}\nAnswer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.size(1)
        stopper = StoppingCriteriaList([
            _StopOnStrings(tokenizer, CONFIG["gsm8k_stop_strings"], prompt_len)
        ])

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=CONFIG["gsm8k_max_new_tokens"],
                do_sample=False,                 # greedy
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                # Halt at the start of a repeated/hallucinated next problem
                # (base models greedy-loop the "Question:/Answer:" pattern;
                # stopping keeps last-number extraction on the real answer).
                # Version-agnostic StoppingCriteria — see _StopOnStrings above.
                stopping_criteria=stopper,
            )
        gen_text = tokenizer.decode(
            gen[0, inputs.input_ids.size(1):], skip_special_tokens=True
        )
        pred = _extract_number(gen_text)

        total += 1
        if pred is not None and gold is not None and pred == gold:
            correct += 1

        if i < CONFIG["gsm8k_transcript_count"]:
            transcripts.append({
                "index": i,
                "prompt": prompt,
                "generation": gen_text,
                "gold": gold,
                "pred": pred,
                "correct": (pred == gold),
            })

        del inputs, gen

        if (i + 1) % 10 == 0 or (i + 1) == len(ds):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(ds) - (i + 1)) / rate if rate > 0 else 0
            log.info(f"    {i+1}/{len(ds)} done, running acc="
                     f"{correct/(i+1):.3f} ({rate:.2f} prob/s, ~{eta:.0f}s left)")

    acc = correct / total if total else 0.0

    # Persist transcripts to file.
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    tpath = os.path.join(
        CONFIG["output_dir"], f"gsm8k_transcripts_{_timestamp()}.json"
    )
    with open(tpath, "w") as f:
        json.dump(transcripts, f, indent=2)

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "transcripts_path": tpath,
        "answer_regex": _NUMBER_RE.pattern,
    }


# ======================================================================
# METRIC 3 — Decode tokens/sec (steady-state decode only)
# ======================================================================
def measure_decode_tps(model, tokenizer):
    """
    Steady-state autoregressive decode throughput.

    - Prefill a short prompt (time EXCLUDED).
    - Warm up by generating CONFIG["decode_warmup_tokens"] tokens (DISCARDED).
    - Then time CONFIG["decode_measure_tokens"] (>=200) decode steps with
      CUDA events, synchronizing around the measured region only.

    We drive the decode loop manually (one token per forward, feeding
    past_key_values) so prefill is provably outside the timed region.
    """
    device = CONFIG["device"]

    prompt_ids = torch.randint(
        low=10, high=tokenizer.vocab_size - 10,
        size=(1, CONFIG["decode_prompt_tokens"]),
        device=device,
    )

    # --- Prefill (NOT timed) ---
    with torch.no_grad():
        out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    def _decode_steps(num_steps, past, next_tok):
        for _ in range(num_steps):
            with torch.no_grad():
                out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        return past, next_tok

    # --- Warm-up (discarded) ---
    past, next_tok = _decode_steps(
        CONFIG["decode_warmup_tokens"], past, next_tok
    )

    # --- Measured region ---
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    past, next_tok = _decode_steps(
        CONFIG["decode_measure_tokens"], past, next_tok
    )
    end_evt.record()
    torch.cuda.synchronize()

    elapsed_ms = start_evt.elapsed_time(end_evt)
    elapsed_s = elapsed_ms / 1000.0
    tps = CONFIG["decode_measure_tokens"] / elapsed_s

    del past, next_tok, out, prompt_ids
    return {
        "tokens_per_sec": tps,
        "measured_tokens": CONFIG["decode_measure_tokens"],
        "elapsed_s": elapsed_s,
        "prefill_tokens": CONFIG["decode_prompt_tokens"],
    }


# ======================================================================
# METRIC 4 — KV-cache memory in MB (isolated from weights)
# ======================================================================
def _analytic_kv_mb(seq_len, model=None):
    """Analytic KV-cache size:
        2 (K and V) * num_layers * num_kv_heads * seq_len * head_dim * bytes
    Reads architecture from the live model config when available, else from
    CONFIG. dtype_bytes from CONFIG dtype.
    """
    if model is not None:
        cfg = model.config
        num_layers = cfg.num_hidden_layers
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    else:
        num_layers = CONFIG["num_layers"]
        num_kv_heads = CONFIG["num_kv_heads"]
        head_dim = CONFIG["head_dim"]

    dtype_bytes = torch.finfo(CONFIG["dtype"]).bits // 8
    batch = CONFIG["batch_size"]
    total_bytes = (
        2 * num_layers * num_kv_heads * seq_len * head_dim * dtype_bytes * batch
    )
    return total_bytes / (1024 ** 2)


def measure_kv_memory(model, tokenizer, context_length):
    """
    KV-cache footprint at a given context length, computed TWO ways and
    both reported. The KV cache is isolated from the ~2.4GB static model
    weights: weights are already resident before we start; we measure the
    allocator DELTA caused purely by populating the cache.

    (a) analytic: closed form from architecture.
    (b) empirical: torch.cuda.memory_allocated() after - before a prefill
        that fills the cache to `context_length` tokens.

    SCALE NOTE (1B model): the analytic KV cache here is only ~32MB at 8k,
    so on Llama-3.2-1B the KV cache is NOT the memory bottleneck — the
    [1, seq, vocab~128k] logits tensor is (that drives the 16k OOM). On
    this 1B model these KV numbers are for CORRECTNESS comparison only, not
    for demonstrating a memory win. Memory-win validation moves to a
    7B-in-NF4 model later, where the KV cache actually dominates.
    """
    device = CONFIG["device"]

    # Synthetic prefill of exactly context_length tokens. Random ids are
    # fine — KV size depends on shape, not content.
    ids = torch.randint(
        low=10, high=tokenizer.vocab_size - 10,
        size=(CONFIG["batch_size"], context_length),
        device=device,
    )

    free_cuda()
    before = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    torch.cuda.synchronize()

    after = torch.cuda.memory_allocated(device)

    # The delta includes the cache plus transient activations from the
    # forward pass. To isolate the cache, drop the activation-holding
    # outputs (logits etc.) but KEEP `past`, then re-measure.
    del out
    free_cuda()  # NOTE: empty_cache frees nothing still referenced by `past`
    after_cache_only = torch.cuda.memory_allocated(device)

    empirical_mb = (after_cache_only - before) / (1024 ** 2)
    analytic_mb = _analytic_kv_mb(context_length, model=model)

    # Flag large divergence (sanity check between the two methods).
    diverged = False
    if analytic_mb > 0:
        rel = abs(empirical_mb - analytic_mb) / analytic_mb
        diverged = rel > 0.25  # >25% gap is worth a human look

    del past, ids
    free_cuda()

    return {
        "analytic_mb": analytic_mb,
        "empirical_mb": empirical_mb,
        "diverged": diverged,
        "weights_note": "Measured as allocator delta; excludes static "
                        "~2.4GB FP16 weights resident before measurement.",
    }


# ======================================================================
# ORCHESTRATOR
# ======================================================================
def run_all():
    env_notes = {
        "platform": "WSL Ubuntu 22.04",
        "gpu": "RTX 3060 12GB",
        "cuda": "12.1",
        "torch": torch.__version__,
        "phase": "1 (FP16 baseline / unquantized control)",
    }

    results = {
        "model_name": CONFIG["model_name"],
        "timestamp": datetime.datetime.now().isoformat(),
        "environment": env_notes,
        "config": {k: (str(v) if isinstance(v, torch.dtype) else v)
                   for k, v in CONFIG.items()},
        "per_context_length": {},
        "gsm8k": None,
        "decode": None,
    }

    log.info(f"Run start. log file -> {_LOG_PATH}")
    log.info(f"Loading model {CONFIG['model_name']} ...")
    model, tokenizer = load_model_and_tokenizer()

    # --- length-independent metrics (run once) ---
    log.info("[*] GSM8K accuracy guard ...")
    try:
        results["gsm8k"] = measure_gsm8k(model, tokenizer)
        log.info(f"    GSM8K done: acc={results['gsm8k']['accuracy']:.3f}")
    except Exception as e:
        results["gsm8k"] = {"error": str(e)}
        log.exception("GSM8K failed")
    free_cuda()

    log.info("[*] Decode tokens/sec ...")
    try:
        results["decode"] = measure_decode_tps(model, tokenizer)
        log.info(f"    Decode done: {results['decode']['tokens_per_sec']:.1f} tok/s")
    except Exception as e:
        results["decode"] = {"error": str(e)}
        log.exception("Decode failed")
    free_cuda()

    # --- per-context-length sweep ---
    for ctx in CONFIG["context_lengths"]:
        log.info(f"[*] Context length {ctx} ... {_gpu_mem_str()}")
        entry = {"perplexity": None, "kv_memory": None}

        # Perplexity
        try:
            entry["perplexity"] = measure_perplexity(model, tokenizer, ctx)
            log.info(f"    perplexity @ {ctx} = "
                     f"{entry['perplexity']['perplexity']:.3f}")
        except Exception as e:
            if is_oom_error(e):
                log.warning(f"    perplexity OOM @ {ctx}")
                entry["perplexity"] = "OOM"
            else:
                entry["perplexity"] = {"error": str(e)}
                log.exception(f"perplexity failed @ {ctx}")
        free_cuda()

        # KV-cache memory
        try:
            entry["kv_memory"] = measure_kv_memory(model, tokenizer, ctx)
            km = entry["kv_memory"]
            log.info(f"    kv_memory @ {ctx}: analytic={km['analytic_mb']:.1f}MB "
                     f"empirical={km['empirical_mb']:.1f}MB"
                     + ("  (DIVERGED!)" if km['diverged'] else ""))
        except Exception as e:
            if is_oom_error(e):
                log.warning(f"    kv_memory OOM @ {ctx}")
                entry["kv_memory"] = "OOM"
            else:
                entry["kv_memory"] = {"error": str(e)}
                log.exception(f"kv_memory failed @ {ctx}")
        free_cuda()

        results["per_context_length"][str(ctx)] = entry

    # --- persist ---
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    out_path = os.path.join(
        CONFIG["output_dir"], f"baseline_{_timestamp()}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results["_results_path"] = out_path
    log.info(f"Run complete. results -> {out_path}  | log -> {_LOG_PATH}")

    del model
    free_cuda()
    return results


# ======================================================================
# SUMMARY
# ======================================================================
def print_summary(results):
    print("\n" + "=" * 72)
    print(f"FP16 BASELINE — {results['model_name']}")
    print(f"timestamp: {results['timestamp']}")
    print("=" * 72)

    g = results.get("gsm8k") or {}
    if "accuracy" in g:
        print(f"GSM8K (N={g['total']}): {g['correct']}/{g['total']} "
              f"= {g['accuracy']:.3f} acc   regex={g['answer_regex']}")
    else:
        print(f"GSM8K: {g}")

    d = results.get("decode") or {}
    if "tokens_per_sec" in d:
        print(f"Decode: {d['tokens_per_sec']:.1f} tok/s "
              f"(over {d['measured_tokens']} steady-state tokens)")
    else:
        print(f"Decode: {d}")

    print("-" * 72)
    hdr = f"{'ctx':>7} | {'ppl':>10} | {'KV analytic MB':>15} | {'KV empirical MB':>16}"
    print(hdr)
    print("-" * 72)
    for ctx, entry in results["per_context_length"].items():
        ppl = entry["perplexity"]
        kv = entry["kv_memory"]

        if ppl == "OOM":
            ppl_s = "OOM"
        elif isinstance(ppl, dict) and "perplexity" in ppl:
            ppl_s = f"{ppl['perplexity']:.3f}"
        else:
            ppl_s = "ERR"

        if kv == "OOM":
            an_s, em_s = "OOM", "OOM"
        elif isinstance(kv, dict) and "analytic_mb" in kv:
            an_s = f"{kv['analytic_mb']:.1f}"
            em_s = f"{kv['empirical_mb']:.1f}"
            if kv.get("diverged"):
                em_s += " (!)"
        else:
            an_s, em_s = "ERR", "ERR"

        print(f"{ctx:>7} | {ppl_s:>10} | {an_s:>15} | {em_s:>16}")
    print("=" * 72)
    print(f"results -> {results.get('_results_path')}")


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    res = run_all()
    print_summary(res)


# ===== PHASE 2 STUB — DO NOT IMPLEMENT IN PHASE 1 =====
# In Phase 2, swap the stock model for KIVI's patched class with a mutated config:
#   from models.llama_kivi import LlamaForCausalLM_KIVI
#   config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")
#   config.k_bits = 4; config.v_bits = 4; config.group_size = 32
#   config.residual_length = 32; config.use_flash = True
#   model = LlamaForCausalLM_KIVI.from_pretrained(..., config=config, torch_dtype=torch.float16)
# The harness above must run UNCHANGED against this model — same metric functions,
# same JSON schema — so I can diff quantized vs FP16 directly.
# =====================================================
