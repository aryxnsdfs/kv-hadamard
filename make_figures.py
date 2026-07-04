#!/usr/bin/env python3
"""
Publication figures for the KV-Hadamard result. All numbers below are the
MEASURED values from this project's runs (sources noted per block):
  - 7B cache/no-cache PPL: robust_sweep_run2.log (main sweep, completed)
  - long-context + TinyLlama: baseline_results/robust_sweep_resumed.json
  - B2 packed memory/speed: baseline_results/b2_packed_*.json
  - fp16 KV @2048 = 1024 MB: phase2 nf4 baseline

Produces PNGs in figures/. Run in the venv:  python make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

C = {"kivi": "#444", "had_int2": "#1a7f3c", "ternary": "#b3341f",
     "had_tern": "#c47a15", "fp16": "#8a8a8a"}

# ---- 7B cache-path PPL, per passage/length (from run2 log) ----
passages = ["P0/896", "P0/2048", "P1/896", "P1/2048", "P2/896", "P2/2048"]
kivi_cache = [3.6595, 3.8395, 5.3027, 6.5815, 4.1106, 4.1545]
kivi_noc   = [3.6416, 3.8360, 5.2548, 6.5600, 4.1073, 4.1472]
tern_cache = [3.9013, 4.4285, 6.1426, 7.3108, 4.5089, 4.5560]
hi2_v32    = [3.6548, 3.8544, 5.3804, 6.6373, 4.1187, 4.1526]
hi2_v0     = [3.6786, 3.9520, 5.5604, 6.7829, 4.1842, 4.2249]

# ---- TinyLlama (resumed.json) ----
tl_pass = ["P0/896", "P0/1792", "P1/896", "P1/1792"]
tl_kivi = [4.836, 4.830, 7.965, 8.884]
tl_hi2  = [4.914, 4.916, 8.083, 8.984]
tl_tern = [7.272, 7.164, 11.742, 12.623]


# ======================================================================
# FIG 1 — the measurement bug: blind (no-cache) hides everything.
# ======================================================================
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(passages)); w = 0.35
ax.bar(x - w/2, kivi_noc, w, label="no-cache (blind eval)", color="#bbb", edgecolor="k", linewidth=.5)
ax.bar(x + w/2, kivi_cache, w, label="cache path (real eval)", color=C["kivi"], edgecolor="k", linewidth=.5)
ax.set_xticks(x); ax.set_xticklabels(passages, rotation=20)
ax.set_ylabel("Perplexity (KIVI int4-V)")
ax.set_title("Fig 1. Blind vs cache-path evaluation\n(the metric that never touched the quantized cache)")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_measurement_bug.png"); plt.close(fig)

# The sharper version: under blind eval, ALL schemes collapse to one number.
fig, ax = plt.subplots(figsize=(6.5, 4.2))
schemes = ["KIVI\nint4", "Ternary\n1.58b", "Had-INT2\n2b"]
blind = [3.6416, 3.6416, 3.6416]        # identical — probe_cache_ppl result
real  = [3.6595, 3.9013, 3.6548]
x = np.arange(3); w = 0.35
ax.bar(x - w/2, blind, w, label="blind eval (use_cache=False)", color="#ccc", edgecolor="k", linewidth=.5)
ax.bar(x + w/2, real,  w, label="cache-path eval (real)", color=["#444", C["ternary"], C["had_int2"]],
       edgecolor="k", linewidth=.5)
ax.set_xticks(x); ax.set_xticklabels(schemes)
ax.set_ylim(3.5, 4.0); ax.set_ylabel("Perplexity @ P0/896")
ax.set_title("Fig 1b. Blind eval scores every scheme identically (3.6416)\nonly the cache path reveals the real differences")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig1b_blind_collapse.png"); plt.close(fig)

# ======================================================================
# FIG 2 — main result: PPL by scheme across passages (7B).
# ======================================================================
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(passages)); w = 0.2
ax.bar(x - 1.5*w, kivi_cache, w, label="KIVI int4 (4b, ref)", color=C["kivi"], edgecolor="k", linewidth=.4)
ax.bar(x - 0.5*w, hi2_v32,    w, label="Had-INT2 v32 (2b)", color=C["had_int2"], edgecolor="k", linewidth=.4)
ax.bar(x + 0.5*w, hi2_v0,     w, label="Had-INT2 v0 (2b, no shield)", color="#5cbf82", edgecolor="k", linewidth=.4)
ax.bar(x + 1.5*w, tern_cache, w, label="Ternary v0 (1.58b)", color=C["ternary"], edgecolor="k", linewidth=.4)
ax.set_xticks(x); ax.set_xticklabels(passages, rotation=20)
ax.set_ylabel("Cache-path perplexity"); ax.set_title("Fig 2. Llama-2-7B — Had-INT2 tracks KIVI at half the bits; ternary does not")
ax.legend(ncol=2, fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_main_7b.png"); plt.close(fig)

# ======================================================================
# FIG 3 — bits-vs-quality tradeoff (the money figure).
# ======================================================================
fig, ax = plt.subplots(figsize=(6.5, 4.6))
# x = bits/elem, y = PPL @ P0/896 (7B). fp16 approximated by no-cache (exact V).
pts = [("fp16 (exact V)", 16, 3.6416, C["fp16"]),
       ("KIVI int4", 4, 3.6595, C["kivi"]),
       ("Had-INT2 (ours)", 2, 3.6548, C["had_int2"]),
       ("Ternary", 1.58, 3.9013, C["ternary"])]
for name, b, p, c in pts:
    ax.scatter(b, p, s=140, color=c, edgecolor="k", zorder=3)
    ax.annotate(name, (b, p), textcoords="offset points", xytext=(8, 6), fontsize=9)
ax.axhline(3.6595, ls="--", color=C["kivi"], alpha=.5, label="KIVI quality line")
ax.set_xlabel("bits per V element"); ax.set_ylabel("Perplexity @ P0/896 (7B)")
ax.set_xlim(0.8, 17); ax.set_title("Fig 3. Bits vs quality — Had-INT2 hits KIVI quality at 2 bits;\nternary falls off the line below 2 bits")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_bits_vs_quality.png"); plt.close(fig)

# ======================================================================
# FIG 4 — REAL measured KV memory (B2).
# ======================================================================
fig, ax = plt.subplots(figsize=(6, 4.4))
labels = ["fp16 KV", "KIVI int4", "B2 packed\nHad-INT2 (ours)"]
mem = [1024, 327.5, 263.6]
cols = [C["fp16"], C["kivi"], C["had_int2"]]
bars = ax.bar(labels, mem, color=cols, edgecolor="k", linewidth=.5)
for b, m in zip(bars, mem):
    ax.text(b.get_x()+b.get_width()/2, m+12, f"{m:.0f} MB", ha="center", fontsize=10)
ax.set_ylabel("KV cache @2048 ctx (MB, measured)")
ax.set_title("Fig 4. Measured KV memory — real packed 2-bit storage\n3.9x smaller than fp16, 20% smaller than KIVI")
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_memory.png"); plt.close(fig)

# ======================================================================
# FIG 5 — cross-model robustness (gap over KIVI, %).
# ======================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.4))
def gap(a, b): return [100*(x/y - 1) for x, y in zip(a, b)]
hi2_gap_7b = gap(hi2_v32, kivi_cache)
tern_gap_7b = gap(tern_cache, kivi_cache)
hi2_gap_tl = gap(tl_hi2, tl_kivi)
tern_gap_tl = gap(tl_tern, tl_kivi)
groups = ["Had-INT2\n7B", "Had-INT2\nTinyLlama", "Ternary\n7B", "Ternary\nTinyLlama"]
means = [np.mean(hi2_gap_7b), np.mean(hi2_gap_tl), np.mean(tern_gap_7b), np.mean(tern_gap_tl)]
cols = [C["had_int2"], "#5cbf82", C["ternary"], "#e07a5f"]
bars = ax.bar(groups, means, color=cols, edgecolor="k", linewidth=.5)
for b, m in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, m+0.4, f"+{m:.1f}%", ha="center", fontsize=10)
ax.axhline(0, color="k", lw=.8)
ax.set_ylabel("Mean PPL gap over KIVI (%)")
ax.set_title("Fig 5. Cross-model — Had-INT2 stays within ~2% of KIVI on both;\nternary's gap explodes on the smaller model")
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_cross_model.png"); plt.close(fig)

print("wrote:")
for f in sorted(os.listdir(FIG)):
    print(f"  {FIG}/{f}")
