#!/usr/bin/env bash
# ======================================================================
# Phase 2 setup — Llama-2-7B, KIVI-native, ONE pinned venv for all three
# baselines (fp16 / nf4 / kivi). Run inside WSL:  bash setup_phase2.sh
#
# Why ONE venv works here: Llama-2 (unlike Llama-3.2) is supported by KIVI's
# pinned transformers, AND bitsandbytes NF4 works on that transformers too. So
# fp16, nf4 and kivi all run in the same environment = perfect parity, no
# version fight. That is the whole reason we chose Llama-2-7B (path 3).
#
# This step gets STOCK KIVI 4-bit KV working on 7B-NF4 locally + the two
# reference baselines. NO ternary kernel yet. Read PHASE2_README.md.
# ======================================================================
set -e

CUDA_DIR="/mnt/c/Users/aryan/Downloads/cuda"
KIVI_REPO="${KIVI_REPO:-https://github.com/jy-yuan/KIVI.git}"
KIVI_DIR="$HOME/KIVI"
VENV="$HOME/kvenv_phase2"

echo "==> 1. GPU + nvcc check"
nvidia-smi || { echo "no GPU in WSL"; exit 1; }
if ! command -v nvcc >/dev/null 2>&1; then
  echo "!! nvcc (CUDA toolkit) not found — needed to compile KIVI kernels."
  echo "   Install a CUDA toolkit matching your driver, e.g.:"
  echo "     sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-1"
  echo "   Re-run this script after."
  exit 1
fi
nvcc --version | tail -1

echo "==> 2. clone KIVI"
[ -d "$KIVI_DIR" ] && echo "   $KIVI_DIR exists, skipping" \
                   || git clone "$KIVI_REPO" "$KIVI_DIR"

echo "==> 3. fresh venv: $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools ninja packaging

echo "==> 4. torch (CUDA 12.1)"
pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cu121

echo "==> 5. transformers pin (the ONE version that matters) + our deps"
# KIVI's requirements.txt is over-constrained for paper repro (pins torch==2.1.2,
# lm_eval, etc.) and breaks on modern Python. We only need the transformers
# version KIVI's patched modeling_llama targets. Extract just that pin and
# install it, NOT the whole file. Llama-2 is supported by it, so all three
# baselines share this version.
# Use KIVI's exact tested combo (transformers/accelerate/bitsandbytes) so we
# match the versions its kernels + patched modeling were validated against.
# Llama-2 is supported by transformers 4.36.2 -> all three baselines share it.
pip install transformers==4.36.2 accelerate==0.25.0 bitsandbytes==0.43.0
# data deps (unpinned is fine; newer datasets needs namespaced dataset IDs,
# which the harness already uses: openai/gsm8k, Salesforce/wikitext).
pip install datasets sentencepiece protobuf

echo "==> 5b. gcc-12 for nvcc (CUDA 12.1 rejects gcc>12)"
# CUDA 12.1's nvcc only supports gcc <= 12; modern Ubuntu defaults to gcc 13+,
# which fails with 'unsupported GNU version'. Install gcc-12 and point the CUDA
# host compiler at it for both the KIVI kernel build and flash-attn.
if ! command -v gcc-12 >/dev/null 2>&1; then
  sudo apt-get install -y gcc-12 g++-12
fi
export CC=gcc-12 CXX=g++-12 CUDAHOSTCXX=g++-12

echo "==> 6. build KIVI CUDA quant kernels"
# --no-build-isolation: KIVI/quant/setup.py imports torch at build time to
# compile the CUDA extension. Build isolation hides the venv's torch, giving
# 'No module named torch'. Disable isolation so it sees the installed torch.
if [ -d "$KIVI_DIR/quant" ]; then
  ( cd "$KIVI_DIR/quant" && pip install -e . --no-build-isolation )
else
  echo "!! $KIVI_DIR/quant not found — check the KIVI repo layout"
fi

echo "==> 7. flash-attention 2.5.6 (KIVI's pin, use_flash=True). Slow build (~10-20 min)."
pip install flash-attn==2.5.6 --no-build-isolation || \
  echo "!! flash-attn build failed — set KIVI_CONFIG['use_flash']=False in "\
       "phase2_kivi_harness.py to skip (KIVI has a non-flash path), or fix nvcc/ninja"

echo "==> 8. make KIVI importable + sanity checks"
export PYTHONPATH="$KIVI_DIR:$PYTHONPATH"
echo "export PYTHONPATH=$KIVI_DIR:\$PYTHONPATH" >> "$VENV/bin/activate"
python3 -c "import torch,bitsandbytes,transformers as t; \
print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0)); \
print('transformers',t.__version__)"
python3 - <<'PY'
try:
    from models.llama_kivi import LlamaForCausalLM_KIVI
    print("KIVI import OK")
except Exception as e:
    print("KIVI import FAILED:", e, "\n-> see PHASE2_README.md")
PY

cat <<EOF

==> setup done. Next (same venv):
   source $VENV/bin/activate
   cd $CUDA_DIR
   python3 phase2_kivi_harness.py --baseline nf4    # NF4 weights + FP16 KV (local)
   python3 phase2_kivi_harness.py --baseline kivi   # NF4 weights + KIVI 4-bit KV (local)
   python3 phase2_kivi_harness.py --baseline fp16   # FP16 weights (cloud/big-GPU only)
   python3 phase2_kivi_harness.py --compare         # diff the three
EOF
