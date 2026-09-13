#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== Big Pickle training env setup (gfx1030 / RX 6900 XT) =="
echo "Detecting GPU..."
if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "[warn] rocm-smi not found — only CPU training possible (very slow)."
  echo "       Install ROCm first: https://rocm.docs.amd.com/"
  echo "       Kernel driver (kfd) is already present if Ollama ROCm works."
fi

PY="${VENV_PY:-python3}"
echo "== Step 1: venv =="
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  echo "venv created at .venv"
else
  echo "venv exists (.venv)"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "== Step 2: PyTorch ROCm 6.2 (last prebuilt with gfx1030 support) =="
pip install --upgrade pip
pip install torch==2.5.1+rocm6.2 torchvision --index-url https://download.pytorch.org/whl/rocm6.2

echo "== Step 3: unsloth (QLoRA, AMD-compatible) =="
# unsloth auto-selects the right torch backend on install from PyPI wheels
pip install "unsloth[colab-amd] @ git+https://github.com/unslothai/unsloth.git"

echo "== Step 4: datasets, trl, peft, bitsandbytes =="
pip install datasets peft trl sentencepiece protobuf "bitsandbytes>=0.43.1"

echo "== Step 5: gfx1030 env override (every launch) =="
echo "export HSA_OVERRIDE_GFX_VERSION=10.3.0" > .rocm-env
echo "export OMP_NUM_THREADS=4" >> .rocm-env
echo "export OPENBLAS_NUM_THREADS=4" >> .rocm-env
echo "export MKL_NUM_THREADS=4" >> .rocm-env
echo "wrote .rocm-env (source it before training, or run train.py which sets these)"

echo
echo "== DONE =="
echo "  - source .venv/bin/activate"
echo "  - source .rocm-env"
echo "  - python replicate/train.py --dry-run   # verify config + headroom"
echo "  - python replicate/dataset.py           # build SFT dataset from captured sessions"
echo "  - python replicate/train.py             # train QLoRA adapter"