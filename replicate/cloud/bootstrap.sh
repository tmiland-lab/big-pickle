#!/usr/bin/env bash
# Big Pickle cloud GPU bootstrap — runs ONCE on the setup droplet (as user-data)
# to prepare a reusable snapshot: CUDA torch + unsloth + deps + Qwen2.5-7B model
# + warm compile cache. Future `train --cloud` runs deploy from the snapshot
# and skip all of this.
set -euo pipefail
exec > /root/bootstrap.log 2>&1

echo "=== bootstrap started $(date -Is) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git curl > /dev/null

echo "=== venv ==="
python3 -m venv /opt/bpvenv
source /opt/bpvenv/bin/activate
pip install --upgrade pip wheel setuptools > /dev/null

echo "=== torch CUDA ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    2>&1 | tail -3

echo "=== unsloth ==="
pip install \
  "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git" \
  datasets trl peft bitsandbytes sentencepiece protobuf 2>&1 | tail -3

echo "=== verify torch + GPU ==="
python - <<'PY'
import torch
print("[bootstrap] torch", torch.__version__)
print("[bootstrap] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[bootstrap] GPU:", torch.cuda.get_device_name(0))
    print("[bootstrap] vram:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
PY

echo "=== prefetch Qwen2.5-7B-Instruct ==="
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("unsloth/Qwen2.5-7B-Instruct")
print("[bootstrap] model cached at:", p)
PY

echo "=== warm unsloth compile cache (load first 4-bit model) ==="
python - <<'PY'
import os
os.environ["UNSLOTH_COMPILED_CACHE"] = "/opt/unsloth-cache"
from unsloth import FastLanguageModel
m, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
m = FastLanguageModel.get_peft_model(
    m, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
del m
import gc; gc.collect()
print("[bootstrap] unsloth compile cache warmed")
PY

echo "=== bootstrap complete $(date -Is) ==="
echo "MARKER_BOOTSTRAP_OK=1" > /root/bootstrap-status
touch /root/bootstrap.done