#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import json
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def detect_vram_gb() -> float:
    import glob
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=5, text=True, stderr=subprocess.DEVNULL,
        )
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if lines:
            return float(lines[0]) / 1024
    except Exception:
        pass
    try:
        for f in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
            with open(f) as fh:
                return int(fh.read().strip()) / (1024 ** 3)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            timeout=5, text=True, stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
        total = data[0].get("vram", [{}])[0].get("Total Memory (bytes)", 0)
        return total / (1024 ** 3)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["lspci", "-v"], timeout=5, text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"\[size=(\d+)(G|M)\]", out)
        if m:
            val = float(m.group(1))
            return val if m.group(2) == "G" else val / 1024
    except Exception:
        pass
    return 0.0


def detect_ram_gb() -> float:
    try:
        out = subprocess.check_output(["free", "-g"], timeout=3, text=True)
        for line in out.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                return float(parts[1])
    except Exception:
        pass
    return 0.0


def get_system_mem_free_gb() -> float:
    try:
        out = subprocess.check_output(["free", "-g"], timeout=3, text=True)
        for line in out.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                return float(parts[5])
    except Exception:
        pass
    return 0.0


def get_disk_free_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return -1.0


def get_vram_free_gb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=5, text=True, stderr=subprocess.DEVNULL,
        )
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if lines:
            return float(lines[0]) / 1024
    except Exception:
        pass
    try:
        with open("/sys/class/drm/card0/device/mem_info_vram_used") as fh:
            used = int(fh.read().strip()) / (1024 ** 3)
        return detect_vram_gb() - used
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            timeout=5, text=True, stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
        free = data[0].get("vram", [{}])[0].get("Free Memory (bytes)", 0)
        return free / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def check_headroom(headroom_gb: float = 5.0, model_gb: float = 6.0, desktop_safe: bool = True,
                   data_path: str = "", output_path: str = "") -> bool:
    vram_total = detect_vram_gb()
    ram_total = detect_ram_gb()
    vram_free = get_vram_free_gb()
    ram_free = get_system_mem_free_gb()

    print(f"[train] system: {ram_total:.0f}GB RAM, {vram_total:.0f}GB VRAM")
    print(f"[train] free: {ram_free:.0f}GB RAM, {vram_free:.1f}GB VRAM")

    disk_free = get_disk_free_gb(data_path or ROOT)
    disk_free_out = get_disk_free_gb(output_path or ROOT)
    disk_min_gb = 10.0
    if disk_free >= 0:
        print(f"[train] free disk: {disk_free:.1f}GB ({data_path or ROOT})")
        if disk_free < disk_min_gb:
            print(f"[train] BLOCKED: only {disk_free:.0f}GB free on data dir (need >= {disk_min_gb:.0f}GB)")
            return False
    if disk_free_out >= 0:
        print(f"[train] free disk: {disk_free_out:.1f}GB ({output_path or ROOT})")
        if disk_free_out < disk_min_gb:
            print(f"[train] BLOCKED: only {disk_free_out:.0f}GB free on output dir (need >= {disk_min_gb:.0f}GB)")
            return False

    if vram_total > 0:
        headroom = vram_total - model_gb
        if headroom < headroom_gb:
            print(f"[train] BLOCKED: model needs ~{model_gb:.0f}GB, but headroom reserve is {headroom_gb:.0f}GB")
            print(f"  available headroom: {headroom:.1f}GB (target: {headroom_gb:.0f}GB)")
            print(f"  reduce headroom with --headroom-gb or stop GPU-heavy tasks")
            return False
    else:
        print("[train] WARNING: VRAM not detected — running CPU-only, training will be slow")

    if desktop_safe:
        if ram_free < 2.0 and ram_total > 0:
            print(f"[train] BLOCKED: only {ram_free:.1f}GB RAM free — desktop will hang")
            return False

    print(f"[train] headroom check PASSED")
    return True


def _hash_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _read_stamp(stamp_path: str) -> tuple[str | None, int]:
    try:
        with open(stamp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("digest"), int(data.get("count", 0))
    except Exception:
        return None, 0


def _write_stamp(stamp_path: str, digest: str, count: int) -> None:
    try:
        with open(stamp_path, "w", encoding="utf-8") as f:
            json.dump({"digest": digest, "count": count, "trained_at": time.strftime("%Y-%m-%d %H:%M")}, f)
    except OSError as e:
        print(f"[train] WARNING: could not write stamp: {e}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Big Pickle QLoRA trainer — gfx1030 RX 6900 XT optimized")
    p.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct", help="Base model for QLoRA SFT")
    p.add_argument("--data", default=None, help="SFT JSONL dataset (default: data/sft-dataset.jsonl)")
    p.add_argument("--output", default=None, help="Output adapter dir (default: data/lora-output)")
    p.add_argument("--epochs", type=int, default=2, help="Training epochs")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--r", type=int, default=16, help="LoRA rank")
    p.add_argument("--max-seq", type=int, default=2048, help="Max sequence length")
    p.add_argument("--batch", type=int, default=1, help="Micro batch size")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--headroom-gb", type=float, default=5.0, help="Reserve this much VRAM for desktop")
    p.add_argument("--desktop-safe", action="store_true", default=True, help="Desktop-safe mode (thread cap + grad ckpt)")
    p.add_argument("--no-desktop-safe", action="store_false", dest="desktop_safe")
    p.add_argument("--dry-run", action="store_true", help="Check headroom and print config, don't train")
    p.add_argument("--venv", default=None, help="Path to venv with torch+unsloth (auto-detect if omitted)")
    p.add_argument("--workers", type=int, default=1, help="DataLoader workers (desktop-safe: keep low)")
    p.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N steps")
    p.add_argument("--force", action="store_true", help="Retrain even if dataset since last train is unchanged")
    args = p.parse_args()

    if args.desktop_safe:
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        os.environ.setdefault("GOTO2_NUM_THREADS", "4")
        if args.batch > 1:
            print("[train] desktop-safe: forcing batch=1")
            args.batch = 1

    data_path = args.data or os.path.join(ROOT, "data", "sft-dataset.jsonl")
    output_path = args.output or os.path.join(ROOT, "data", "lora-output")
    stamp_path = os.path.join(ROOT, "data", ".trained-stamp")

    if not os.path.exists(data_path):
        print(f"[train] dataset not found: {data_path}", file=sys.stderr)
        print("  run first: python replicate/dataset.py", file=sys.stderr)
        return 1

    with open(data_path, "r") as f:
        n_examples = sum(1 for _ in f if _.strip())
    print(f"[train] dataset: {n_examples} examples")

    data_digest = _hash_file(data_path)
    last_digest, last_count = _read_stamp(stamp_path)
    if last_digest:
        if last_digest == data_digest:
            print(f"[train] dataset unchanged since last train ({last_count} ex, {os.path.basename(stamp_path)})")
            if not args.force:
                print("[train] SKIPPING (no new data). Use --force to retrain anyway.")
                return 0
            print("[train] --force: retraining on unchanged data")
        else:
            print(f"[train] data changed since last train ({last_count} -> {n_examples} examples); proceeding")

    if not check_headroom(args.headroom_gb, desktop_safe=args.desktop_safe,
                          data_path=data_path, output_path=output_path):
        return 1

    if args.dry_run:
        print(json.dumps({
            "model": args.model,
            "data": data_path,
            "output": output_path,
            "epochs": args.epochs,
            "lr": args.lr,
            "r": args.r,
            "max_seq": args.max_seq,
            "batch": args.batch,
            "grad_accum": args.grad_accum,
            "desktop_safe": args.desktop_safe,
            "workers": args.workers,
        }, indent=2))
        return 0

    venv = args.venv
    if not venv:
        candidates = [
            os.path.join(ROOT, ".venv"),
            os.path.expanduser("~/.venv"),
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "bin", "activate")):
                venv = c
                break
    if not venv or not os.path.exists(os.path.join(venv, "bin", "activate")):
        print("[train] ERROR: no venv with torch+unsloth found", file=sys.stderr)
        print("  run: bash setup-training.sh", file=sys.stderr)
        print("  or pass --venv /path/to/venv", file=sys.stderr)
        return 1

    python = os.path.join(venv, "bin", "python")
    print(f"[train] venv: {venv}")
    print(f"[train] model: {args.model} -> {output_path}")
    print(f"[train] config: r={args.r}, lr={args.lr}, epochs={args.epochs}, batch={args.batch}x{args.grad_accum}")

    train_code = f'''
import os
import subprocess

def _is_amd():
    try:
        ok = subprocess.run(["rocm-smi", "--version"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=5).returncode == 0
    except Exception:
        ok = False
    return ok or any(fn.startswith("kfd") for fn in os.listdir("/dev"))

if _is_amd():
    # AMD gfx1030 (RX 6900 XT): pin the ROCm fallback layer
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    _use_bf16 = False
else:
    # NVIDIA: nothing to override, just report
    print("[train] backend: CUDA (NVIDIA)")
    _use_bf16 = True

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

print("[train] loading model: __MODEL__")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="__MODEL__",
    max_seq_length=__MAX_SEQ__,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=__R__,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=__R__,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

dataset = load_dataset("json", data_files="__DATA__", split="train")
print("[train] loaded", len(dataset), "examples")

def format_example(example):
    msgs = example.get("conversations", [])
    text = ""
    for m in msgs:
        role = m.get("from", "human")
        if role == "human":
            text += "<|user|>\\n" + m.get("value", "") + "\\n"
        elif role == "gpt":
            text += "<|assistant|>\\n" + m.get("value", "") + "\\n"
    return {{"text": text + "<|end|>\\n"}}

formatted = dataset.map(format_example, num_proc=1)

training_args = SFTConfig(
    output_dir="__OUTPUT__",
    num_train_epochs=__EPOCHS__,
    per_device_train_batch_size=__BATCH__,
    gradient_accumulation_steps=__GRAD_ACCUM__,
    learning_rate=__LR__,
    max_seq_length=__MAX_SEQ__,
    logging_steps=5,
    save_steps=__CHECKPOINT_EVERY__,
    save_total_limit=3,
    fp16=not _use_bf16,
    bf16=_use_bf16,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    seed=3407,
    report_to="none",
    dataloader_num_workers=__WORKERS__,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted,
    args=training_args,
    packing=True,
)
trainer.train()

model.save_pretrained("__OUTPUT__")
tokenizer.save_pretrained("__OUTPUT__")
print("[train] saved adapter to __OUTPUT__")
'''.replace(
        "__MODEL__", args.model,
    ).replace(
        "__MAX_SEQ__", str(args.max_seq),
    ).replace(
        "__R__", str(args.r),
    ).replace(
        "__DATA__", data_path,
    ).replace(
        "__OUTPUT__", output_path,
    ).replace(
        "__EPOCHS__", str(args.epochs),
    ).replace(
        "__BATCH__", str(args.batch),
    ).replace(
        "__GRAD_ACCUM__", str(args.grad_accum),
    ).replace(
        "__LR__", repr(args.lr),
    ).replace(
        "__CHECKPOINT_EVERY__", str(args.checkpoint_every),
    ).replace(
        "__WORKERS__", str(args.workers),
    )

    env = os.environ.copy()
    if args.desktop_safe:
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(k, "4")

    start = time.time()
    proc = subprocess.Popen(
        [python, "-c", train_code],
        cwd=ROOT, stdout=sys.stdout, stderr=sys.stderr, env=env,
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[train] interrupted — saving checkpoint before exit...")
        proc.terminate()
        proc.wait()

    elapsed = time.time() - start
    if proc.returncode == 0:
        _write_stamp(stamp_path, data_digest, n_examples)
        print(f"[train] complete in {elapsed:.0f}s")
        print(f"[train] adapter saved to: {output_path}")
        print(f"[train] test it: bp --lora {output_path} chat")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main() or 0)