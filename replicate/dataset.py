from __future__ import annotations

import json
import os
import re
import sys


def load_conversations(dir_path: str) -> list[list[dict]]:
    all_transcripts = []
    if not os.path.isdir(dir_path):
        return all_transcripts
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".jsonl"):
            continue
        msgs = []
        with open(os.path.join(dir_path, fname), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                    msgs.append(m)
                except json.JSONDecodeError:
                    continue
        if len(msgs) >= 2:
            all_transcripts.append(msgs)
    return all_transcripts


def to_sharegpt(transcripts: list[list[dict]]) -> list[dict]:
    """Convert message transcripts to sharegpt format for Unsloth SFT."""
    dataset = []
    for msgs in transcripts:
        pairs = []
        pending_user = None
        for m in msgs:
            role = m.get("role")
            content = str(m.get("content", "")).strip()
            if role == "user":
                if content and not content.startswith("Tool ") and " result:" not in content:
                    if pending_user is None:
                        pending_user = content
                    else:
                        pending_user = content
                continue
            if role == "assistant":
                if content.startswith("TOOL CALL"):
                    continue
                if pending_user is not None and content:
                    pairs.append({"from": "human", "value": pending_user})
                    pairs.append({"from": "gpt", "value": content})
                    pending_user = None
        if pairs:
            dataset.append({"conversations": pairs})
    return dataset


def deduplicate(dataset: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for item in dataset:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def filter_min_turns(dataset: list[dict], min_pairs: int = 1) -> list[dict]:
    return [d for d in dataset if len(d.get("conversations", [])) >= min_pairs]


def build_dataset(transcripts_path: str, output_path: str, min_pairs: int = 1, dedup: bool = True, extra_path: str = "") -> int:
    transcripts = load_conversations(transcripts_path)
    dataset = to_sharegpt(transcripts)
    if extra_path and os.path.exists(extra_path):
        with open(extra_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item.get("conversations"), list):
                    dataset.append(item)
        print(f"[dataset] + {sum(1 for _ in open(extra_path, encoding='utf-8') if _.strip())} teacher examples from {extra_path}")
    before = len(dataset)
    dataset = filter_min_turns(dataset, min_pairs)
    if dedup:
        dataset = deduplicate(dataset)
    print(f"[dataset] {before} sources -> {len(dataset)} examples after filters (dedup={dedup})")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(dataset)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Transcript directory (default: data/conversations)")
    p.add_argument("--output", default=None, help="Output JSONL path (default: data/sft-dataset.jsonl)")
    p.add_argument("--curriculum", default=None, help="Teacher curriculum JSONL to merge (default: data/sft-curriculum.jsonl)")
    p.add_argument("--min-pairs", type=int, default=1)
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    inp = args.input or os.path.join(project_dir, "data", "conversations")
    out = args.output or os.path.join(project_dir, "data", "sft-dataset.jsonl")
    extra = args.curriculum if args.curriculum is not None else os.path.join(project_dir, "data", "sft-curriculum.jsonl")
    n = build_dataset(inp, out, args.min_pairs, extra_path=extra)
    print(f"[dataset] wrote {n} examples to {out}")