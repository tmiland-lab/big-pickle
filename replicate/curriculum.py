from __future__ import annotations

import json
import os
import sys


def load_seeds(seeds_dir: str) -> list[dict]:
    """Load teacher-curriculum seed files (*.json) from a directory.

    Schema per file:
      {"persona": "code", "examples": [
         {"instruction": "...", "thinking": "reasoning trace", "response": "final answer"}
      ]}
    """
    examples = []
    if not os.path.isdir(seeds_dir):
        return examples
    for fname in sorted(os.listdir(seeds_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(seeds_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[curriculum] skip {fname}: {e}", file=sys.stderr)
            continue
        persona = data.get("persona", "chat")
        examples.append({"persona": persona, "path": fname, "examples": data.get("examples", [])})
    return examples


def to_sharegpt(seeds: list[dict], include_thinking: bool = True) -> list[dict]:
    """Convert seed examples to sharegpt SFT lines.

    When include_thinking, the response becomes "thinking trace\n\nfinal answer"
    so the local model learns the reasoning path, not just the output.
    """
    dataset = []
    for seed in seeds:
        for ex in seed.get("examples", []):
            instruction = str(ex.get("instruction", "")).strip()
            response = str(ex.get("response", "")).strip()
            if not instruction or not response:
                continue
            thinking = str(ex.get("thinking", "")).strip()
            if include_thinking and thinking:
                gpt_value = f"{thinking}\n\n{response}"
            else:
                gpt_value = response
            dataset.append({
                "persona": seed.get("persona", "chat"),
                "conversations": [
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": gpt_value},
                ],
            })
    return dataset


def build_curriculum(seeds_dir: str, output_path: str, include_thinking: bool = True) -> int:
    seeds = load_seeds(seeds_dir)
    dataset = to_sharegpt(seeds, include_thinking=include_thinking)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[curriculum] wrote {len(dataset)} teacher examples -> {output_path}")
    return len(dataset)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build teacher-curriculum SFT dataset")
    p.add_argument("--seeds", default=None, help="Seeds dir (default: data/curriculum, falls back to examples/curriculum)")
    p.add_argument("--output", default=None, help="Output jsonl (default: data/sft-curriculum.jsonl)")
    p.add_argument("--no-thinking", action="store_true", help="Omit the thinking trace (responses only)")
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seeds = args.seeds
    if not seeds:
        personal = os.path.join(root, "data", "curriculum")
        seeds = personal if os.path.isdir(personal) else os.path.join(root, "examples", "curriculum")
    out = args.output or os.path.join(root, "data", "sft-curriculum.jsonl")
    build_curriculum(seeds, out, include_thinking=not args.no_thinking)