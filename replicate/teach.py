#!/usr/bin/env python3
"""Teaching loop: make the child learn a lesson the way a kid learns.

For each lesson:
  1. Ask the bot the instruction (the thing we want it to learn).
  2. Await its answer.
  3. Deliver the gold answer (Big Pickle's known answer) as a teacher correction.
  4. Ask it to re-answer in its own words.
  5. The bot's REVISED answer is the closest it has come to thinking like me —
     capture it as a training seed AND record the whole exchange as a conversation.

Run: `bp teach` (uses the seed bank's lessons) or `bp teach "<custom lesson>"`.

Output:
  data/curriculum/taught-<persona>.json   <- revised answers, folded into SFT by `bp build`
  data/conversations/*-teach.jsonl         <- full dialogue, folded by `bp build`
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bp.brain import Brain, load_brain, BrainResult  # noqa: E402
from bp.personas import system_prompt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config() -> dict:
    with open(os.path.join(ROOT, "bp.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def lessons_from_seeds(seed_files: list[str], max_per_persona: int) -> list[dict]:
    """Read the seed banks (core/code/know/write/chat) as lessons.

    Each lesson = {"instruction", "thinking", "response"} where response is the gold answer.
    """
    lessons = []
    per_persona: dict[str, int] = {}
    for path in seed_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        persona = data.get("persona", "chat")
        for ex in data.get("examples", []):
            if max_per_persona and per_persona.get(persona, 0) >= max_per_persona:
                break
            lessons.append({
                "persona": persona,
                "instruction": str(ex.get("instruction", "")).strip(),
                "thinking": str(ex.get("thinking", "")).strip(),
                "response": str(ex.get("response", "")).strip(),
            })
            per_persona[persona] = per_persona.get(persona, 0) + 1
    return lessons


def ask(brain: Brain, model: str, persona: str, instruction: str, temperature: float) -> BrainResult:
    msgs = [{"role": "system", "content": system_prompt(persona)},
            {"role": "user", "content": instruction}]
    return brain.chat(msgs, model=model, temperature=temperature)


def correct(brain: Brain, model: str, persona: str, instruction: str, first_answer: str, gold: str, temperature: float = 0.5) -> tuple[BrainResult, list]:
    """Teacher correction round. Returns (revised result, full message log)."""
    msgs = [{"role": "system", "content": system_prompt(persona)},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": first_answer},
            {"role": "user", "content":
             "Teacher feedback: the ideal answer is the following. Read it carefully, "
             "then re-answer the original question in your own words, concise and complete. "
             "Do not copy word-for-word; internalize it.\n\n" + gold},
            ]
    revised = brain.chat(msgs, model=model, temperature=temperature - 0.1)
    return revised, msgs


def write_seed(seeds_dir: str, persona: str, example: dict) -> None:
    os.makedirs(seeds_dir, exist_ok=True)
    path = os.path.join(seeds_dir, f"taught-{persona}.json")
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f).get("examples", [])
        except (json.JSONDecodeError, OSError):
            existing = []
    seen = {e.get("instruction", "") for e in existing}
    if example.get("instruction", "") not in seen:
        existing.append(example)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"persona": persona, "examples": existing}, f, indent=2, ensure_ascii=False)


def write_conversation(cap_dir: str, sid: str, log: list, lesson: dict, first: str, revised: str) -> None:
    """Persist the teaching dialogue as a conversation capture."""
    os.makedirs(cap_dir, exist_ok=True)
    path = os.path.join(cap_dir, f"{sid}-teach.jsonl")
    role_map = {"user": "user", "assistant": "assistant", "system": "system"}
    with open(path, "w", encoding="utf-8") as f:
        for m in log:
            f.write(json.dumps({"role": role_map.get(m["role"], "user"), "content": m["content"]}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"role": "system", "content": f"teacher-gold << {lesson['thinking']} >> {lesson['response']}"}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"role": "assistant", "content": revised}, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Teach the child via question -> attempt -> correction -> re-answer")
    p.add_argument("prompts", nargs="*", help="Custom lessons; if none, use the seed bank's lessons")
    p.add_argument("--model", default="", help="Bot model (default: chat persona model)")
    p.add_argument("--personas", nargs="*", default=None, help="Only these personas (default: all in seed bank)")
    p.add_argument("--max-per-persona", type=int, default=0, help="Cap lessons per persona from the bank (default: all)")
    p.add_argument("--sleep", type=float, default=1.0, help="Pause between rounds")
    args = p.parse_args()

    config = load_config()
    seeds_dir = os.path.join(ROOT, config.get("curriculum_dir", "data/curriculum"))
    cap_dir = os.path.join(ROOT, config.get("capture_dir", "data/conversations"))
    brain = load_brain(config)
    chat_cfg = config.get("personas", {}).get("chat", {})
    bot_model = args.model or chat_cfg.get("model", "")
    temp = chat_cfg.get("temperature", 0.5)

    if args.prompts:
        lessons = [{"persona": "chat", "instruction": pr, "thinking": "", "response": ""} for pr in args.prompts]
    else:
        files = [os.path.join(seeds_dir, f) for f in sorted(os.listdir(seeds_dir)) if f.endswith(".json")]
        lessons = lessons_from_seeds(files, args.max_per_persona)
        if args.personas:
            lessons = [l for l in lessons if l["persona"] in args.personas]
        if not lessons:
            print("No lessons in seed bank. Pass custom prompts or seed data/curriculum/.*.json first.", file=sys.stderr)
            sys.exit(1)

    print(f"[teach] {len(lessons)} lessons, bot model={bot_model or '<chat default>'}")
    ok = 0
    for i, lesson in enumerate(lessons, 1):
        instruction = lesson["instruction"]
        gold = lesson["response"]
        persona = lesson.get("persona", "chat")
        if not gold:
            print(f"\n# {i}/{len(lessons)} [custom] {instruction}")
            print("  (custom lesson, no gold answer in bank — skipping correction).")
            continue
        print(f"\n# {i}/{len(lessons)} [{persona}] {instruction}")
        try:
            first = ask(brain, bot_model, persona, instruction, temp)
            ftext = (first.content or "").strip()
            print(f"  child: {ftext[:300]}" + ("..." if len(ftext) > 300 else ""))
            time.sleep(args.sleep)
            revised, log = correct(brain, bot_model, persona, instruction, ftext or "(no answer)", gold, temperature=temp)
            rtext = (revised.content or "").strip()
            print(f"  after teacher correction: {rtext[:300]}" + ("..." if len(rtext) > 300 else ""))
            if rtext:
                sid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                write_seed(seeds_dir, persona, {
                    "instruction": instruction,
                    "thinking": lesson["thinking"] or "Teacher-corrected: see gold.",
                    "response": rtext,
                })
                write_conversation(cap_dir, sid, log, lesson, ftext, rtext)
                ok += 1
        except Exception as e:
            print(f"  teaching failed for {instruction!r}: {e}", file=sys.stderr)
        time.sleep(args.sleep)

    print(f"\n[teach] {ok}/{len(lessons)} lessons taught. Run `bp build` to fold them into the dataset.")


if __name__ == "__main__":
    main()