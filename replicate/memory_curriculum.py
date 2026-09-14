#!/usr/bin/env python3
"""Convert memory-bank files (~/.opencode-database) into `know`-persona SFT seeds.

Each `### <date> — Title` section becomes one example:
  instruction = a question whose answer is that lesson
  thinking    = the header context (title + scope line when useful)
  response    = the terse body lines

The memory bank is distilled knowledge the child ALREADY consults at runtime;
making it curriculum lets the model internalize it instead of re-reading it.

Usage: python replicate/memory_curriculum.py [--bank PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BANK = os.path.expanduser("~/.opencode-database")
DEFAULT_OUT = os.path.join(ROOT, "data", "curriculum", "memory-know.json")

HEADER_RE = re.compile(r"^###\s+(.+)$")
INDEXED_FILES = ("user.md", "learnings.md")
PROJECT_GLOB = ("projects", "*.md")
MAX_RESPONSE_CHARS = 2000


def is_indexed(fn: str) -> bool:
    return fn in INDEXED_FILES or fn == "_TEMPLATE.md"


def parse_sections(text: str) -> list[tuple[str, str]]:
    """Return [(title, body)] for every `### ...` section."""
    sections = []
    cur_title = None
    cur_body: list[str] = []
    for raw in text.splitlines():
        m = HEADER_RE.match(raw)
        if m:
            if cur_title is not None:
                sections.append((cur_title, "\n".join(cur_body).strip()))
            cur_title = m.group(1).strip()
            cur_body = []
        else:
            if cur_title is not None:
                cur_body.append(raw)
    if cur_title is not None:
        sections.append((cur_title, "\n".join(cur_body).strip()))
    return sections


def build_examples(bank: str) -> list[dict]:
    examples = []

    def _emit(instruction: str, thinking: str, response: str):
        response = response.strip()
        if not response:
            return
        if len(response) > MAX_RESPONSE_CHARS:
            response = response[:MAX_RESPONSE_CHARS] + " …"
        examples.append({
            "instruction": instruction,
            "thinking": thinking.strip(),
            "response": response,
        })

    # user.md + learnings.md — top-level lessons
    for fn in INDEXED_FILES:
        path = os.path.join(bank, fn)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for title, body in parse_sections(text):
            scope = ""
            for line in body.splitlines():
                if line.lower().startswith("- scope:"):
                    scope = line.strip()
                    break
            instruction = title.split(" — ")[-1]
            # turn the title into an implicit question
            if not instruction.endswith("?"):
                instruction += " — what should I know here?"
            _emit(
                instruction,
                f"Memory-bank lesson ({fn}); {scope}".replace(" ;", ";") + f" Context: {title}",
                body,
            )

    # projects/*.md — per-project lessons
    projects_dir = os.path.join(bank, PROJECT_GLOB[0])
    if os.path.isdir(projects_dir):
        for fn in sorted(os.listdir(projects_dir)):
            if not fn.endswith(".md") or fn == "_TEMPLATE.md":
                continue
            full = os.path.join(projects_dir, fn)
            with open(full, "r", encoding="utf-8") as f:
                text = f.read()
            for title, body in parse_sections(text):
                instruction = f"(Project {fn[:-3]}) {title} — what is the takeaway?"
                _emit(instruction, f"Per-project memory note in {fn}; {title}", body)

    # also capture plain sections that have no ### header (project intro lines)
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default=DEFAULT_BANK)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--persona", default="know")
    args = p.parse_args()

    examples = build_examples(args.bank)
    data = {"persona": args.persona, "examples": examples}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"[memory_curriculum] {len(examples)} examples -> {args.out}")


if __name__ == "__main__":
    main()