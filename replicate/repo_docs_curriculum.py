#!/usr/bin/env python3
"""Generate `code`-persona seeds from the repo's own docs (AGENTS.md, README, bp.json).

The repo is self-describing: commands, conventions, architecture, hardware
notes. Distilling these into curriculum lets the fine-tuned model know the
exact facts of its own deployment instead of re-reading the docs at runtime.

Emits `{instruction, thinking, response}` triples: a natural question, the
`thinking` = the exact directive from the source doc, `response` = the precise
answer. Persona: code.

Usage: python replicate/repo_docs_curriculum.py [--root PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "data", "curriculum", "repo-docs.json")
PERSONA = "code"

HEADER_RE = re.compile(r"^#{1,4}\s+(.+)$")
BULLET_RE = re.compile(r"^\s*[-*]\s+(.+)$")
CODE_RE = re.compile(r"^\s*`([^`]+)`\s+")


def sections_from_md(text: str, max_lines: int = 8) -> list[dict]:
    """Split markdown into (section_header, bullet list) chunks."""
    out = []
    cur_title = None
    cur_lines: list[str] = []
    for raw in text.splitlines():
        m = HEADER_RE.match(raw)
        if m:
            if cur_title is not None and cur_lines:
                out.append({"title": cur_title, "lines": cur_lines})
            cur_title = m.group(1).strip()
            cur_lines = []
        else:
            if cur_title is not None:
                cur_lines.append(raw)
    if cur_title is not None and cur_lines:
        out.append({"title": cur_title, "lines": cur_lines})

    chunks = []
    for sec in out:
        title = sec["title"]
        # gather terse factual bullets under the header (skip prose paragraphs)
        bullets = []
        in_code = False
        for ln in sec["lines"]:
            stripped = ln.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                # only code-fence commands become facts when short
                if len(stripped) <= 90:
                    bullets.append(stripped)
                continue
            if BULLET_RE.match(ln) and 8 <= len(stripped) <= 260:
                bullets.append(stripped)
        if bullets:
            chunks.append({"title": title, "bullets": bullets[:max_lines]})
    return chunks


def doc_to_words(text: str, title: str) -> dict:
    from collections import Counter
    words = Counter(re.findall(r"[a-z0-9]+", text.lower()))
    return {"title": title, "top": [w for w, _ in words.most_common(12)]}


def build_examples(root: str) -> list[dict]:
    examples = []
    anchors = []

    agents_md = os.path.join(root, "AGENTS.md")
    readme_md = os.path.join(root, "README.md")
    bp_json = os.path.join(root, "bp.json")
    bin_bp = os.path.join(root, "bin", "bp")

    facts = []  # (question, answer)

    # --- AGENTS.md: conventions + commands -------------------------------
    if os.path.isfile(agents_md):
        with open(agents_md, "r", encoding="utf-8") as f:
            text = f.read()
        sections = sections_from_md(text)
        for sec in sections:
            title = sec["title"]
            for b in sec["bullets"]:
                b_clean = re.sub(r"^\s*[-*]\s+", "", b).strip()
                if re.match(r"^`?\.?/bin/bp\b", b_clean):
                    facts.append(
                        (f"What does `{b_clean.split(' — ')[0].strip('`')}` do?",
                         b_clean))
                elif title.lower() in ("conventions", "publish hygiene", "hardware note"):
                    question = (f"Repo convention ({title.lower()}): {b_clean.split('—')[0].strip()} "
                                "— what exactly?")
                    facts.append((question, b_clean))

    # --- README: quick start + training notes -----------------------------
    if os.path.isfile(readme_md):
        with open(readme_md, "r", encoding="utf-8") as f:
            text = f.read()
        sections = sections_from_md(text)
        saw_code_block = False
        for sec in sections:
            title = sec["title"]
            for b in sec["bullets"]:
                if b.startswith("./bin/bp"):
                    facts.append((f"Quick-start: what does `{b}` run?", b))
                if title.lower() in ("training notes (rx 6900 xt / gfx1030)",
                                     "the self-replication loop",
                                     "why this exists"):
                    q = (f"Snippet: {b.split('—')[0].strip()} — explain.")
                    facts.append((q, b))
        # also pull the ASCII quick-start usage lines
        if os.path.isfile(readme_md):
            for line in text.splitlines():
                s = line.strip()
                if s.startswith("./bin/bp") and "  " in s:
                    facts.append((f"Usage of `{s.split()[2]}` persona?", s))

    # --- bp.json: per-key facts -------------------------------------------
    if os.path.isfile(bp_json):
        with open(bp_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        personas = cfg.get("personas", {})
        for pname, pconf in personas.items():
            model = pconf.get("model"); fb = pconf.get("fallback")
            if model:
                facts.append(
                    (f"Which model routes the `{pname}` persona (and its fallback)?",
                     f"`{pname}` -> {model} (fallback {fb})."))
        if "tools" in cfg and isinstance(cfg["tools"], dict):
            on = [k for k, v in cfg["tools"].items() if isinstance(v, dict) and v.get("enabled")]
            facts.append(("Which tools does big-pickle enable?", ", ".join(on) + " (see bp.json tools)."))
        if "train" in cfg:
            tr = cfg["train"]
            facts.append(("What are the trainer defaults (model, quant, headroom, safe mode)?",
                          json.dumps(tr, ensure_ascii=False)))
        if "loops_max" in cfg:
            facts.append(("What is the agent loop budget and tool-round cap?",
                          f"loops_max={cfg['loops_max']}, tool_round_limit={cfg['tool_round_limit']}."))

    # --- bin/bp: the command surface ----------------------------------------
    if os.path.isfile(bin_bp):
        with open(bin_bp, "r", encoding="utf-8") as f:
            src = f.read()
        # TODO: pull the `echo ...` usage lines in `help` for rich command facts.
        case_block = re.search(r"(?s)case \"\$cmd\" in(.*?)esac", src)
        if case_block:
            usage_by_branch = {}
            branch_re = re.compile(r"^\s*([a-z][a-z| ]*?)\)\s*$", re.M)
            branch_re = re.compile(r"^\s*([\w|]+)\s*\)\s*$", re.MULTILINE)
            branches = list(branch_re.finditer(case_block.group(1)))
            for i, m in enumerate(branches):
                cmds = [c for c in m.group(1).split("|") if c]
                end = branches[i + 1].start() if i + 1 < len(branches) else len(case_block.group(1))
                body = case_block.group(1)[m.end():end]
                route = body.strip().splitlines()[1] if body.strip() else ""
                action = ""
                if "ocdb_export" in route:
                    action = "Extracts Big Pickle behavior-clone transcripts from the live opencode session DB."
                elif "memory_curriculum" in route:
                    action = "Turns the persistent memory bank (~/.opencode-database lessons) into know-persona curriculum seeds."
                elif "curriculum" in route:
                    action = "Merges all curriculum seed files into teacher SFT examples."
                elif "dataset" in route:
                    action = "Merges ocdb + curriculum + conversation captures into the final SFT dataset."
                elif "distill" in route:
                    action = "Bulk-generates curriculum seeds from a topic via the brain."
                elif "teach" in route:
                    action = "Takes a tenant prompt, runs the child, accepts a correction, and stores the gold pair."
                elif "train" in route:
                    action = "Runs QLoRA fine-tuning (with delta-gate stamping)."
                elif "dry" in route:
                    action = "Trainer dry-run: prints config + pre-flight headroom check."
                elif "cli" in route:
                    action = "Runs a persona (or the REPL) through bp.cli."
                for c in cmds:
                    if action:
                        facts.append((f"`bin/bp {c}` — what does this subcommand do?", action))
        # the `help` echo block is the canonical cmd->desc mapping
        help_block = re.search(r'echo "  bp (.*?)\n    exit 0', src, re.S)
        if help_block:
            for line in help_block.group(1).splitlines():
                m = re.match(r'\s*echo "  bp (\S+)\s+(\S+)', line)
                if m:
                    cmd, desc = m.group(1), line.split('"', 2)[1].strip()
                    desc = desc.split('"')[0].strip()
                    facts.append((f"`{cmd}` — what does this subcommand do?", desc))

    seen = set()
    for q, a in facts:
        q = re.sub(r"\s+", " ", q).strip()
        a = re.sub(r"\s+", " ", a).strip()
        key = (q[:60], a[:60])
        if not q or not a or key in seen:
            continue
        seen.add(key)
        examples.append({
            "instruction": q,
            "thinking": "From repo docs (AGENTS.md / README / bp.json / bin/bp).",
            "response": a,
        })

    anchors.append(doc_to_words(text if locals().get("text") else "", "repo-hash"))
    return examples


def main():
    p = argparse.ArgumentParser(description="repo docs -> code-persona seeds")
    p.add_argument("--root", default=ROOT)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    examples = build_examples(args.root)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"persona": PERSONA, "examples": examples}, f,
                  ensure_ascii=False, indent=1)
    print(f"[repo_docs_curriculum] {len(examples)} repo-doc examples -> {args.out}")


if __name__ == "__main__":
    main()