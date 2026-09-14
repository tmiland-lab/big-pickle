#!/usr/bin/env python3
"""Extract failed->retried tool corrections from Big Pickle opencode sessions.

For each tool part that ended in `error` status inside a Big Pickle session,
emit one `code`-persona curriculum seed:
  instruction = the situation (tool + what was being done)
  thinking    = the failed inputs + exact error
  response    = what the agent did next that moved forward (retried call or text)

Only the user's own Big Pickle sessions are scanned (same `big-pickle` model
filter as ocdb_export.py) so external-model misbehavior never leaks in.

Usage: python replicate/corrective.py [--db PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.expanduser("~/.local/share/opencode/opencode.db")
DEFAULT_OUT = os.path.join(ROOT, "data", "curriculum", "corrective.json")
PERSONA = "code"

BIG_PICKLE_MODELS = {"big-pickle", "zen/big-pickle"}
NOISE_SUBSTRINGS = ("<system-reminder>", "<tool_result>", "You have enough context")
MIN_RESPONSE_CHARS = 24
MAX_RESPONSE_CHARS = 900
MAX_CONTEXT_CHARS = 260
MAX_THINKING_CHARS = 900
UNHELPFUL_ERRORS = ("user rejected permission",)  # environment-specific, not a lesson


def _parse_state(state) -> str:
    if isinstance(state, dict):
        return state.get("status") or "?"
    if isinstance(state, str):
        m = None
        import re
        m = re.search(r"""status['"]?\s*[:=]\s*['"]([^'"]+)['"]""", state)
        return m.group(1) if m else "?"
    return "?"


def _clean(s: str, limit: int) -> str:
    s = s.strip()
    if len(s) > limit:
        s = s[:limit].rstrip() + " …"
    return s


def _json_str(state, key: str, limit: int) -> str:
    if isinstance(state, dict):
        v = state.get(key, "")
        if isinstance(v, str):
            return _clean(v, limit)
        if v:
            out = json.dumps(v, ensure_ascii=False, default=str)
            return _clean(out, limit)
    return ""


def is_bp(session_model_raw: str) -> bool:
    try:
        mid = json.loads(session_model_raw or "{}").get("id", "")
    except Exception:
        mid = ""
    return mid in BIG_PICKLE_MODELS


def build_examples(db_path: str) -> list[dict]:
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row

    # scan parts in time order; pair each error with the next forward-move
    rows = con.execute(
        """SELECT p.session_id AS sid, p.time_created AS ts, p.data AS pdata,
                  json_extract(m.data,'$.role') AS role, s.model AS smodel
           FROM part p
           JOIN message m ON m.id = p.message_id
           JOIN session s ON s.id = p.session_id
           WHERE parent_id IS NULL
           ORDER BY p.session_id, p.time_created"""
    ).fetchall()
    con.close()

    examples = []
    seen = set()
    for sid in set(r["sid"] for r in rows):
        srows = [r for r in rows if r["sid"] == sid]
        if not is_bp(srows[0]["smodel"]):
            continue
        # replay: last assistant text, tool events in order
        seq = []
        for r in srows:
            try:
                p = json.loads(r["pdata"])
            except Exception:
                continue
            ptype = p.get("type")
            if ptype == "text" and r["role"] == "assistant":
                t = str(p.get("text", "")).strip()
                if len(t) >= MIN_RESPONSE_CHARS and not any(
                    s in t.lower() for s in NOISE_SUBSTRINGS
                ):
                    seq.append({"kind": "text", "ts": r["ts"], "text": t})
            elif ptype == "tool":
                st = p.get("state")
                status = _parse_state(st)
                tool = p.get("tool", "?")
                if status == "error":
                    seq.append({
                        "kind": "error", "ts": r["ts"], "tool": tool,
                        "input": _json_str(st, "input", MAX_THINKING_CHARS),
                        "error": _json_str(st, "error", 240),
                    })
                elif status == "completed" and tool != "invalid":
                    seq.append({
                        "kind": "success", "ts": r["ts"], "tool": tool,
                        "input": _json_str(st, "input", 180),
                    })

        # walk: each error gets the previous text ctx + next non-error event
        for i, ev in enumerate(seq):
            if ev["kind"] != "error":
                continue
            err_sub = ev["error"].lower()
            if any(u in err_sub for u in UNHELPFUL_ERRORS):
                continue
            # context: nearest prior assistant text
            ctx = ""
            for j in range(i - 1, -1, -1):
                if seq[j]["kind"] == "text":
                    ctx = _clean(seq[j]["text"], MAX_CONTEXT_CHARS)
                    break
            # forward: next success or text event
            fwd = None
            for j in range(i + 1, len(seq)):
                if seq[j]["kind"] in ("success", "text"):
                    fwd = seq[j]
                    break
            if not fwd:
                continue
            if fwd["kind"] == "success":
                response = f"[{fwd['tool']}] probes a corrected input: {fwd['input']}"
            else:
                response = _clean(fwd["text"], MAX_RESPONSE_CHARS)
            if len(response) < MIN_RESPONSE_CHARS:
                continue

            thinking = ev["error"]
            if ev["input"]:
                thinking = f"Submitted {ev['input']}\n\nError: {ev['error']}"
            instruction = (f"Working on: {ctx or '(session context)'}\n\n"
                           f"{ev['tool'].upper()} failed — {ev['error']}\n\n"
                           f"Recovered with: ")

            dedup_key = (ev["tool"], ev["error"][:80], response[:80])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            examples.append({
                "instruction": _clean(instruction, 500),
                "thinking": _clean(thinking, MAX_THINKING_CHARS),
                "response": _clean(response, MAX_RESPONSE_CHARS),
            })
    return examples


def main():
    p = argparse.ArgumentParser(description="tool-error corrections -> code-persona seeds")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    if not os.path.exists(args.db):
        print(f"[corrective] ERROR: db not found: {args.db}", file=sys.stderr)
        sys.exit(1)
    examples = build_examples(args.db)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"persona": PERSONA, "examples": examples}, f,
                  ensure_ascii=False, indent=1)
    print(f"[corrective] {len(examples)} correction examples -> {args.out}")


if __name__ == "__main__":
    main()