#!/usr/bin/env python3
"""Export opencode session DB -> sharegpt SFT jsonl.

Rebuilds per-session user/assistant text conversations from the `part` table
(type=text); drops reasoning/tool/step/patch/compaction parts (keeps the final
visible assistant text). This is behavior-clone data: real Big Pickle turns.
Reads the live DB read-only so an active opencode session is never locked.

Usage: python replicate/ocdb_export.py [--db PATH] [--out PATH] [--min-pairs N] [--max-chars N]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import to_sharegpt, deduplicate, filter_min_turns  # noqa: E402

DEFAULT_DB = os.path.expanduser("~/.local/share/opencode/opencode.db")
DEFAULT_OUT = os.path.join(ROOT, "data", "ocdb-sharegpt.jsonl")

NOISE_SUBSTRINGS = (
    "<system-reminder>",
    "<system_reminder>",
    "Your response MUST start with a thinking block",
    "You have enough context. Do NOT call tools again",
    "Your previous response was empty",
    "Tool call ",
    "<tool_result>",
)


def is_noise(text: str) -> bool:
    t = text.lower().strip()
    if len(t) < 2:
        return True
    return any(s.lower() in t for s in NOISE_SUBSTRINGS)


def rebuild_session(con: sqlite3.Connection, sid: str, max_chars: int) -> list[dict] | None:
    """Return per-message list of {'role','content'} for one session, else None."""
    msgs = []
    rows = con.execute(
        """SELECT m.id, json_extract(m.data, '$.role') AS role, m.data AS mdata
           FROM message m WHERE m.session_id = ? ORDER BY m.time_created, m.id""",
        (sid,),
    )
    for mid, role, mdata_raw in rows:
        try:
            mdata = json.loads(mdata_raw)
        except Exception:
            mdata = {}
        role = role or mdata.get("role", "")
        parts = con.execute(
            "SELECT data FROM part WHERE message_id = ? ORDER BY time_created, id", (mid,)
        )
        text_parts = []
        for (pdata_raw,) in parts:
            try:
                p = json.loads(pdata_raw)
            except Exception:
                continue
            if p.get("type") == "text":
                t = p.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
        if not text_parts:
            # some user messages store text in message.data directly
            c = mdata.get("content")
            if isinstance(c, str) and c.strip():
                text_parts.append(c)
        content = "\n".join(t.strip() for t in text_parts if t.strip()).strip()
        if not content:
            continue
        if role in ("user", "assistant"):
            msgs.append({"role": role, "content": content[:max_chars] if role == "assistant" else content})
    return msgs if len(msgs) >= 2 else None


BIG_PICKLE_MODELS = {"big-pickle", "zen/big-pickle"}


def export(db_path: str, out_path: str, min_pairs: int, max_chars: int) -> None:
    if not os.path.exists(db_path):
        print(f"[ocdb_export] ERROR: db not found: {db_path}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row

    # Only top-level sessions (parent_id IS NULL) where the session model is Big Pickle.
    # The session `model` column stores JSON like {"id":"big-pickle","providerID":"opencode",...}.
    # Filter to just the local Big Pickle brain — reject all external models (Claude, Gemini, etc.)
    # which produce different behavior that would corrupt the SFT data.
    try:
        all_sessions = con.execute(
            "SELECT id, model FROM session WHERE parent_id IS NULL ORDER BY time_created").fetchall()
    except sqlite3.OperationalError:
        all_sessions = con.execute("SELECT id, model FROM session ORDER BY time_created").fetchall()

    sessions = []
    skipped_other = 0
    for row in all_sessions:
        model_raw = row["model"] or ""
        try:
            model_id = json.loads(model_raw).get("id", "")
        except Exception:
            model_id = ""
        if model_id in BIG_PICKLE_MODELS:
            sessions.append(row["id"])
        else:
            skipped_other += 1

    print(f"[ocdb_export] {len(sessions)} Big Pickle sessions "
          f"(skipped {skipped_other} non-BP sessions)")
    transcripts = []
    dropped_empty = 0
    for n, sid in enumerate(sessions, 1):
        msgs = rebuild_session(con, sid, max_chars)
        if not msgs:
            dropped_empty += 1
            continue
        transcripts.append(msgs)
        if n % 20 == 0:
            print(f"[ocdb_export]   session {n}/{len(sessions)}...")
    con.close()

    dataset = to_sharegpt(transcripts)
    before = len(dataset)
    dataset = filter_min_turns(dedup_prune(dataset), min_pairs)
    n_conv = sum(len(d.get("conversations", [])) // 2 for d in dataset)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[ocdb_export] done -> {out_path}")
    print(f"[ocdb_export]   sessions parsed: {len(sessions)}, empty/dropped: {dropped_empty}")
    print(f"[ocdb_export]   pairs: {before} -> after dedup+filter(min={min_pairs}): {len(dataset)} (human/assistant turns: {n_conv})")
    return 0


def dedup_prune(dataset: list[dict]) -> list[dict]:
    return deduplicate(dataset)


def main():
    p = argparse.ArgumentParser(description="opencode.db -> sharegpt SFT export")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--min-pairs", type=int, default=1)
    p.add_argument("--max-chars", type=int, default=4000,
                   help="Truncate assistant turns past this many chars")
    args = p.parse_args()
    sys.exit(export(args.db, args.out, args.min_pairs, args.max_chars))


if __name__ == "__main__":
    main()