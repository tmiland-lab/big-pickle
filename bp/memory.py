from __future__ import annotations

import datetime
import os
import re


class Memory:
    """Persistent memory bank mirroring the protocol: index + topic files.

    Entries:  ### YYYY-MM-DD — Title / - Scope: <x> / - terse facts.
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.index_path = os.path.join(root, "index.md")
        self.user_path = os.path.join(root, "user.md")
        self.learnings_path = os.path.join(root, "learnings.md")
        self.topics_dir = os.path.join(root, "topics")
        os.makedirs(self.topics_dir, exist_ok=True)
        if not os.path.exists(self.index_path):
            self._write(self.index_path, "# Knowledge Index\n\n- user.md — user preferences\n- learnings.md — lessons\n- topics/ — per-topic notes\n")

    def _write(self, path: str, content: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _read(self, path: str) -> str:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def skim(self) -> str:
        return self._read(self.index_path)

    def get_user(self) -> str:
        return self._read(self.user_path)

    def get_learnings(self) -> str:
        return self._read(self.learnings_path)

    def list_topics(self) -> list[str]:
        return sorted(f[:-3] for f in os.listdir(self.topics_dir) if f.endswith(".md"))

    def read_topic(self, topic: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", topic)
        return self._read(os.path.join(self.topics_dir, f"{safe}.md"))

    def write_topic(self, topic: str, content: str):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", topic)
        path = os.path.join(self.topics_dir, f"{safe}.md")
        if not content.startswith("# "):
            content = f"# {topic}\n\n" + content
        self._write(path, content)
        self._refresh_index()

    def append_topic(self, topic: str, entry: str):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", topic)
        path = os.path.join(self.topics_dir, f"{safe}.md")
        existing = self._read(path)
        new = existing.rstrip() + "\n\n" + entry + "\n" if existing else f"# {topic}\n\n" + entry + "\n"
        self._write(path, new)
        self._refresh_index()

    def remember(self, topic: str, title: str, scope: str, facts: list[str]):
        today = datetime.date.today().isoformat()
        entry = f"### {today} — {title}\n- Scope: {scope}\n"
        for fact in facts:
            entry += f"- {fact}\n"
        self.append_topic(topic, entry)

    def _refresh_index(self):
        topics = self.list_topics()
        lines = ["# Knowledge Index\n", "- user.md — user preferences\n- learnings.md — lessons\n", "- topics/ — per-topic notes"]
        for t in topics:
            lines.append(f"  - topics/{t}.md")
        self._write(self.index_path, "\n".join(lines) + "\n")

    def context_for(self, query: str) -> str:
        """Return index + topics that mention the query."""
        query_l = query.lower()
        parts = ["INDEX", self.skim()]
        for word in re.findall(r"[A-Za-z0-9_]{4,}", query_l):
            for topic in self.list_topics():
                if word in topic.lower():
                    body = self.read_topic(topic)
                    if any(word in line.lower() for line in body.splitlines()[:80]):
                        parts.append(f"TOPIC: {topic}\n{body[:4000]}")
                        break
        return "\n\n".join(parts) if len(parts) > 1 else ""