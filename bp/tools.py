from __future__ import annotations

import glob as globmod
import os
import re
import subprocess
import urllib.request


class Tools:
    def __init__(self, root_dir: str, memory, cfg: dict, agent=None):
        self.root_dir = root_dir
        self.memory = memory
        self.agent = agent
        self.cfg = cfg.get("tools", {})
        self.timeout = cfg.get("shell", {}).get("timeout", 60)

    _SPECS = {
        "shell": ("Run a shell command in the local workspace. Returns stdout+stderr. Use for git, compilers, installers, system queries.", {"command": {"type": "string"}}, ["command"]),
        "shell_pipe": ("Run a shell command with input piped via stdin. Not commonly needed.", {"command": {"type": "string"}, "stdin": {"type": "string"}}, ["command"]),
        "read_file": ("Read a full text file at an absolute or relative path.", {"path": {"type": "string"}}, ["path"]),
        "write_file": ("Write content to a file at an absolute or relative path (overwrites).", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        "list_dir": ("List entries in a directory. Accepts a glob pattern.", {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
        "grep": ("Search file contents for a regex pattern in a directory/glob.", {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
        "web_fetch": ("Fetch a URL and return its text content. For research/docs/APIs.", {"url": {"type": "string"}}, ["url"]),
        "memory_remember": ("Record a durable fact/lesson to persistent memory.", {"topic": {"type": "string"}, "title": {"type": "string"}, "scope": {"type": "string"}, "facts": {"type": "array", "items": {"type": "string"}}}, ["topic", "title", "scope", "facts"]),
        "memory_recall": ("Search persistent memory for a topic or keyword.", {"query": {"type": "string"}}, ["query"]),
    }

    def available(self) -> list[dict]:
        tools = []
        enabled_groups = [
            ("shell", ["shell", "shell_pipe"]),
            ("files", ["read_file", "write_file", "list_dir", "grep"]),
            ("web", ["web_fetch"]),
        ]
        allowed = ["memory_remember", "memory_recall"]
        for group, names in enabled_groups:
            if self.cfg.get(group, {}).get("enabled", True):
                allowed.extend(names)
        for name in allowed:
            desc, props, required = self._SPECS[name]
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            })
        return tools

    def run(self, name: str, args: dict) -> str:
        handlers = {
            "shell": self.shell,
            "shell_pipe": self.shell_pipe,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_dir": self.list_dir,
            "grep": self.grep,
            "web_fetch": self.web_fetch,
            "memory_remember": self.memory_remember,
            "memory_recall": self.memory_recall,
        }
        fn = handlers.get(name)
        if not fn:
            return f"ERROR: unknown tool {name}"
        if self.agent and getattr(self.agent, "interactive", False):
            confirm = input(f"\nRunning tool {name}: {args}\nAllow? [Y/n]: ")
            if confirm.strip().lower() not in ("", "y", "yes"):
                return "aborted by user"
        return fn(args)

    def shell(self, args: dict) -> str:
        cmd = str(args.get("command", "")).strip()
        if not cmd:
            return "ERROR: empty command"
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=self.timeout, cwd=self.root_dir, executable="/bin/bash",
            )
            out = (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.stderr else "")
            return out[-8000:] or f"(exit {r.returncode}, no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {self.timeout}s"
        except Exception as e:
            return f"ERROR: {e}"

    def shell_pipe(self, args: dict) -> str:
        cmd = str(args.get("command", "")).strip()
        stdin = str(args.get("stdin", ""))
        if not cmd:
            return "ERROR: empty command"
        try:
            r = subprocess.run(
                cmd, shell=True, input=stdin, capture_output=True, text=True,
                timeout=self.timeout, cwd=self.root_dir, executable="/bin/bash",
            )
            return (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.stderr else "")
        except subprocess.TimeoutExpired:
            return "ERROR: timed out"

    def read_file(self, args: dict) -> str:
        path = self._resolv(args.get("path", ""))
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()[-8000:]
        except Exception as e:
            return f"ERROR: {e}"

    def write_file(self, args: dict) -> str:
        path = self._resolv(args.get("path", ""))
        content = args.get("content", "")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"wrote {os.path.getsize(path)} bytes to {path}"
        except Exception as e:
            return f"ERROR: {e}"

    def list_dir(self, args: dict) -> str:
        pattern = args.get("pattern", args.get("path", "*"))
        if not pattern.startswith(("/", self.root_dir)):
            pattern = os.path.join(self.root_dir, pattern)
        try:
            hits = globmod.glob(pattern, recursive=True)
            hits = [h for h in hits if not h.startswith(".git")]
            return "\n".join(hits[:200]) or "(no matches)"
        except Exception as e:
            return f"ERROR: {e}"

    def grep(self, args: dict) -> str:
        pattern = args.get("pattern", "")
        path = self._resolv(args.get("path", ""))
        if not pattern:
            return "ERROR: empty pattern"
        hits = []
        try:
            for root, _, files in os.walk(path):
                if ".git" in root:
                    continue
                for fn in files:
                    fp = os.path.join(root, fn)
                    try:
                        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f, 1):
                                if re.search(pattern, line):
                                    hits.append(f"{fp}:{i}: {line.rstrip()[:200]}")
                                    if len(hits) >= 100:
                                        return "\n".join(hits)
                    except Exception:
                        continue
            return "\n".join(hits[:100]) or "(no matches)"
        except Exception as e:
            return f"ERROR: {e}"

    def web_fetch(self, args: dict) -> str:
        url = str(args.get("url", ""))
        timeout = int(args.get("timeout", self.cfg.get("web", {}).get("timeout", 25)))
        if not url.startswith(("http://", "https://")):
            return "ERROR: URL must start with http(s)://"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "big-pickle-agent/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8", errors="replace")
            text = re.sub(r"<script.*?</script>", " ", raw, flags=re.DOTALL | re.I)
            text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:6000]
        except Exception as e:
            return f"ERROR: {e}"

    def memory_remember(self, args: dict) -> str:
        topic = str(args.get("topic", "misc"))
        title = str(args.get("title", ""))
        scope = str(args.get("scope", "big-pickle"))
        facts = args.get("facts", [])
        if isinstance(facts, str):
            facts = [facts]
        self.memory.remember(topic, title, scope, facts)
        return f"recorded to memory topic '{topic}'"

    def memory_recall(self, args: dict) -> str:
        query = str(args.get("query", ""))
        ctx = self.memory.context_for(query)
        return ctx or "(nothing found in memory)"

    def _resolv(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return os.path.join(self.root_dir, path)