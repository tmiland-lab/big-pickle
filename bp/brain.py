from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class BrainResult:
    content: str
    tool_calls: list = field(default_factory=list)
    model: str = ""
    raw: object = None


class Brain:
    def __init__(self, base_url: str = "", api_key: str = "", timeout: int = 300, max_tokens: int = 2048):
        self.base_url = (base_url or os.environ.get("BP_BASE_URL", "http://localhost:4143/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("BP_API_KEY", "")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"brain HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise RuntimeError(f"brain error: {e}") from e

    def check_model(self, model_id: str) -> bool:
        req = urllib.request.Request(f"{self.base_url}/models")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return any(m.get("id") == model_id for m in data.get("data", []))
        except Exception:
            return True

    def chat(
        self,
        messages: list,
        model: str,
        temperature: float = 0.5,
        tools: list | None = None,
        stream: bool = False,
    ) -> BrainResult:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = self._post(payload)

        if stream:
            return self._parse_stream(data, model)

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        reasoning = msg.get("reasoning") or ""
        if not content and reasoning:
            content = reasoning
        return BrainResult(
            content=content,
            tool_calls=[{
                "id": c.get("id"),
                "name": (c.get("function") or {}).get("name", ""),
                "arguments": (c.get("function") or {}).get("arguments", ""),
            } for c in calls],
            model=data.get("model", model),
            raw=data,
        )

    def _parse_stream(self, data: dict, model: str):
        content_parts = []
        calls = {}
        arg_buf = {}
        usage = None
        if data.get("choices"):
            for choice in data.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    calls.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    if tc.get("id"):
                        calls[idx]["id"] = tc["id"]
                    if fn.get("name"):
                        calls[idx]["name"] += fn["name"]
                    if fn.get("arguments"):
                        calls[idx]["arguments"] += fn["arguments"]
                if choice.get("finish_reason") == "stop":
                    break
            usage = (data.get("usage") or {})
        formatted = [{
            "id": c["id"],
            "name": c["name"],
            "arguments": c["arguments"],
        } for c in calls.values()] if calls else []
        return BrainResult(
            content="".join(content_parts),
            tool_calls=formatted,
            model=model,
            raw=data,
        )

    @staticmethod
    def extract_calls(content: str) -> list:
        pattern = r'<bptool>(.*?)</bptool>'
        matches = re.findall(pattern, content, re.DOTALL)
        calls = []
        for m in matches:
            try:
                calls.append(json.loads(m.strip()))
            except json.JSONDecodeError:
                pass
        return calls


def load_brain(cfg: dict) -> Brain:
    brain_cfg = cfg.get("brain", {})
    return Brain(
        base_url=brain_cfg.get("base_url", ""),
        api_key=brain_cfg.get("api_key", ""),
        timeout=brain_cfg.get("timeout", 300),
        max_tokens=brain_cfg.get("max_tokens", 2048),
    )