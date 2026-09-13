from __future__ import annotations

import datetime
import json
import os
import re
import sys


class Capture:
    """Logs every exchange to data/conversations/<id>.jsonl as sharegpt-style training data."""

    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        os.makedirs(dir_path, exist_ok=True)

    def start(self, persona: str, model: str) -> str:
        sid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        session = {"id": sid, "persona": persona, "model": model, "messages": []}
        return sid

    def append(self, sid: str, role: str, content: str):
        path = os.path.join(self.dir_path, f"{sid}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")

    def get_transcript(self, sid: str) -> list[dict]:
        path = os.path.join(self.dir_path, f"{sid}.jsonl")
        msgs = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        msgs.append(json.loads(line))
        return msgs


class Agent:
    def __init__(self, brain, memory, tools, cfg: dict, capture: Capture | None = None, interactive: bool = False):
        self.brain = brain
        self.memory = memory
        self.tools = tools
        self.cfg = cfg
        self.capture = capture
        self.interactive = interactive
        self.loops_max = int(cfg.get("loops_max", 8))

    def run(self, persona_key: str, prompt: str, model: str = "", sid: str = "") -> str:
        from .personas import system_prompt

        if not sid:
            sid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        persona_cfg = self.cfg.get("personas", {}).get(persona_key, {})
        model = model or persona_cfg.get("model", self.cfg.get("brain", {}).get("default_model", "auto/big"))
        temperature = persona_cfg.get("temperature", 0.5)

        sys_prompt = system_prompt(persona_key)
        memory_ctx = self.memory.context_for(prompt)
        if memory_ctx:
            sys_prompt += "\n\nRELEVANT MEMORY (verify before trusting):\n" + memory_ctx

        messages = [{"role": "system", "content": sys_prompt}]
        if self.capture:
            self.capture.append(sid, "system", sys_prompt)
            self.capture.append(sid, "user", prompt)

        messages.append({"role": "user", "content": prompt})

        last_answer = ""
        tools_list = self.tools.available()
        prev_call = None
        prev_call_count = 0
        silent_rounds = 0
        forced_answer = False
        tool_rounds = 0
        tool_round_limit = int(self.cfg.get("tool_round_limit", 6))

        for _ in range(self.loops_max):
            try:
                result = self.brain.chat(messages, model=model, temperature=temperature, tools=tools_list)
            except Exception as e:
                return f"[error] {e}"

            content = result.content or ""

            if tool_rounds >= tool_round_limit and not forced_answer and result.tool_calls:
                msg = "You have enough context. Do NOT call tools again. Answer the user now using what you already gathered."
                print(f"model reached tool round cap ({tool_rounds}); forcing answer", file=sys.stderr)
                messages.append({"role": "user", "content": msg})
                forced_answer = True
                prev_call = None
                prev_call_count = 0
                if self.capture:
                    self.capture.append(sid, "user", msg)
                continue

            if result.tool_calls:
                tc = result.tool_calls[0]
                name = tc.get("name", "")
                try:
                    args = json.loads(tc.get("arguments", "{}") or "{}")
                except json.JSONDecodeError:
                    args = {}
                call_key = f"{name}{json.dumps(args, sort_keys=True)}"
                if call_key == prev_call:
                    prev_call_count += 1
                else:
                    prev_call = call_key
                    prev_call_count = 1
                if prev_call_count >= 2 and not forced_answer:
                    msg = "You have enough context. Do NOT call tools again. Answer the user now using what you already gathered."
                    print(f"model looped on {name}({args}); forcing answer", file=sys.stderr)
                    messages.append({"role": "user", "content": msg})
                    forced_answer = True
                    prev_call = None
                    prev_call_count = 0
                    if self.capture:
                        self.capture.append(sid, "user", msg)
                    continue
                if prev_call_count >= 2 and forced_answer:
                    msg = f"model looped on {name}({args}); aborting further tool calls"
                    print(msg, file=sys.stderr)
                    if self.capture:
                        self.capture.append(sid, "assistant", content or msg)
                    if content:
                        return content.strip()
                    return "I tried to answer but kept looping on the same action. The task may need clarification or a different model."
                tool_rounds += 1
                if self.capture:
                    self.capture.append(sid, "assistant", f"TOOL CALL: {name}({args})")
                tool_out = self.tools.run(name, args)
                if self.capture:
                    self.capture.append(sid, "tool", f"{name} -> {tool_out[:200]}")
                args_json = tc.get("arguments", "{}") or "{}"
                tc_id = tc.get("id") or f"call-{len(messages)}"
                messages.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [{
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args_json},
                    }],
                })
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": tool_out})
                continue

            fallback_calls = self.brain.extract_calls(content)
            if not fallback_calls:
                fallback_calls = self._echo_fallback(content)
            if fallback_calls:
                call = fallback_calls[0]
                name = call.get("name", "")
                args = call.get("args", call.get("arguments", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call_key = f"{name}{json.dumps(args, sort_keys=True)}"
                if call_key == prev_call:
                    prev_call_count += 1
                else:
                    prev_call = call_key
                    prev_call_count = 1
                if prev_call_count >= 2 and not forced_answer:
                    msg = "You have enough context. Do NOT call tools again. Answer the user now using what you already gathered."
                    print(f"model looped on {name}({args}); forcing answer", file=sys.stderr)
                    messages.append({"role": "user", "content": msg})
                    forced_answer = True
                    prev_call = None
                    prev_call_count = 0
                    if self.capture:
                        self.capture.append(sid, "user", msg)
                    continue
                if prev_call_count >= 2 and forced_answer:
                    msg = f"model looped on {name}({args}); aborting further tool calls"
                    print(msg, file=sys.stderr)
                    if self.capture:
                        self.capture.append(sid, "assistant", content or msg)
                    if content:
                        return content.replace("<bptool>", "").replace("</bptool>", "").strip()
                    return "I tried to answer but kept looping on the same action."
                tool_rounds += 1
                if self.capture:
                    self.capture.append(sid, "assistant", f"TOOL CALL: {name}({args})")
                tool_out = self.tools.run(name, args)
                if isinstance(tool_out, str) and tool_out.startswith("unknown tool"):
                    return content.replace("<bptool>", "").replace("</bptool>", "").strip()
                if self.capture:
                    self.capture.append(sid, "tool", f"{name} -> {tool_out[:200]}")
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Tool {name} result:\n{tool_out}"})
                continue

            last_answer = content
            if self.capture:
                self.capture.append(sid, "assistant", content)
            stripped = last_answer.strip()
            if stripped:
                return stripped
            silent_rounds += 1
            if silent_rounds >= 3:
                return "I couldn't produce a response from the model (empty output). Try again or use a different model."
            print("model returned empty output; nudging", file=sys.stderr)
            messages.append({"role": "user", "content":
                "Your previous response was empty. Answer the user directly now, even if you are unsure."})
            if self.capture:
                self.capture.append(sid, "user",
                    "Your previous response was empty. Answer the user directly now, even if you are unsure.")
            continue

    @staticmethod
    def _echo_fallback(content: str) -> list:
        """Parse 'Tool call name({...})' echoed as plain text by degraded models."""
        import json as _json
        m = re.match(r'^\s*Tool call (\w+)(\(.*\))?\s*$', content, re.DOTALL)
        if not m:
            return []
        name = m.group(1)
        args_raw = m.group(2)
        args = {}
        if args_raw:
            args_raw = args_raw.strip()
            if args_raw.startswith("(") and args_raw.endswith(")"):
                args_raw = args_raw[1:-1]
            try:
                args = _json.loads(args_raw)
            except Exception:
                return []
        return [{"name": name, "args": args, "arguments": args}]