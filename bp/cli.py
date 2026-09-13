#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(ROOT_DIR, "bp.json")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_agent(persona: str, prompt: str, model: str = "", interactive: bool = False, system: str = ""):
    from bp.personas import system_prompt as default_system_prompt, available as available_personas
    from bp.brain import load_brain
    from bp.memory import Memory
    from bp.tools import Tools
    from bp.agent import Agent, Capture

    cfg = load_config()
    mem_dir = os.path.join(ROOT_DIR, cfg.get("memory_dir", "memory"))
    cap_dir = os.path.join(ROOT_DIR, cfg.get("capture_dir", "data/conversations"))

    brain = load_brain(cfg)
    memory = Memory(mem_dir)
    tools = Tools(ROOT_DIR, memory, cfg)
    capture = Capture(cap_dir) if cfg.get("capture", True) else None
    agent = Agent(brain, memory, tools, cfg, capture=capture, interactive=interactive)

    if cfg.get("verify_models", True):
        p = cfg.get("personas", {}).get(persona, {})
        m = model or p.get("model", "")
        if m:
            if not brain.check_model(m):
                fb = p.get("fallback", "")
                if fb and brain.check_model(fb):
                    print(f"[bp] {m} unavailable, falling back to {fb}", file=sys.stderr)
                    model = fb
                else:
                    print(f"[bp] WARNING: model {m} not found in gateway", file=sys.stderr)

    sid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if interactive:
        return repl_agent(agent, persona, cfg, memory, capture, sid)

    result = agent.run(persona, prompt, model=model, sid=sid)
    print(result)
    return result


def repl_agent(agent, persona_key, cfg, memory, capture, sid):
    from bp.personas import available as available_personas, system_prompt as default_system_prompt

    print(f"Big Pickle [{persona_key}] — type 'exit' or Ctrl-C to quit")
    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q", "/quit"):
                print("bye")
                break
            if user_input.startswith("/"):
                cmd_handler(user_input, memory, agent, persona_key, cfg, capture, sid)
                continue
            result = agent.run(persona_key, user_input, sid=sid)
            print(result)
        except KeyboardInterrupt:
            print("\n[bp] interrupted")
            break
        except EOFError:
            break


def cmd_handler(cmd: str, memory, agent, persona_key, cfg, capture, sid):
    parts = cmd.split(None, 1)
    if parts[0] == "/help":
        print("""
Commands:
  /help        Show this help
  /memory      Show memory index
  /memory <q>  Search memory for <q>
  /topic <t>   Show topic file <t>
  /topics      List all topics
  /remember <topic> | <title> | <scope> | <fact>
  /persona     Switch persona: write | know | code | chat
  /model <m>   Override model for this session
  /exit        Exit
""")
    elif parts[0] == "/memory" and len(parts) > 1:
        ctx = memory.context_for(parts[1])
        print(ctx[:3000] or "(nothing found)")
    elif parts[0] == "/memory":
        print(memory.skim()[:3000])
    elif parts[0] == "/topics":
        for t in memory.list_topics():
            print(f"  {t}")
    elif parts[0] == "/topic" and len(parts) > 1:
        print(memory.read_topic(parts[1].strip())[:3000])
    elif parts[0] == "/remember" and len(parts) > 1:
        args = [a.strip() for a in parts[1].split("|")]
        while len(args) < 4:
            args.append("")
        memory.remember(args[0], args[1] or "note", args[2] or "chat", [args[3]] if args[3] else [])
        print(f"recorded to memory topic '{args[0]}'")
    else:
        print(f"unknown command: {parts[0]} — type /help")


def main():
    parser = argparse.ArgumentParser(description="Big Pickle — local AI agent clone")
    subs = parser.add_subparsers(dest="command")

    for p in ("write", "know", "code", "chat"):
        sub = subs.add_parser(p, help=f"Run in {p} mode")
        sub.add_argument("prompt", nargs="*", help="Prompt text (or leave empty for REPL)")
        sub.add_argument("-m", "--model", default="", help="Override model")
        sub.add_argument("-r", "--repl", action="store_true", help="Interactive REPL mode")
        sub.add_argument("-s", "--system", default="", help="Extra system prompt prefix")

    subs.add_parser("repl", help="Start interactive REPL (choose persona at start)")

    args = parser.parse_args()
    if args.command in ("write", "know", "code", "chat"):
        prompt = " ".join(args.prompt) if args.prompt else ""
        if args.repl or not prompt:
            run_agent(args.command, prompt="(waiting for input)", model=args.model, interactive=True)
        else:
            run_agent(args.command, prompt, model=args.model, system=getattr(args, "system", ""))
    elif args.command == "repl":
        print("Choose persona: [w]rite  [k]now  [c]ode  [g]eneral (default)")
        choice = input("> ").strip().lower()
        persona_map = {"w": "write", "k": "know", "c": "code", "g": "chat", "": "chat"}
        persona = persona_map.get(choice, "chat")
        run_agent(persona, prompt="(waiting for input)", interactive=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()