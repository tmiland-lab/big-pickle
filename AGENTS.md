# AGENTS.md — big-pickle

Guidance for agents working in this repo. Keep it public-safe (no personal
paths, tokens, or conversation data in commits).

## What this is
A self-replicating local AI agent: a tool-first behavior protocol distilled
onto any OpenAI-compatible brain, with session capture → SFT dataset → local
LoRA fine-tuning. Author: big-pickle. Director: the boss.

## Commands
- `./bin/bp <code|write|know|chat> "<prompt>"` — run a persona (auto-sources `.env.local`)
- `./bin/bp chat` — interactive REPL (also `-r`)
- `./bin/bp build` — curriculum + conversations → `data/sft-dataset.jsonl`
- `./bin/bp dry` / `./bin/bp train` — trainer dry-run / run (ROCm venv)
- `./bin/bp env` — how env resolution works / show effective vars

## Conventions
- Personas + models live in `bp.json` (per-persona model + `fallback`).
- Tool schemas MUST be `{"type":"function","function":{...}}` — plain
  `{"name","parameters"}` → Nvidia 400s "missing field type".
- Some brains (Nemotron) return everything in `reasoning` with empty
  `content` — brain.py falls back to reasoning. Agent force-answers on
  loop/stall via `tool_round_limit` in bp.json.
- Store tool-call history natively (`assistant.tool_calls[]` + `role:"tool"`),
  never flatten to text — flattening makes deepseek echo `Tool call ...` as prose.
- Memory bank: index + `topics/*.md`; entry format `### DATE — Title` + 1–3
  terse lines. Verify-before-claim is a core persona rule.

## Publish hygiene (public repo)
- `data/`, `memory/`, `.env.local`, `*.pyc` are gitignored — never `git add -f`.
- `examples/curriculum/*.json` are the sanitized public copy of seeds;
  personal seeds live in `data/curriculum/` (gitignored).
- Verify before commit: `git status --porcelain --ignored`.

## Hardware note (train.py)
- gfx1030 (RX 6900 XT): pins torch 2.5.1+rocm6.2; `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
- `--desktop-safe` default; pre-flight VRAM/RAM guard; 5GB headroom reserve.