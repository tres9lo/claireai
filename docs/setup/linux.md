# Linux setup

This is the supported path for the current runtime: **Ollama + Qwen**, not a
checkpoint trained in this repository. The commands and setup report were
exercised on Zorin OS 18.1 (Ubuntu-based), x86_64, 7.1 GiB RAM, Python 3.12.3,
on 2026-11-25. Live Qwen inference still needs the two model pulls below and
must be recorded separately; the session that made the switch did not have the
weights installed.

## 1. Prerequisites

| Tool | Check | Notes |
| --- | --- | --- |
| Python 3.11+ | `python3 --version` | 3.11 and 3.12 are supported |
| Git | `git --version` | |
| Ollama | `ollama --version` | install from <https://ollama.com/download> |
| Tk | `python3 -c "import tkinter"` | the current desktop window uses it |
| Piper / eSpeak NG | `command -v piper espeak-ng` | optional local speech output |

PyTorch is **not** required for the default runtime. It is only needed if you
deliberately run the archived `brain-lab` research code.

## 2. Install Ollama and the models

```bash
curl -fsSL https://ollama.com/install.sh | sh   # or use the desktop installer
ollama --version
ollama pull qwen3:1.7b
ollama pull qwen2.5-coder:1.5b
curl -s http://127.0.0.1:11434/api/tags | head
```

The daemon listens on loopback only. ClaireAI uses it at
`http://127.0.0.1:11434/api`; no API key is involved.

## 3. Python environment

```bash
cd claireai
uv venv --seed --python 3.12 assistant/.venv
uv pip install --python assistant/.venv/bin/python -e "assistant[dev,desktop]"
```

The standard-library alternative is `python3 -m venv assistant/.venv`, then
`assistant/.venv/bin/python -m pip install -e "assistant[dev,desktop]"`.

`desktop` installs the current Tk window and optional speech input. It no longer
pulls PyTorch.

## 4. Run

```bash
export CLAIREAI_DATA_DIR=./runtime-data
assistant/.venv/bin/python scripts/doctor.py
assistant/.venv/bin/python -m claireai.cli --once "Hello"
assistant/.venv/bin/python -m claireai.cli
./scripts/start.sh
```

Without `CLAIREAI_DATA_DIR`, data lives under `~/.local/share/ClaireAI`.

Exit codes: `0` success, `2` invalid configuration, `3` the model runtime or a
tool turn failed.

## 5. File tools

The allow-list is deny-by-default. The file tools can see nothing until you name
a folder:

```bash
CLAIREAI_ALLOWED_FOLDERS='["~/Downloads"]' \
  assistant/.venv/bin/python -m claireai.cli --once \
  "List the biggest files in my Downloads folder"
```

Writes print the exact tool, arguments and target, then ask `approve? [y/N]`.
Piped input, cron jobs and non-interactive shells cannot approve a write.

## 6. Diagnostics and tests

```bash
assistant/.venv/bin/python scripts/doctor.py
assistant/.venv/bin/python scripts/tool_call_eval.py --runs 3
cd assistant && .venv/bin/python -m pytest
```

`scripts/doctor.py` is read-only and never prints secret values. The unit suite
does not need a model. Tool and conversation evaluations need `ollama serve` and
the pulled Qwen models.

## 7. Speech

```bash
assistant/.venv/bin/python -m pip install piper-tts
sudo apt install espeak-ng pulseaudio-utils
```

Piper voices go under `runtime-data/models/voice/piper/` (or
`CLAIREAI_PIPER_DIR`); each voice needs its `.onnx` and `.onnx.json` side by
side. eSpeak NG is the fallback. Nothing is uploaded.

## 8. Windows

The runtime is shared. Use [windows.md](windows.md) for PowerShell commands.
