# Windows setup (Week 1)

> **Status: not yet tested.** No Windows machine was available in this session.
> These are the documented steps with the equivalent commands; they must be run
> on a real Windows 10/11 machine and the result recorded in
> `docs/platform-support.md` before the phase gate claims Windows support.

## 1. Prerequisites

- Python 3.11+ from python.org (check `py -3 --version`).
- Git for Windows.
- Ollama for Windows from <https://ollama.com/download> (runs as a tray app and
  listens on `http://127.0.0.1:11434`).
- Later phases only: Node.js LTS, `rustup`, and the Microsoft C++ build tools
  required by Tauri v2 (<https://v2.tauri.app/start/prerequisites/>).

## 2. Python environment

```powershell
py -3 --version
py -3 -m venv assistant\.venv
.\assistant\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e "assistant[dev]"
```

If PowerShell blocks activation, do **not** weaken the machine-wide execution
policy. Either call the interpreter directly
(`.\assistant\.venv\Scripts\python.exe -m claireai.cli`), or use the documented
`-ExecutionPolicy Bypass -Scope Process` form for the current shell only.

## 3. Ollama and the local model

```powershell
ollama --version
ollama pull qwen2.5:1.5b
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/tags | ConvertTo-Json -Depth 3
```

Do not expose port 11434 on the LAN; the assistant rejects non-loopback model
endpoints unless explicitly overridden.

## 4. Run

```powershell
$env:CLAIREAI_DATA_DIR = "$PWD\runtime-data"
.\assistant\.venv\Scripts\python.exe -m claireai.cli --once "Explain what a tokenizer does."
.\assistant\.venv\Scripts\python.exe -m claireai.cli
```

Without `CLAIREAI_DATA_DIR`, data lives under `%LOCALAPPDATA%\ClaireAI\ClaireAI`
(resolved by `platformdirs`, not hard-coded).

## 5. Tests

```powershell
cd assistant
.\.venv\Scripts\python.exe -m pytest
```

Record the Python version, Windows build, resolved dependency versions, and the
pytest output in `docs/platform-support.md`.
