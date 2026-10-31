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

## 5. Diagnostics

```powershell
$env:CLAIREAI_DATA_DIR = "$PWD\runtime-data"
.\assistant\.venv\Scripts\python.exe scripts\doctor.py
```

Read-only: no database, model or config is created, and no secret values are
printed.

## 6. File tools and the allow-list

The allow-list is a JSON array, so Windows backslashes need doubling. Use
forward slashes instead — Windows accepts them and the value stays readable:

```powershell
$env:CLAIREAI_ALLOWED_FOLDERS = '["C:/Users/tresor/Downloads"]'
.\assistant\.venv\Scripts\python.exe -m claireai.cli --once "List the biggest files in my Downloads folder"
```

Setting it in a `.env` file follows the same rule (`CLAIREAI_ALLOWED_FOLDERS=["C:/Users/tresor/Downloads"]`),
because pydantic-settings parses a JSON array before any Windows path handling
happens. Empty or unset means every file tool is refused.

Typed tool results use forward slashes on every OS (`sub/inner.txt`), so what the
model sees and what lands in the audit log is identical on Windows and Linux.

## 7. Tests

```powershell
cd assistant
.\.venv\Scripts\python.exe -m pytest
```

Record the Python version, Windows build, resolved dependency versions, and the
pytest output in `docs/platform-support.md`.

Three allow-list tests create symlinks, which Windows only permits with
**Developer Mode** enabled or in an elevated shell. Elsewhere they are reported
as skipped with that reason — a skip there is an environment limit, not a
passing security check, so run the suite once on a Windows machine where
symlinks work before claiming Windows coverage.

## 8. Checking the install end to end

```powershell
.\assistant\.venv\Scripts\python.exe scripts\smoke.py
.\assistant\.venv\Scripts\python.exe scripts\smoke.py --skip-model
```

Ten checks covering database, model, tool loop, approval refusal, presence stream
and the append-only audit log. This script uses only `pathlib` and `tempfile`, so
it is the first thing to run on a real Windows machine; its result belongs in
`docs/platform-support.md`.

## 9. Writing files and the approval prompt

```powershell
$env:CLAIREAI_ALLOWED_FOLDERS = '["C:/Users/tresor/Downloads"]'
.\assistant\.venv\Scripts\python.exe -m claireai.cli --once "Create a folder called reports in my Downloads"
```

ClaireAI prints the tool, arguments, target and approval reference, then asks
`approve? [y/N]`. Non-interactive runs refuse every write. A moved file goes to
`%LOCALAPPDATA%\ClaireAI\ClaireAI\trash\<timestamp>\`, never to permanent deletion.

## 10. Presence HUD preview

```powershell
.\assistant\.venv\Scripts\python.exe scripts\presence_preview.py --demo
Start-Process http://127.0.0.1:8765/
```
