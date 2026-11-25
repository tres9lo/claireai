# Windows setup

> **Status: CI-tested, not manually verified.** The unit suite runs on
> `windows-latest` for Python 3.11 and 3.12. A human still needs to run the
> installer, Ollama and the Tk window on Windows 10/11 before
> `docs/platform-support.md` can mark it verified.

The runtime is shared with Linux: Python sidecar, SQLite, allow-list, approvals
and Ollama on loopback. Only the commands differ.

## 1. Prerequisites

- Python 3.11+ from python.org (check `py -3 --version`).
- Git for Windows.
- Ollama for Windows from <https://ollama.com/download>.
- Optional: Node.js LTS and Rust only when working on the future Tauri shell.

PyTorch is not required for the default runtime.

## 2. Install Ollama and the models

The Windows installer starts Ollama as a background service. In a new terminal:

```powershell
ollama --version
ollama pull qwen3:1.7b
ollama pull qwen2.5-coder:1.5b
Invoke-RestMethod http://127.0.0.1:11434/api/tags | Select-Object -Expand models
```

ClaireAI uses `http://127.0.0.1:11434/api`. No key is required.

## 3. Python environment

```powershell
py -3 -m venv assistant\.venv
.\assistant\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e "assistant[dev,desktop]"
```

If PowerShell blocks activation, do **not** weaken the machine-wide execution
policy. Call the interpreter directly
(`.\assistant\.venv\Scripts\python.exe -m claireai.cli`) or use the documented
`-ExecutionPolicy Bypass -Scope Process` form for the current shell only.

## 4. Run

```powershell
$env:CLAIREAI_DATA_DIR = "$PWD\runtime-data"
.\assistant\.venv\Scripts\python.exe scripts\doctor.py
.\assistant\.venv\Scripts\python.exe -m claireai.cli --once "Hello"
.\assistant\.venv\Scripts\python.exe -m claireai.cli
```

Without `CLAIREAI_DATA_DIR`, data lives under
`%LOCALAPPDATA%\ClaireAI\ClaireAI`.

## 5. File tools and the allow-list

Use forward slashes in the JSON array; Windows accepts them and the value stays
readable:

```powershell
$env:CLAIREAI_ALLOWED_FOLDERS = '["C:/Users/tresor/Downloads"]'
.\assistant\.venv\Scripts\python.exe -m claireai.cli --once `
  "List the biggest files in my Downloads folder"
```

An empty or unset allow-list refuses every file tool. Writes require `y`/`yes`
at an interactive prompt. Non-interactive runs refuse them.

## 6. Diagnostics and tests

```powershell
.\assistant\.venv\Scripts\python.exe scripts\doctor.py
cd assistant
.\.venv\Scripts\python.exe -m pytest
```

Symlink tests need Developer Mode or elevation; a skip in a normal shell is an
environment limit, not a passing security check.

## 7. Speech

Optional speech output requires eSpeak NG on `PATH`; Piper can be installed with
`pip install piper-tts` and a voice under
`runtime-data\models\voice\piper\`. Windows playback uses `winsound`; live audio
has not been manually verified.

## 8. Installer

`.\install.ps1` prints the detected OS and Python, lists downloads, and asks
before fetching. `-dry-run` prints the plan only. It installs the Python
application; Ollama remains a separate system install.
