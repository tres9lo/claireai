# Linux setup (Week 1)

Verified on **Zorin OS 18.1 (Ubuntu-based), x86_64, 7.1 GiB RAM, Python 3.12.3**
on 2026-10-29. Commands below are what were actually run, with the results
recorded in `PROGRESS.md`.

## 1. Prerequisites

| Tool | Needed now | Check | Notes |
| --- | --- | --- | --- |
| Python 3.11+ | yes | `python3 --version` | 3.12.3 was used; 3.11 and 3.12 are both supported by the pinned deps |
| Git | yes | `git --version` | 2.43.0 |
| Ollama | yes, for real inference | `ollama --version` | install from <https://ollama.com/download> |
| Node.js 20+ | later (desktop/mobile) | `node --version` | 24.20.0 present, unused so far |
| Rust / Tauri prereqs | later (desktop) | `rustc --version` | not installed yet |

## 2. Python environment

`python3 -m venv` fails on this machine because the `python3.12-venv` system
package is not installed (the venv is created without `pip`). Two ways forward;
this project used the first, which needs no system-wide change:

```bash
# A. uv (already available at ~/.local/bin/uv here)
uv venv --seed --python 3.12 assistant/.venv
uv pip install --python assistant/.venv/bin/python -e "assistant[dev]"

# B. Standard library venv, if python3-venv is installed
python3 -m venv assistant/.venv
source assistant/.venv/bin/activate
python -m pip install -U pip
pip install -e "assistant[dev]"
```

Put `assistant/.venv/bin` on your PATH or call `assistant/.venv/bin/python`
directly; do not rely on the current working directory for imports.

Resolved versions from this install are frozen in `assistant/requirements.lock.txt`
(regenerate with `uv pip freeze --python assistant/.venv/bin/python`).

## 3. Ollama (local model runtime)

Official instructions (checked 2026-10-29, <https://docs.ollama.com/linux>):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

Then download the configured model once:

```bash
ollama pull qwen2.5:1.5b   # candidate tag for ~8 GB RAM machines
ollama list                # confirm the exact tag you configured
curl -s http://127.0.0.1:11434/api/tags | head -c 200
```

Notes:

- The model tag is a **candidate**, not a guarantee. Confirm it exists in the
  current library and that it answers acceptably before relying on it.
- Weights have their own licence (see `docs/model-and-data-licenses.md`).
  Downloading for local use is not permission to redistribute.
- Do not expose port 11434 on the LAN. The assistant refuses non-loopback
  model endpoints unless `CLAIREAI_ALLOW_NON_LOOPBACK_OLLAMA=true` is set.

## 4. Run

```bash
# Development data directory (git-ignored); omit to use ~/.local/share/ClaireAI
export CLAIREAI_DATA_DIR=./runtime-data

assistant/.venv/bin/python -m claireai.cli --once "Explain what a tokenizer does."
assistant/.venv/bin/python -m claireai.cli        # interactive; /log shows the audit tail
```

Exit codes: `0` success, `2` invalid configuration, `3` model runtime unavailable.

## 5. Tests

```bash
cd assistant && .venv/bin/python -m pytest
```

The suite runs without Ollama: the adapter tests use `httpx.MockTransport`
(mocked, not proof that a real runtime answers). The real check is the CLI smoke
test in step 4.
