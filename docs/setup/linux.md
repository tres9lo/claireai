# Linux setup (Week 1)

Verified on **Zorin OS 18.1 (Ubuntu-based), x86_64, 7.1 GiB RAM, Python 3.12.3**
on 2026-10-29. Commands below are what were actually run, with the results
recorded in `PROGRESS.md`.

## 1. Prerequisites

| Tool | Needed now | Check | Notes |
| --- | --- | --- | --- |
| Python 3.11+ | yes | `python3 --version` | 3.12.3 was used; 3.11 and 3.12 are both supported by the pinned deps |
| Git | yes | `git --version` | 2.43.0 |
| PyTorch | yes, for training and inference | `python -c 'import torch; print(torch.__version__)'` | CPU build is sufficient; training is slow on low-end CPUs |
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

## 3. Install the scratch model code

```bash
uv pip install --python assistant/.venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python assistant/.venv/bin/python -e "assistant[model]"
uv pip install --python assistant/.venv/bin/python -e ./brain-lab
```

Use the CPU wheel for this host; PyTorch's official installer provides a CPU
compute-platform option for pip. See [PyTorch Start Locally](https://pytorch.org/get-started/locally/).

The repository includes `brain-lab/data/bootstrap-v1/`, a small synthetic MIT-
licensed starter corpus with generated SVG workspace screens. It checks the
training pipeline but is far too small and template-based to create a useful
assistant. Add only licensed plain-text material under `brain-lab/data/raw/`;
record the source and terms before using it. The split-aware trainer learns its
tokenizer from `train/`, validates on `validation/`, and ignores `test/`.

```bash
export CLAIREAI_DATA_DIR=./runtime-data
assistant/.venv/bin/python brain-lab/train.py brain-lab/data/bootstrap-v1/text --steps 1000
```

The resulting checkpoint is stored under `runtime-data/models/` and ignored by
Git. Keep steps and context small on the documented 4-thread, 7 GiB machine.

## 4. Run

```bash
# Development data directory (git-ignored); omit to use ~/.local/share/ClaireAI
export CLAIREAI_DATA_DIR=./runtime-data

assistant/.venv/bin/python -m claireai.cli --once "Explain what a tokenizer does."
assistant/.venv/bin/python -m claireai.cli        # interactive; /log shows the audit tail
```

Exit codes: `0` success, `2` invalid configuration, `3` scratch checkpoint or PyTorch unavailable.

## 5. Diagnostics

`scripts/doctor.py` is read-only: it reports versions, paths, database/schema
state and local checkpoint status, never creates anything and never prints secret
values (only the names of set `CLAIREAI_*` variables).

```bash
export CLAIREAI_DATA_DIR=./runtime-data
assistant/.venv/bin/python scripts/doctor.py        # exit 0 ready, 1 not ready, 2 bad config
assistant/.venv/bin/python scripts/doctor.py --json
```

## 6. Tests

```bash
cd assistant && .venv/bin/python -m pytest
```

The unit suite does not train a model. A real inference check requires a trained
checkpoint and must be reported separately from test results.

## Checking the install end to end

```bash
assistant/.venv/bin/python scripts/smoke.py              # needs a scratch checkpoint
assistant/.venv/bin/python scripts/smoke.py --skip-model # runtime only
```

Ten checks: settings, migrations, model reachability, a real chat turn, the tool
loop, a refused write, the presence stream and the append-only audit log. Exit
code 0 means everything passed.

## Writing files (approval prompt)

```bash
CLAIREAI_ALLOWED_FOLDERS='["~/Downloads"]' \
  assistant/.venv/bin/python -m claireai.cli --once "Create a folder called reports in my Downloads"
```

ClaireAI prints the exact tool, arguments, target and approval reference, then
asks `approve? [y/N]`. Only `y`/`yes` runs it. Piped input, cron jobs and
non-interactive shells are refused by design.

## Presence HUD preview

Optional speech output uses the host's eSpeak NG package and an installed audio
player. On Ubuntu-based systems, one-time install:

```bash
sudo apt install espeak-ng pulseaudio-utils
```

Then enable speech, adjust pitch/rate, and use Preview from the HUD settings.
If eSpeak NG is missing, the UI reports that speech is unavailable.

```bash
assistant/.venv/bin/python scripts/presence_preview.py --demo
# then open http://127.0.0.1:8765/
```

Standard library only, loopback only. Without `--demo` it shows `idle` until
something feeds the stream (the desktop window does that in Week 4).
