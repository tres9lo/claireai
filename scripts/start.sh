#!/usr/bin/env bash
# Start ClaireAI with one command.
#
#   ./scripts/start.sh                 # start the local app, open it
#   ./scripts/start.sh --no-browser    # start it, print the address
#   ./scripts/start.sh -- <cli args>   # any CLI flag, e.g. --summary
#
# Non-destructive by design: it creates the virtual environment only if it is
# missing, never runs sudo, and never touches anything outside this checkout and
# the ClaireAI data directory.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$root/assistant/.venv"
python=""

if [ -x "$venv/bin/python" ]; then
  python="$venv/bin/python"
else
  echo "No virtual environment yet; creating one (first run only)."
  if command -v uv >/dev/null 2>&1; then
    uv venv --seed --python 3.12 "$venv"
    uv pip install --python "$venv/bin/python" -e "$root/assistant[dev]"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$venv"
    "$venv/bin/python" -m pip install --upgrade pip >/dev/null
    "$venv/bin/python" -m pip install -e "$root/assistant[dev]"
  else
    echo "ERROR: no python3 found. Install Python 3.11+ and run this again." >&2
    exit 2
  fi
  python="$venv/bin/python"
fi

# One data directory for the app and the CLI unless the owner set one.
if [ -z "${CLAIREAI_DATA_DIR:-}" ] && [ -d "$root/runtime-data" ]; then
  export CLAIREAI_DATA_DIR="$root/runtime-data"
fi

exec "$python" -m claireai.app "$@"
