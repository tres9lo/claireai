#!/usr/bin/env bash
# Start ClaireAI with one command.
#
#   ./scripts/start.sh                 # open the desktop window
#   ./scripts/start.sh --cli           # the terminal chat
#   ./scripts/start.sh --once "..."    # any CLI flag, unchanged
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
  echo "No virtual environment yet; the installer will set one up."
  "$root/install.sh"
  python="$venv/bin/python"
fi

# One data directory for the app and the CLI unless the owner set one.
if [ -z "${CLAIREAI_DATA_DIR:-}" ] && [ -d "$root/runtime-data" ]; then
  export CLAIREAI_DATA_DIR="$root/runtime-data"
fi

exec "$python" -m claireai.app "$@"
