#!/usr/bin/env bash
# ClaireAI installer. Detects the host, prints every download and asks first.
#   ./install.sh              install the desktop app + voice input
#   ./install.sh --dry-run    print the plan without changing anything
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    exec "$candidate" scripts/bootstrap.py "$@"
  fi
done

echo "ERROR: no Python found. Install Python 3.11+ and run this again." >&2
exit 2
