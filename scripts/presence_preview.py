#!/usr/bin/env python3
"""Serve the presence HUD for a visual check.

    assistant/.venv/bin/python scripts/presence_preview.py --demo

Loopback only, no extra dependencies: the HUD is a canvas 2D page served by the
standard library, driven by the same PresenceEvent stream the agent emits. Use
`--demo` to cycle states without running the model.
"""

from __future__ import annotations

import sys

from claireai.presence_server import main

if __name__ == "__main__":
    sys.exit(main())
