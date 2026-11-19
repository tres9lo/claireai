"""``python -m claireai.desktop`` opens the native window.

Through the launcher, not the window module: the autostart entry runs exactly
this command, so a core install without the desktop extra must be able to say
what is missing at login instead of raising ``ModuleNotFoundError``.
"""

from claireai.desktop.launcher import main

raise SystemExit(main())
