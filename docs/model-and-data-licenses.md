# Model and data licences

Every model, voice, wake-word asset and dataset must be recorded here with the
version checked, the licence, the source URL, the date checked, and whether
redistribution is allowed. Downloading something for local personal use is not
permission to redistribute it or bundle it in an installer.

## Local assistant model (Track A)

| Component | Version / tag | Licence | Source | Checked | Redistribution |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5 1.5B Instruct (Ollama packaging) | `qwen2.5:1.5b`, Q4_K_M, 986 MB | Apache License 2.0 (all Qwen2.5 sizes except 3B and 72B, which use the Qwen licence) | <https://ollama.com/library/qwen2.5:1.5b> (tag page includes the licence text); <https://huggingface.co/Qwen> | 2026-10-29 | Apache-2.0 permits redistribution if the licence and notices are retained — still confirm the exact upstream model card before bundling. Not bundled in this repo. |
| Ollama runtime | not installed when this was written | MIT (verify on the release you install) | <https://github.com/ollama/ollama> | pending install | n/a until packaging |

The 3B and 72B Qwen2.5 variants use a different (Qwen) licence; if the
configured model changes to one of those, re-check before use.

## Speech and wake word (Phase 2)

Not yet selected. faster-whisper model weights, Piper voices and any
openWakeWord or custom "Hey Claire" model each need their own entry here before
use, including whether redistribution is permitted.

## Training data (Track B, `brain-lab/`)

No datasets have been downloaded. When data is added, record per-dataset:
source URL, licence, retrieval date, size, and processing steps in
`brain-lab/data/manifests/` (tracked) while the raw files stay git-ignored.

## Presence renderer (planned, Month 1 Week 4)

Not installed yet — the desktop shell does not exist, so nothing is bundled.
These are the intended packages, checked from upstream package metadata:

| Component | Version seen | Licence | Source | Checked | Redistribution |
| --- | --- | --- | --- | --- | --- |
| three.js | 0.187.0 | MIT | <https://registry.npmjs.org/three/latest> (license field); upstream `LICENSE` | 2026-10-31 | Yes, with the copyright notice retained |
| @react-three/fiber | 9.8.2 | MIT | <https://registry.npmjs.org/@react-three/fiber/latest> | 2026-10-31 | Yes, with the copyright notice retained |

Every bundled asset (avatar art, meshes, textures, shaders, fonts, sounds) needs
its own row here before it ships, including the licence, source URL and whether
redistribution is allowed. No runtime CDN fetches: an offline assistant cannot
depend on a remote asset host, and remote assets would leak what is displayed.

## Machine access (planned, Months 2–3)

| Component | Version seen | Licence | Use | Checked |
| --- | --- | --- | --- | --- |
| `mss` | 10.2.0 | MIT | screenshots | 2026-10-31 |
| `opencv-python` | 5.0.0.93 | Apache-2.0 | camera frames | 2026-10-31 |
| `pyautogui` | 0.9.54 | BSD | cursor/keyboard on X11 and Windows | 2026-10-31 |
| `Pillow` | 13.0.0 | MIT-CMU per upstream LICENSE (PyPI metadata carries no licence classifier — confirm at packaging) | image handling | 2026-10-31 |
| `pynput` | 1.8.2 | LGPLv3 | **not adopted**: LGPL obligations are not worth it while `pyautogui` covers the need | 2026-10-31 |

System dependencies, not distributed with this repository: `xdg-desktop-portal`
(Wayland screen/input grants), `ydotool` + `uinput` permissions (Wayland input),
`wmctrl`/`xdotool` (X11 window control), Windows `SendInput` (OS API).

Camera frames and screenshots are processed in memory and never uploaded; if a
task saves one, the saved file is the owner's data and follows the same rules as
any other file ClaireAI writes.

## Dependencies

Python dependencies are pinned in `assistant/requirements.lock.txt`. A full
licence inventory is required before packaging (roadmap 7A.10).
