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

## Dependencies

Python dependencies are pinned in `assistant/requirements.lock.txt`. A full
licence inventory is required before packaging (roadmap 7A.10).
