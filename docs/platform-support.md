# Platform support

Feature status is only marked ✅ after it was actually run on that OS. Everything
else says what is missing. A compile or a mocked test is not a platform result.

Last updated: 2026-10-30 (Week 1 acceptance passed on Linux: real local
inference, offline, logged).

| Capability | Linux | Windows | Android |
| --- | --- | --- | --- |
| Text chat with local model | ✅ tested 2026-10-30 with Ollama 0.40.2 + `qwen2.5:1.5b` (online, offline and logged) | ⛔ never run | ⛔ not started (Phase 3) |
| Local model management | ⛔ not started | ⛔ never run | ⛔ not planned |
| File tools and allow-list | ⛔ Week 2 | ⛔ never run | ⛔ n/a |
| Browser automation | ⛔ Week 3 | ⛔ never run | ⛔ n/a |
| Push-to-talk voice | ⛔ Phase 2 | ⛔ never run | ⛔ later |
| Wake-word detection | ⛔ no "Hey Claire" model exists yet; treat as pending | ⛔ never run | ⛔ separate implementation, may not be permitted in background |
| Daily summary and local log | 🟨 task/audit log works; summary agent ⛔ Week 4 | ⛔ never run | ⛔ client phase |
| Installer / package | ⛔ not started | ⛔ not started | ⛔ not started |
| Local model management (pull/list/health) | ✅ reachability + installed-model checks with `scripts/doctor.py`; no pull-from-app UI | ⛔ never run | ⛔ not planned |

## Tested environments

| OS | Version | Python | Result |
| --- | --- | --- | --- |
| Linux (Zorin OS) | 18.1, x86_64, Intel i5-6300U, 7.1 GiB RAM | 3.12.3 | 75 unit tests pass; real local inference verified online and in a network namespace with no route; task/audit rows verified in SQLite; 9/10 model eval cases stable |
| Windows | — | — | not tested |
| Android | Tecno Spark 20 | — | not tested |

## Known platform differences to keep in mind

- Data locations come from `platformdirs` (`~/.local/share/ClaireAI` on Linux,
  `%LOCALAPPDATA%\ClaireAI\ClaireAI` on Windows). Use `CLAIREAI_DATA_DIR` to
  override for development; never hard-code a home path in shared code.
- `python3 -m venv` needs the `python3-<version>-venv` system package on
  Debian/Ubuntu-family systems. `uv venv --seed` avoids that dependency.
- Ollama installs via an install script + systemd service on Linux and an
  installer/tray app on Windows; both expose the same loopback HTTP API, so the
  adapter code is shared.
- Audio, microphone and desktop integration will need manual acceptance testing
  per OS; CI cannot cover them.
- Measured on the machine above: `qwen2.5:1.5b` Q4_K_M holds ~1.12 GiB RSS in
  `llama-server`, ~57 MiB in `ollama serve`; first call ~8 s (model load), warm
  short replies 1.2–2.5 s. Any larger default model needs a memory decision
  first, especially on 8 GiB Windows laptops.
- On Linux the snap build runs `ollama serve` plus a `llama-server` child; the
  model files live under `/var/snap/ollama/common/models/blobs/`. Do not assume
  a user-writable model path when packaging.
