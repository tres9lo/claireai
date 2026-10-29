# Platform support

Feature status is only marked ✅ after it was actually run on that OS. Everything
else says what is missing. A compile or a mocked test is not a platform result.

Last updated: 2026-10-29 (Week 1 foundations).

| Capability | Linux | Windows | Android |
| --- | --- | --- | --- |
| Text chat with local model | 🟨 code complete, ⛔ model runtime not yet installed here | ⛔ never run | ⛔ not started (Phase 3) |
| Local model management | ⛔ not started | ⛔ never run | ⛔ not planned |
| File tools and allow-list | ⛔ Week 2 | ⛔ never run | ⛔ n/a |
| Browser automation | ⛔ Week 3 | ⛔ never run | ⛔ n/a |
| Push-to-talk voice | ⛔ Phase 2 | ⛔ never run | ⛔ later |
| Wake-word detection | ⛔ no "Hey Claire" model exists yet; treat as pending | ⛔ never run | ⛔ separate implementation, may not be permitted in background |
| Daily summary and local log | 🟨 task/audit log works; summary agent ⛔ Week 4 | ⛔ never run | ⛔ client phase |
| Installer / package | ⛔ not started | ⛔ not started | ⛔ not started |

## Tested environments

| OS | Version | Python | Result |
| --- | --- | --- | --- |
| Linux (Zorin OS) | 18.1, x86_64, 7.1 GiB RAM | 3.12.3 | 49 unit tests pass; CLI migration + config paths verified; inference not yet verified (Ollama not installed when this ran) |
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
