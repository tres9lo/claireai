# Threat model

Scope: the ClaireAI desktop assistant on the owner's own machine — the agent
loop, the tool registry, file access, the presence renderer, and the planned
camera/screen/cursor capabilities. Updated 2026-10-31. Not yet in scope: the
phone client, LAN sync, email/calendar, and packaging.

This is a working document, not a claim of completeness. Every row says what is
implemented **today**, or marks the control as required-when-built.

## Assets

| Asset | Why it matters |
| --- | --- |
| The owner's files | Irreplaceable; a wrong move or overwrite is real damage |
| Screen and camera content | Banking tabs, messages, rooms in the house — the most private data here |
| Typed text and clipboard | Passwords, tokens, personal notes |
| The audit log | The only account of what the assistant did |
| Permission state | Allow-list, capability list, approval decisions |
| The model runtime | Local only; prompts must never leave the machine |

## Trust boundaries

```
 owner ──prompt──▶ CLI/UI ──request──▶ agent loop ──tool──▶ filesystem / OS
                    ▲                    │
                    │ approval           ├──▶ model (Ollama, loopback)
                    └──── decision ◀─────┘
 renderer ◀── PresenceEvent (read-only, loopback) ── agent / API
 untrusted input: file contents, file names, screenshots, OCR, clipboard,
                  window titles, web pages, model output
```

Two rules hold the design together:

1. **The model is untrusted.** It proposes; code validates, authorises and logs.
2. **The renderer is not privileged.** It receives state; it cannot act.

## Threats and controls

| # | Threat | Current controls | Status |
| --- | --- | --- | --- |
| T1 | Prompt injection in a file, screenshot or clipboard makes the assistant call a write tool | Tools are separate from text; the model cannot grant itself approval; every write needs a human decision bound to the exact arguments (roadmap 7.9). Perception output is documented as untrusted. | File path **implemented**; perception **required when built** |
| T2 | A screenshot or camera frame is stored or uploaded | Frames stay in memory; no upload path exists in this codebase; saving requires an explicit write tool and approval; indicators always on. | **Required when built** (no capture code yet) |
| T3 | A cursor/keyboard action clicks something destructive | `dangerous` risk, per-action approval, dry run, step cap, kill switch, no permanent delete tool. | **Required when built** |
| T4 | Approval is granted for one action but a different action runs | Approval request carries the validated arguments and target; the tool row stores the same `approval_ref`; validation happens *before* asking. Tests assert both rows share the reference. | **Implemented** |
| T5 | A piped/automated run self-approves | `CliApprovalBroker` refuses when stdin is not a TTY; `DenyAll` is the default for any programmatic run; there is no "always approve" broker in the product. | **Implemented** |
| T6 | Typed secrets or file contents end up in the log | `redact()` on request text, targets and details; tool results are previewed (160 chars) rather than stored whole; typed text is to be logged by key name only. | Partial (**implemented** for file tools; **required** for keyboard) |
| T7 | A malicious file name injects markup into the renderer | Renderer treats every label as text, never HTML; length caps; no remote assets. | **Implemented** in the v0 HUD (`textContent` only) |
| T8 | The renderer or its HTTP server becomes a control path | Server serves one static file, one generated page and read-only state; no POST handler (every POST answers 501); binds loopback only and refuses a non-loopback host; no tool invocation path. The `/today` page opens SQLite with `mode=ro`, sends a `default-src 'none'` CSP, contains no JavaScript, no form and no button, and escapes every stored string. There are deliberately no approve/deny controls on it. | **Implemented** (tested) |
| T9 | The presence server is exposed to the LAN | `build_server` raises unless the host is loopback; no CORS headers; no auth token (a loopback-only reviewer). | **Implemented** |
| T10 | An agent loops forever or burns the CPU | `agent_max_steps` (default 8) fails the task and logs `agent.step_limit`; step counter is explicit. An empty reply triggers **exactly one** further model attempt, gated on tokens actually having been generated and on the decoding not being pinned to temperature 0, and is logged as `agent.retry` — a retry that cannot succeed is refused rather than looped. | **Implemented** |
| T11 | Symlink or a path trick escapes the allow-list | Every path is resolved (symlinks followed) then contained in a root; a link inside the allow-list pointing out is refused; relative names must match a root. | **Implemented** (security tests) |
| T12 | TOCTOU: the path is swapped between check and write | Paths are resolved once and used; the window is small but real on a shared machine. No openat/O_NOFOLLOW equivalent yet. | **Accepted residual risk** |
| T13 | Trash and summaries grow until the disk fills | Trash lives in the app data dir with per-run timestamps. `claireai --prune` plans removals and `--prune --apply` carries them out, refusing symlinked batches, unparseable names and any path outside the data dir. The audit log is reported (rows and bytes) and deliberately not pruned — dropping its triggers to allow deletion would remove the append-only guarantee. | **Implemented** for trash and summaries; audit log growth **accepted by design** |
| T14 | The model claims an action it did not perform | System prompt forbids it; every action has a log row; the final answer is not treated as evidence. No automatic cross-check yet. | Partial |
| T15 | The audit log is edited afterwards | Append-only triggers reject UPDATE/DELETE; the smoke check verifies it. | **Implemented** |
| T16 | Overwriting a file the owner wanted to keep | Writes require approval; `create_text_file` refuses existing files; `move_file` never overwrites; `rename_file` refuses a taken name. `move_file` also refuses arguments that arrive reversed (a folder offered as the source with a file-shaped destination), observed for real with a thin model, so the refusal lands before the approval prompt rather than after it. | **Implemented** |

## Assumptions

- The owner's machine account is not already compromised. A same-user attacker
  can read the database and the config regardless of what this app does.
- Loopback is trusted. Anyone who can bind `127.0.0.1` is already the same user.
- The model weights and the Ollama runtime are not adversarial. They are
  downloaded out of band and recorded in `docs/model-and-data-licenses.md`.
- The human reads the approval prompt. The prompt shows tool, arguments, target
  and reference; it does not hide the effect behind a summary.

## Residual risks we are not fixing yet

1. **Approval fatigue.** Many prompts train the owner to press `y`. Mitigation
   later: trusting named patterns after several identical approvals, never for
   `dangerous` tools.
2. **Screenshot exfiltration by a future model.** Today nothing captures; when it
   does, image data must never enter a prompt sent anywhere but loopback.
3. **Audit-log growth** (T13). Trash and summaries are pruned; `task_log` is not,
   because the append-only triggers are worth more than the kilobytes. It grows a
   few rows per task, forever.
4. **TOCTOU** (T12) on a machine where another process mutates the same folder.
5. **Windows-specific paths** are unit-tested in CI but not exercised by a human;
   path confusion bugs would most likely appear there.

## Verification checklist

- [x] Allow-list escape attempts (parent traversal, absolute outside, file and
      directory symlinks) are refused and tested.
- [x] A denied approval leaves the filesystem untouched, with both log rows
      present and sharing one reference.
- [x] A non-interactive run cannot approve anything.
- [x] The audit log rejects UPDATE and DELETE (checked by `scripts/smoke.py`).
- [x] The generated page cannot write: SQLite is opened `mode=ro`, a write
      attempt raises, and the file's mtime is unchanged after repeated requests.
- [x] No HTTP path grants a capability: POST to any path answers 501.
- [x] Retention deletes only what is old, keeps recent batches, and refuses
      symlinked or unrecognised trash folders.
- [ ] Camera/screen indicators appear only while capture is active.
- [ ] Kill switch aborts a cursor sequence and logs `cancelled`.
- [ ] Typed-secret redaction verified against a real keyboard tool.
- [ ] Windows manual run of the CLI, approval prompt and trash folder.
