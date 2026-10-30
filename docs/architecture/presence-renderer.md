# Presence renderer (visual state sync)

The presence renderer is the assistant's face: a local, real-time scene that
shows what ClaireAI is doing and what the machine is doing on its behalf. It is
inspired by Jarvis-style interfaces, with one hard rule enforced everywhere —
**the renderer displays state, it never decides anything**.

Status: **planned**. Contract written now, built in Month 1 Week 4 (v0, canvas
2D) and extended in Months 2–3 (speaking/lip-sync) and Months 6–7 (WebGL
quality). No renderer code exists in the repository yet.

## Where it runs

- Inside the Tauri v2 webview (`desktop/`), as part of the desktop shell.
- It receives state from the Python sidecar over the authenticated local API /
  Tauri events. It does not open the database, read files, run shell commands or
  call the model.
- Later, the Android client may show the same state as a simple mirror; it is
  optional and never a control path for the desktop agent.

## State machine

| State | Meaning | Typical trigger |
| --- | --- | --- |
| `idle` | ready, nothing running | app start, task finished |
| `listening` | microphone is capturing (push-to-talk or wake word) | audio capture started |
| `thinking` | the model is generating | `brain.generate` in flight |
| `tool_running` | a tool is executing | `tool.call` row opened |
| `awaiting_approval` | waiting for the human | task status `awaiting_approval` |
| `speaking` | audio output is playing | Piper playback started |
| `error` | last action failed | task or step marked `failed` |
| `offline` | model runtime unavailable | `BrainUnavailable` |

States are **derived from the task/tool log**, not invented by the UI: if the log
says a step failed, the renderer shows `error`. When the renderer and the log
disagree, the log is right and the renderer is a bug.

## Event contract

```ts
type PresenceEvent = {
  seq: number;              // monotonic per stream; duplicates/regressions dropped
  ts: string;               // ISO-8601 UTC
  taskId: string | null;
  state: "idle" | "listening" | "thinking" | "tool_running"
       | "awaiting_approval" | "speaking" | "error" | "offline";
  detail?: { tool?: string; label?: string; progress?: number };
  active: { microphone: boolean; camera: boolean; screen: boolean; input: boolean };
};
```

- One stream, ordered, resumable: the client reconnects with the last `seq` it
  processed and gets the current snapshot plus subsequent events.
- `active` is what the indicator lights read. It is part of the same event as the
  state so the two cannot drift apart in the UI.
- No free text from files, pages, emails or the model is rendered as HTML. Labels
  are plain text, length-capped.

## Performance budget

Measured target on this development laptop (Intel i5-6300U, integrated GPU,
7.1 GiB RAM, of which the model already holds ~1.2 GiB):

| Budget | Value |
| --- | --- |
| Frame rate | 30 fps default, configurable 5–60 (`CLAIREAI_PRESENCE_FPS_CAP`) |
| CPU when idle | ≤ 5% of one core |
| CPU while animating | ≤ 25% of one core |
| GPU | ≤ 150 MB of textures/geometry in WebGL mode |
| Memory | ≤ 100 MB renderer process overhead |
| Startup | renders a first frame within 500 ms, never blocks the chat box |

Rules: drop frames rather than queue them, no per-frame allocations, pause
animation when the window is hidden or unfocused, and never animate on a timer
faster than the cap.

## Modes and graceful degradation

`CLAIREAI_PRESENCE_MODE` = `off` | `auto` | `2d` | `webgl` (default `auto`).

1. `webgl` — three.js scene (avatar/HUD) with real-time state animation.
2. `2d` — canvas/SVG HUD, no GPU requirement; the guaranteed fallback.
3. `off` — no renderer at all; the chat box, log and approvals still work.

`auto` picks `webgl` only when the webview reports WebGL2 and a smoke animation
holds the frame budget; otherwise `2d`. A renderer crash, a missing GPU or a
disabled WebGL context must never affect a task.

## Accessibility

- Honour the OS "reduce motion" preference (`CLAIREAI_PRESENCE_REDUCE_MOTION`
  overrides it): swap continuous animation for discrete state changes.
- No flashing above 3 Hz; no strobing colour transitions.
- State is conveyed by text/shape as well as colour; every indicator has a label.
- The renderer is non-essential: keyboard and screen-reader users get the same
  information from the task log view.

## Security

- The renderer holds **no** capability. It cannot invoke a tool, approve an
  action, read the database, or talk to the network at runtime.
- No remote assets, fonts, shaders or telemetry: everything is bundled, so the
  window works offline and cannot leak what it displays.
- Any content that originated outside the app (file names, page titles, window
  titles, model text) is untrusted: rendered as text, length-capped, never as
  markup or a link the user might trust.
- The indicator lights are part of the safety contract: camera, screen,
  microphone and input use the same event stream that writes the audit rows.

## Acceptance checks (when built)

1. Force each state and confirm the HUD matches the task log row for the same
   task id.
2. Disable the renderer (`CLAIREAI_PRESENCE_MODE=off`) and confirm every task
   behaves identically, including failures.
3. Idle 60 s in `2d` and in `webgl`; measure CPU and fps and record them here and
   in `PROGRESS.md`.
4. Kill the renderer process mid-task; the task must still complete and log.
5. Toggle reduce-motion and confirm the animation stops while state changes stay
   visible.

## Open questions

- Whether the avatar is 2D art or a 3D scene is a design decision; the contract
  supports either, and `auto` lets the machine decide at runtime.
- Lip-sync fidelity: start with audio-amplitude mouth animation, not visemes.
- Whether the phone mirrors presence at all, or only the task log (Phase 3).
