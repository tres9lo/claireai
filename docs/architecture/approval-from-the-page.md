# Approving an action from the page — security design

**Status: designed, not implemented.** The `/today` page is read-only today and
answers every POST with 501, deliberately. This document is what has to be true
before a button appears on it, so the omission stays a decision rather than an
oversight.

## Why this is not "just add two buttons"

Approval is the product's central safety control (threat-model T4, T5). Today it
runs through `claireai.approval`, and the only brokers are `CliApprovalBroker`
(which refuses when stdin is not a TTY) and `DenyAll`. Adding an HTTP endpoint
would put a *second* approval channel on a port, and a port accepts requests from
anything on the machine — including a web page the owner happens to have open.

A browser will happily POST to `http://127.0.0.1:<port>/approve` from a page on
`evil.example`: the request is a "simple" cross-origin request, so it is sent,
and the fact that the attacker cannot read the response is irrelevant — the
action would already have happened. Any design that does not answer this is a
remote approval bypass, not a feature.

## What the control may and may not do

| May | Must not |
| --- | --- |
| Approve or deny an action the agent has **already proposed** and logged | Name a tool, supply arguments, or start a task |
| Show the exact tool, arguments and target being decided | Read files, list folders, or query anything beyond its own pending item |
| Record who decided and when | Persist "always allow" for a tool or a path |

The endpoint resolves `(task_id, approval_ref)` against the broker's waiting
request. It never accepts an action description from the request body, which is
what keeps threat-model T8 ("the renderer never becomes a control path") true.

## Design

**1. The run token is the second factor, printed where the owner already is.**
`claireai --serve` generates a 32-byte token and prints it in the terminal, never
to a file and never to the audit log. A page that has not been unlocked can
approve nothing, so a web page that discovers the port still cannot act.

**2. Unlock sets a session cookie; the token is not put in a URL.**
`POST /unlock` with the token sets `HttpOnly; SameSite=Strict; Path=/`. The token
itself stays out of browser history and out of the page source. `SameSite=Strict`
means a cross-site POST from `evil.example` arrives *without* the cookie, which
is the property the design depends on.

**3. Every decision needs a single-use nonce bound to the decision.**
When a request is rendered, the server issues a nonce (32 bytes) for that
`approval_ref`. The form posts `ref`, `nonce` and `approve|deny`. The server
consumes the nonce under a lock before acting: a replayed page approves nothing,
and a nonce from one request cannot decide another. Nonces expire with the
request (default `approval_timeout`, 120 s).

**4. Both headers are checked, so a plain form post is not enough.**
The endpoint requires `Content-Type: application/json` and an `X-ClaireAI-Action`
header. A cross-origin HTML form can set neither; a cross-origin `fetch` that
sets a custom header triggers a preflight, which the server refuses by sending no
CORS headers at all.

**5. Decisions are bound to what was shown.**
The broker stores a hash of the rendered arguments with the waiting request. A
decision whose hash does not match is refused (409) and logged, so a stale page
cannot approve a re-proposed action with different arguments.

**6. Nothing else changes.** Binding stays loopback-only (`build_server` already
raises otherwise), there is no cookie-free authorisation path, no LAN listener,
and no token in any log line. The page keeps `default-src 'none'` and gains
`form-action 'self'`; it still ships no JavaScript.

**7. Everything is audited, including refusals.** Each decision writes
`approval.decision` with `actor="owner-via-page"`, the ref, the target and the
decision; failed unlock attempts and replayed or expired nonces write their own
rows so a fishing attempt is visible afterwards.

**8. Bounded retries.** Failed unlocks are rate-limited (10 per minute, then 429
and an audit row). 32 bytes of entropy makes guessing infeasible anyway; the
limit stops a log flood.

## What this design does *not* defend against, stated plainly

- **A process running as the same user.** It can read the token from the
  terminal's memory or the serve process, and could already edit the files
  directly. The token defends against *browser-originated* requests and against
  anything that can reach the port but not the process — it is not a sandbox.
- **A malicious browser extension** with permission to read page content and act
  on behalf of the user. Out of scope, same as for any local web UI.
- **Shoulder-surfing the terminal.** The token is short-lived per run and does
  not grant file access on its own: every approval is still a decision about one
  already-logged proposal.

## Test plan (before this ships)

1. No token → `POST /approve` is 401 and nothing is decided.
2. Wrong token → 401, audit row, rate-limited after 10.
3. Correct token but no nonce / used nonce / expired nonce → 409, no decision.
4. A nonce from request A cannot decide request B.
5. Decision with mismatched argument hash → 409.
6. `Content-Type` or missing custom header → 415/400, no decision.
7. Cross-site simulation: POST with an `Origin` that is not the server's own →
   refused, cookie not honoured.
8. Approve then replay the same body → second attempt is 409, tool runs once.
9. Deny is honoured, logged, and the tool never runs.
10. The token never appears in `/today` HTML, in the audit log, or in an error
    message.

## Where this lives

- `claireai.approval` — a `PageApprovalBroker` implementing the existing broker
  protocol; the agent loop needs no change.
- `claireai.presence_server` (or a sibling) — `/unlock` and `/approve`, still
  loopback-only, still no CORS headers.
- `claireai.today` — renders the pending request and its form; still escapes
  every stored string.

Until all seven points and the test plan exist, `/today` stays read-only and the
CLI remains the only approval channel.
