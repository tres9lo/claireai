# Security policy

ClaireAI is designed around a single assumption: **the model is untrusted**. The runtime — not the prompt — decides what a tool may touch. Security reports are taken seriously for exactly that reason.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Use GitHub's private reporting flow: **Security → Report a vulnerability** on <https://github.com/tres9lo/claireai/security/advisories/new>.

Include:

- the version or commit you tested;
- your operating system and Python version;
- the smallest reproduction you can manage;
- what you expected to happen, and what happened instead.

You will get an acknowledgement, an assessment, and a fix or a documented decision. Please give a reasonable window before public disclosure.

## In scope

Anything that breaks one of the guarantees the project claims:

- **Sandbox escape** — reading or writing outside `CLAIREAI_ALLOWED_FOLDERS`, including via symlinks, `..` traversal, drive-relative paths on Windows, or path separators that parse differently than they resolve.
- **Approval bypass** — executing a write without an explicit interactive approval, or self-approving in a non-interactive run.
- **Audit tampering** — any path that updates or deletes `task_log` rows, or records a tool call without the matching approval reference.
- **Risk-policy bypass** — a tool whose declared risk does not match what it does, or a call that reaches execution above the configured ceiling.
- **Injection through content** — a file, web page or tool result that causes the runtime to execute something the owner did not approve.
- **Data exposure** — secrets written to the database, logs or disk unredacted; the `/today` page or presence server exposing state beyond loopback, or accepting a state-changing request.
- **Dependency compromise** in the runtime path.

## Out of scope

- The model producing wrong, unhelpful or creative answers. That is a model-quality issue, not a vulnerability.
- Prompt injection that only changes what the model *says*. It becomes in scope the moment it changes what the runtime *does*.
- Attacks that already require code execution on the machine as the owner.
- The planned, unimplemented features (camera, screen, cursor, keyboard) unless a shipped stub already weakens a boundary.

## Security defaults you should know

- File tools are deny-by-default; the allow-list starts empty.
- The model endpoint must be loopback unless explicitly overridden.
- Write tools ask every time and never delete permanently.
- `task_log` is append-only enforced by triggers.
- Secrets are redacted and long strings truncated before storage.
- Servers bind to loopback and serve read-only state; there is no POST route.

## Supported versions

ClaireAI has not had a public release. Fixes land on `main`, and only the latest commit is supported.
