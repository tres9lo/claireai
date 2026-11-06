# Local UI approval broker

**Status: implemented for the loopback HUD's Computer task flow.** The separate
`/today` page remains read-only. UI tasks use the existing agent, tool registry,
folder allow-list, risk policy, audit trail, and exact-argument approval request.

## Request flow

1. The owner submits a task from the local HUD to `/api/agent-task`.
2. The normal agent validates a proposed tool call and creates its usual
   `ApprovalRequest` when policy requires owner approval.
3. `LocalUIApprovalBroker` exposes that one pending request to the same-origin
   `/api/approvals` inbox. The UI renders the tool, validated arguments, target,
   effect preview, reference, and expiry using `textContent`.
4. The owner approves or denies the displayed reference. The broker accepts a
   decision only for that live pending reference and never accepts a tool or
   arguments from the decision request body.
5. Decisions expire after 120 seconds and shutdown denies pending approvals.
   The agent records the approval decision and links the resulting tool row to
   the same reference in the append-only task log.

## Security boundary

All `/api/*` routes require a loopback Host, same-origin Origin when supplied,
and an allowed Fetch Metadata site. POSTs require JSON; there are no CORS
headers. A pending decision is one-shot, short-lived, and tied to the validated
request already held by the server. Browser tools are included only when the
owner has enabled browser access. File tools still obey the configured folder
allow-list. The owner must read the approval card; a same-user process or a
malicious browser extension remains outside this local UI boundary.

The UI task endpoint invokes registered computer tools; plain chat and speech
transcription do not. Speech currently enters the editable chat text box, and
the owner must choose whether to send it as chat or as a computer task.

## Verification

`test_ui_computer_task_waits_for_exact_owner_approval` runs a simulated folder
creation through the HTTP server. It checks the displayed validated arguments,
refuses an unknown reference, and proves the folder appears only after the
matching UI approval. `test_ui_approval.py` checks exact-reference decisions,
expiry, and shutdown denial. The server's existing same-origin checks cover the
approval POST route as for other local JSON controls.
