# Local conversations, directives and preferences

ClaireAI stores new chat turns in the local SQLite database at
`<data-dir>/db/claireai.sqlite3`. The local HUD can list, open and delete
conversations. User and assistant message text is retained verbatim so history
can be displayed and recent turns can provide context. The task/audit log remains
a separate redacted record of tool and runtime events.

Owner directives and UI preferences use separate SQLite tables. Directives are
editable and can be enabled or disabled in the HUD. Preferences include response
style, generation limits, tool routing, folder allow-list, and optional speech
settings. They are read again for future local chat and CLI agent turns; changing
them does not require a command or restart.

The local chat endpoint does not execute tools. Tool-enabled computer tasks
continue through the agent path and its existing sandbox and approval controls.
The loopback HUD requires same-origin JSON requests for writes. It is not an
account boundary against another process running as the same OS user.

On POSIX, ClaireAI requests mode `0700` for its data/database directories and
`0600` for the SQLite file. Other operating systems rely on the user's OS account
and directory ACLs. No transcript sync or upload path is implemented.

This feature stores conversations created after the migration. Earlier prompts
were not retained as complete transcripts and cannot be reconstructed. Deleting
a conversation removes its messages from the conversation tables; this does not
delete separately redacted task/audit records.
