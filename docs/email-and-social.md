# Email and social posting: status and official routes

ClaireAI can **draft** email and social posts locally. It can also read the
**unread headers** of an IMAP inbox through TLS, opened read-only with
`BODY.PEEK`, so no message is marked seen, deleted, moved or sent. It does **not**
publish to any account. That is not a missing button: posting requires an owner
account, an approved API application, scoped credentials and a per-action
approval. No amount of local code removes those requirements.

## Read-only inbox

```bash
claireai --email-setup       # verify first, then store in the OS keyring
claireai --inbox 10
```

`--email-setup` asks for the address, IMAP host/port and app password with hidden
input. It **verifies** a read-only IMAP login before storing anything, and it
never falls back to a plaintext file. The address, host, port and app password
are stored in the OS keyring, so `--inbox` also works when the window is started
from the desktop menu; `CLAIREAI_EMAIL_ADDRESS` and
`CLAIREAI_EMAIL_IMAP_HOST` / `CLAIREAI_EMAIL_IMAP_PORT` still override them.
Only `FROM`, `SUBJECT` and `DATE` headers are fetched. There is no send, delete,
move or flag-changing path in `claireai.email_read`.

### Gmail App Password

1. Open your Google Account → **Security**.
2. Turn on **2-Step Verification**. App Passwords are unavailable without it.
3. Open **App Passwords** (search for it in Security if it is not shown).
4. Create a password named `ClaireAI` and copy the 16-character value.
5. Run `claireai --email-setup`, enter the Gmail address, accept
   `imap.gmail.com:993`, and paste the App Password at the hidden prompt.
6. Check it:

```bash
claireai --inbox 10
```

If you prefer environment overrides for a particular shell:

```bash
export CLAIREAI_EMAIL_ADDRESS='you@gmail.com'
export CLAIREAI_EMAIL_IMAP_HOST='imap.gmail.com'
claireai --inbox 10
```

Use the App Password, **not** your normal Google password. If the login is
rejected, revoke the old App Password, create a new one and run setup again.
Microsoft 365 may require an app password or OAuth depending on the tenant;
OAuth reading is not implemented yet. Without credentials, `--inbox` reports the
missing configuration and exits without connecting.

## Official APIs

| Platform | Documentation | Access requirement |
| --- | --- | --- |
| Gmail | <https://developers.google.com/workspace/gmail/api/guides> | Google Cloud project, OAuth consent screen, restricted scopes (`gmail.readonly`, `gmail.send`) and verification for external users |
| Microsoft 365 mail | <https://learn.microsoft.com/graph/api/resources/mail-api-overview> | Entra app registration, delegated `Mail.Read` / `Mail.Send` |
| WhatsApp Business | <https://developers.facebook.com/docs/whatsapp/cloud-api> | Meta business portfolio, business phone number, templates and recipient opt-in |
| Instagram | <https://developers.facebook.com/docs/instagram-platform> | Instagram Professional account, Meta app, permissions and app review |
| Facebook Pages | <https://developers.facebook.com/docs/pages-api/> | Meta app, Page role and approved permissions |
| X | <https://docs.x.com/x-api> | Developer account and paid API access |
| LinkedIn | <https://learn.microsoft.com/linkedin/> | Approved LinkedIn developer access |

## Design rules for a future connector

1. **Credentials stay outside the repo.** Use the OS keyring; never put a token
   in a config file, a prompt, a log or a commit.
2. **Preview before sending.** The owner sees the exact recipient, subject, body
   and attachments, and approves that specific message.
3. **One action, one approval.** A blanket "allow email" is not enough for a
   send, delete or bulk operation.
4. **Read only what is needed.** Request the narrowest scope; do not download an
   entire mailbox to train on it.
5. **No cold outreach or automated follow-ups.** Respect platform policy, rate
   limits and the recipient's consent.
6. **Record the result.** Audit the provider, account, message id and status; do
   not record the token or full private body.

Until those credentials exist, the safe workflow is: ClaireAI drafts, the owner
copies the draft into the official client, and the owner sends it. The training
curriculum only contains synthetic, owner-authored, or owner-confirmed
source-bearing examples; an ordinary model answer is journaled but never
compiled into training data.

## Why no scraping fallback

Automating a logged-in browser session violates the terms of most services, puts
the owner's account at risk and can expose other people's private messages. The
project does not do it, even when a page is technically reachable. Access that
has not been granted is not data the project may take.
