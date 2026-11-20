# Email and social posting: status and official routes

ClaireAI can **draft** email and social posts locally. It does **not** currently
read mailboxes or publish to any account. That is not a missing button: posting
requires an owner account, an approved API application, scoped credentials and a
per-action approval. No amount of local code removes those requirements.

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
curriculum only contains synthetic or owner-authored examples.

## Why no scraping fallback

Automating a logged-in browser session violates the terms of most services, puts
the owner's account at risk and can expose other people's private messages. The
project does not do it, even when a page is technically reachable. Access that
has not been granted is not data the project may take.
