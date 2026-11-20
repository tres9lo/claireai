# Training data sources and access

This is a curated starting point, not a bulk-download script. The machine-readable
manifest is [`brain-lab/data-sources.json`](../brain-lab/data-sources.json). Every
source must be recorded with its licence, retrieval date, transformations and
attribution before it enters a corpus.

**Licence rule:** if a source says no LLM training, requires a login, or requires
an agreement you have not accepted, the answer is **no data**. "No permission"
does not mean permission to scrape. That protects the project and the people
whose text it is.

## Recommended first mix

Start small enough to measure. The current model has a 288-token context and a
small vocabulary; dumping terabytes into it will not fix that.

| Share | Content | Sources |
| --- | --- | --- |
| 35–40% | General educational prose | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [Wikipedia dumps](https://dumps.wikimedia.org/), [OpenStax](https://openstax.org/) |
| 15–20% | Human conversation and helpfulness | [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1), [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) |
| 15% | Mathematics and worked reasoning | [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) |
| 10% | Code and tool use | [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2), [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) |
| 10% | Reference, business and analysis | [World Bank](https://data.worldbank.org/), [SEC EDGAR](https://www.sec.gov/edgar/sec-api-documentation), [OpenAlex](https://openalex.org/) |
| 5% | Your own vetted notes and drafts | Author-owned material only |

Do not train on the evaluation cases in `scripts/conversation_cases.json`,
`scripts/tool_call_cases.json` or `scripts/model_eval_cases.json`. Keep the
held-out split separate.

## What each source is good for

- **Chat and advice style:** OASST1, Dolly 15k and HelpSteer2. They teach format,
  helpfulness and ranking. They do **not** establish medical, legal or financial
  truth.
- **Reading and analysis:** FineWeb-Edu, Wikipedia, OpenStax, OpenAlex and SEC
  filings. Use retrieval and cite the source; do not ask a 3M-parameter model to
  memorise a library.
- **Calculation:** GSM8K and OpenMathInstruct-2 teach worked arithmetic, but the
  runtime now answers plain arithmetic, percentages and unit conversions with
  deterministic code (`claireai/calculations.py`). Keep the model away from
  calculations that code can do exactly.
- **Business:** World Bank and SEC filings for primary data; your own invoices,
  reports and policies after removing personal data. Never use a booking site,
  mailbox or social account as a training corpus.
- **Hospitality:** Wikivoyage and your own approved policies/FAQs. There is no
  legitimate public corpus of private hotel conversations.
- **Email and social:** use owned or synthetic drafts. Do not train on another
  person's inbox or private messages.

## Access you must request

These links are official starting points. Most require accepting terms, creating
an app, or an administrator's approval. They are listed so the requirement is
visible, not because access has been granted.

| Service | Official route | What is required |
| --- | --- | --- |
| Gmail | [Gmail API overview](https://developers.google.com/workspace/gmail/api/guides) | Google Cloud project, OAuth consent, restricted-scope verification for `gmail.readonly` / `gmail.send` |
| Microsoft mail | [Graph mail API](https://learn.microsoft.com/graph/api/resources/mail-api-overview) | Entra app registration and delegated `Mail.Read` / `Mail.Send` permission |
| WhatsApp | [WhatsApp Business Cloud API](https://developers.facebook.com/docs/whatsapp/cloud-api) | Meta business app, business phone number, approved templates, recipient opt-in |
| Instagram | [Instagram Platform](https://developers.facebook.com/docs/instagram-platform) | Professional account, Meta app, permissions and app review |
| X | [X API](https://docs.x.com/x-api) | Developer account and a paid access tier |
| LinkedIn | [LinkedIn APIs](https://learn.microsoft.com/linkedin/) | Approved developer access |
| Common Crawl | [Terms of use](https://commoncrawl.org/terms-of-use) | Read and accept the terms before fetching |
| Gated Hugging Face datasets | Dataset page → **Request access** | Accept the dataset's conditions; wait for approval where required |

The official APIs are the only supported route for reading or posting to an
account. The app does not automate a browser session, guess credentials, or
scrape a logged-in page.

## Before a source enters training

1. Read the dataset card and the actual licence, not just the repository name.
2. Record: source URL, version/revision, licence, date checked, attribution text,
   filters applied and the SHA-256 of the local file.
3. Remove personal data, secrets, private messages and any content whose licence
   forbids this use.
4. Keep a held-out evaluation set that never appears in training.
5. Measure with `scripts/eval_model.py`, `conversation_eval.py` and
   `tool_call_eval.py`; a lower training loss is not proof of a better assistant.
6. Update [`docs/model-and-data-licenses.md`](model-and-data-licenses.md).

## Sources deliberately excluded

- **Stack Exchange data dump:** the current distribution requires agreeing not to
  use it for LLM training. Do not work around that.
- **Enron email corpus:** private communications and personal data.
- **Logged-in social feeds, WhatsApp chats, inboxes or private groups:** no
  permission, no corpus.
- **Books or articles without a reuse licence:** a public web page is not a
  training licence.
- **Scraped hotel, medical, legal or financial advice:** high risk of personal
  data, stale facts and unlicensed text.
