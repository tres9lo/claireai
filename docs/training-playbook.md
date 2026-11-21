# Training playbook: conversation, tools and internet fallback

This playbook describes how to improve ClaireAI without pretending that a
224-wide, 5-layer, 512-token CPU model is ChatGPT. The realistic target is a
narrow local assistant that:

- holds a short, coherent English or French conversation;
- answers from retrieved evidence when it does not know something;
- calls a small registered tool set when the request expresses an action;
- says it does not know instead of inventing a fact or an action.

General ChatGPT parity needs far more data and compute than this machine has.
The practical substitute is retrieval for knowledge, deterministic tools for
actions, and a small model trained for short turns and clean tool calls.

## The three loops

Keep these separate. Mixing them is how a model learns to repeat its own guesses.

1. **Answer loop (online).** The model answers. If it admits uncertainty and the
   request is public, the runtime asks the owner for approval, searches, reads a
   source, and answers with the source attached. Nothing is trained during the
   conversation.
2. **Learning loop (nightly).** Only owner-confirmed, source-bearing lessons are
   compiled into the next corpus. The trainer builds a separate candidate and
   the fixed gate decides whether it may replace the active model.
3. **Product loop (tools).** The registry validates tool names and arguments,
   applies permissions and approvals, executes, logs, and returns the result.
   Training may improve tool selection, but it never grants a capability.

The immediate answer is retrieval. Weight updates are a later, gated batch, not
an online gradient step. That avoids poisoning the model with an arbitrary web
page and avoids rewriting weights while a conversation is in progress.

## Remembered today vs learned into weights

These are deliberately different states, with different files and different
authority:

- **Remembered today (journal).** `learning/journal.jsonl` is the owner-readable
  record of an exchange. It may contain unverified model answers, but those are
  never training targets. The journal is not the active model.
- **Approved material (compiled corpus).** `learning/corpus.jsonl` receives only
  owner-confirmed, source-bearing lessons. Compilation is append-only and does
  not touch the running checkpoint. `learning/replay.jsonl` holds a bounded,
  deterministic sample of older confirmed lessons so new material does not crowd
  out greetings, refusals or tool behaviour.
- **Learned weights (candidate).** Only the nightly worker trains a separate
  checkpoint under `models/candidates/`. The fixed gate compares it with the
  installed checkpoint at the same run count. Passing that gate is still not an
  automatic install: the owner promotes it explicitly, and the previous
  checkpoint remains available for rollback.

So an answer with a citation is available immediately, while the same lesson
changes weights only after confirmation, compilation, a separate training run
and the install gate. No gradient step happens during a conversation.

## Data rules

- Prefer small, licensed, behaviour-specific sets over a huge undifferentiated
  dump. A few hundred filtered tool-call examples can teach more than 100k
  examples the model cannot fit in its context.
- Remove exact duplicates and near-duplicates across train and validation.
- Remove every prompt used by `scripts/conversation_cases.json`,
  `scripts/tool_call_cases.json`, `scripts/tool_call_cases_extended.json` and
  `scripts/model_eval_cases.json` from training. Contamination makes the gate
  meaningless.
- Keep direct-answer and no-tool examples beside tool-call examples. Otherwise
  the model learns to call a tool for every question.
- Record source, licence, revision, retrieval date, filters and SHA-256 before a
  dataset enters a corpus.
- Do not train on raw search results, model guesses or unverified summaries. A
  web page is evidence to cite, not a fact to memorise.

## Supervised fine-tuning recipe

1. Render every example with the exact same chat template used at inference.
   Training with a different template teaches a format the runtime will never
   send.
2. Use response-only loss: mask system, user, and environment-provided tool
   results; keep assistant text, tool-call tokens, and the end-of-turn token
   unmasked. The prompt is still model input and still conditions the answer.
3. For multi-turn examples, decide explicitly which assistant turns are targets.
   Unroll long conversations into shorter examples rather than truncating the
   current request or the system prompt.
4. Keep EOS after every assistant turn so the model learns where to stop.
5. Mask padding and verify that every batch has at least one target token.
6. Prefer short, packed sequences that fit the trained context. Do not silently
   train positions that inference will never use.
7. Use a larger batch and a conservative learning rate for small SFT runs; stop
   on held-out behaviour, not on the lowest training loss.
8. Inspect one decoded batch before a long run: token positions, labels, loss
   mask, first response token, final response token and EOS. Most SFT bugs are
   boundary bugs.

Reference: [When to mask prompt tokens during SFT](https://sebastianraschka.com/faq/docs/when-mask-prompt-tokens.html),
[RLHF Book: instruction fine-tuning](https://rlhfbook.com/c/04-instruction-tuning),
and [Hugging Face chat templates](https://huggingface.co/docs/transformers/chat_templating).

## Tool-call recipe

- Show the model only the tools its deterministic route can justify, and order
  the owning tool first. A smaller schema payload is also a large CPU saving.
- Keep the wire format simple and exactly parseable, for example
  `<tool_call>{"name":"...","arguments":{...}}</tool_call>`.
- Train the tool name, arguments and closing token as assistant output; mask the
  tool result supplied by the runtime.
- Include negative examples: a question that should be answered directly, a
  missing-argument clarification, and a request that must be refused.
- Validate every call against the real registry after generation. A model that
  emits a plausible call to a non-existent tool has not succeeded.
- Evaluate exact tool name, argument validity, no-call correctness and final
  answer quality separately. Do not average them into one misleading score.

Useful open candidates are catalogued in
[`training-data-sources.md`](training-data-sources.md): Hermes Function Calling
v1 and ToolACE first, then bounded samples of Glaive Function Calling v2, xLAM
and Smol-SmolTalk. Start with a few hundred filtered examples per source.

## Internet fallback contract

- The local model may say it does not know. It may not grant itself network
  access.
- A private or first-person request ("my files", "my inbox", "my calendar")
  stays local. Uncertainty is not consent to publish it.
- Search and page reading require the normal approval. The result is untrusted
  evidence, not instructions.
- Show the query, publisher, URL and excerpt before saving. Saving a lesson
  requires a second, explicit owner confirmation.
- Strip prompt-shaped markers from page text. A page cannot become a system
  message later.
- Compile only verified, source-bearing lessons. Replay a bounded sample of
  older confirmed lessons (`learning/replay.jsonl`) so a new lesson does not
  erase greetings, refusals or tool use.
- Train the nightly candidate, run both fixed suites at the same run count, and
  install only through the existing gate. If the gate fails, keep the old model.

This is the safe interpretation of "learn from the internet": retrieve now,
confirm with the owner, train later, and prove the result.

## Full-day task assignment

A day plan is a schedule of owner-assigned requests, not a grant of autonomy.

- Store each item with a local day, start time, request, status and optional
  linked task id. Reuse the existing task state machine.
- Let the owner create the plan in chat or from a CLI/UI command, and show the
  resolved times and targets before saving.
- At the due time, run only safe actions automatically. Approval-gated and
  dangerous actions move to `awaiting_approval` and wait; they never inherit
  approval from the plan.
- Keep the kill switch, step limits, audit rows and daily summary authoritative.
- After a restart, mark interrupted items for review instead of replaying a
  potentially destructive action.
- Acceptance: a plan with one safe task, one file write and one appointment
  reminder. The safe task runs, the write pauses for approval, the reminder is
  listed, and the evening summary reports what actually happened.

## Evaluation gates

Run the same suites at the same run count for the installed model and every
candidate. Report conversation, tool-call, refusal and citation behaviour
separately, plus latency and peak memory. Training loss is not evidence of
conversation quality, tool safety or factual accuracy.

Do not start 10K/30K training because a candidate loads. The recall probe has
failed at every tested long-context length and CPU latency is already high.
Retrieval and a smaller measured experiment are the better next step.

## Candidate acceptance and install checklist

Training writes a candidate; it never promotes it. Use this sequence after the
`train_claireai.py` pipeline finishes:

1. Read the candidate's `.pipeline.json`. Confirm `exit_code` is `0`, the
   `train` stage produced the `.pt` file, and both evaluation stages completed.
   A rate-limited optional *fetch* is allowed; a failed gate is not.
2. Run the fixed suites against the candidate at the same run count used for the
   installed checkpoint. Do not compare a 1-run candidate with a 3-run baseline.

   ```bash
   assistant/.venv/bin/python scripts/eval_model.py \
     --model-checkpoint runtime-data/models/candidates/claire-chat-v15-full.pt --runs 3
   assistant/.venv/bin/python scripts/conversation_eval.py \
     --model-checkpoint runtime-data/models/candidates/claire-chat-v15-full.pt --runs 3
   ```

3. Compare conversation, tool-call validity, refusals, latency and peak memory
   with the installed checkpoint. A lower training loss is not a reason to
   install; a regression in any required suite is a reason not to.
4. Confirm the candidate's manifest context is the context it was actually
   trained on. Do not use an expanded position table as if the new rows had been
   learned.
5. Install only through an explicit owner action (desktop **Install checkpoint**
   or `DesktopRuntime.install_checkpoint`). Keep the previous checkpoint so the
   change is reversible.
6. Record the candidate name, evaluation counts, install decision and rollback
   path in `PROGRESS.md`.

## Further reading

- [SmolLM2: training small language models](https://arxiv.org/abs/2502.02737)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [ToolACE: Winning the Points of LLM Function Calling](https://arxiv.org/abs/2409.00920)
- [xLAM: A Family of Large Action Models](https://arxiv.org/abs/2406.18518)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
