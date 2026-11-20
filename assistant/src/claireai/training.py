"""Opt-in background curriculum and bounded local training runs.

The journal in :mod:`claireai.learning` records what actually happened. This
module turns that material plus a small, authored practice curriculum into a
corpus the real trainer understands, then runs one bounded CPU training job when
the owner has allowed self-improvement.

Deliberate limits:

* it runs only while the **Self-improvement** switch is on, at most once a day,
  and only when the network is reachable (the check sends no data);
* training is local and bounded by ``CLAIREAI_TRAINING_STEPS``;
* the result is a *candidate* in ``<data>/models/candidates/``. Nothing is
  promoted automatically: the existing evaluation/install gate still decides,
  so a bad run cannot silently replace the assistant;
* it does not log into email or social networks, post, scrape, or use the
  owner's accounts. Those need separate connectors, credentials and approvals.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from claireai.gate import CandidateGate
from claireai.learning import LearningError, LearningJournal

SYSTEM = (
    "You are ClaireAI, a local-first assistant. Be concise, correct and honest. "
    "Show the calculation when the answer is numeric, and never claim an action "
    "you did not take."
)
DEFAULT_STEPS = 200
MAX_STEPS = 5_000
DEFAULT_RUN_INTERVAL_HOURS = 24
TICK_SECONDS = 3600.0
ONLINE_HOST = ("1.1.1.1", 443)

DOMAINS: dict[str, str] = {
    "conversation": "talking with people",
    "hospitality": "hotel and guest service",
    "reports": "reports and summaries",
    "accounting": "accounting and invoices",
    "arithmetic": "numbers and arithmetic",
    "calculations": "unit and time calculations",
    "coding": "small programming tasks",
    "routines": "daily routines and planning",
    "email": "email drafting",
    "social": "social media drafting",
    "research": "research and sources",
    "business": "business, sales and margins",
    "intent": "understanding what the owner is asking for",
}


def _record(domain: str, user: str, assistant: str) -> dict[str, Any]:
    return {
        "system": SYSTEM,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": f"claireai-curriculum:{domain}",
    }


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _arithmetic() -> list[dict[str, Any]]:
    rng = random.Random(7)
    examples = []
    for _ in range(8):
        left, right = rng.randint(12, 980), rng.randint(3, 240)
        examples.append(_record("arithmetic", f"What is {left} + {right}?", f"{left} + {right} = {left + right}."))
    for _ in range(4):
        part, whole = rng.randint(5, 80), rng.randint(100, 400)
        examples.append(
            _record(
                "arithmetic",
                f"What is {part}% of {whole}?",
                f"{part}% of {whole} = {part / 100 * whole:.2f}.",
            )
        )
    return examples


def _accounting() -> list[dict[str, Any]]:
    rng = random.Random(11)
    examples = []
    for _ in range(6):
        amount = rng.randint(80, 900)
        rate = rng.choice((5, 10, 15))
        tax = amount * rate / 100
        examples.append(
            _record(
                "accounting",
                f"An invoice is {amount} before {rate}% tax. What is the total?",
                f"Tax is {amount} × {rate}% = {tax:.2f}; total = {amount + tax:.2f}.",
            )
        )
    revenue, cost = 4200, 2700
    examples.append(
        _record(
            "accounting",
            "Revenue is 4200 and cost is 2700. What is the gross profit margin?",
            f"Gross profit = 4200 - 2700 = 1500. Margin = 1500 / 4200 = 35.71%.",
        )
    )
    return examples


def _reports() -> list[dict[str, Any]]:
    return [
        _record(
            "reports",
            "Write a one-line weekly report: 12 tasks done, 2 failed, 3 pending.",
            "Weekly report: 12 tasks completed, 2 failed, 3 still pending.",
        ),
        _record(
            "reports",
            "Summarise sales of 120, 90 and 150 units.",
            "Sales were 120 + 90 + 150 = 360 units; the average is 120 units.",
        ),
        _record(
            "reports",
            "Turn these numbers into a short status: 8 of 10 checks passed.",
            "Status: 8 of 10 checks passed (80%); 2 checks need attention.",
        ),
        _record(
            "reports",
            "What should a useful incident report contain?",
            "What happened, when, who was affected, the impact, the cause if known, "
            "what was done, and the next action with an owner.",
        ),
    ]


def _calculations() -> list[dict[str, Any]]:
    return [
        _record("calculations", "How many minutes are in 2.5 hours?", "2.5 × 60 = 150 minutes."),
        _record("calculations", "Convert 7.5 km to metres.", "7.5 × 1000 = 7500 metres."),
        _record("calculations", "A job takes 3 people 8 hours. How many person-hours is that?", "3 × 8 = 24 person-hours."),
        _record("calculations", "What is 18:40 plus 95 minutes?", "95 minutes is 1 h 35 min, so 18:40 + 1:35 = 20:15."),
    ]


def _coding() -> list[dict[str, Any]]:
    return [
        _record(
            "coding",
            "What does sorted([3, 1, 2]) return in Python?",
            "sorted([3, 1, 2]) returns [1, 2, 3]; it leaves the original list unchanged.",
        ),
        _record(
            "coding",
            "How do you read a text file safely in Python?",
            "Use a context manager: `with open(path, encoding='utf-8') as handle: text = handle.read()`.",
        ),
        _record(
            "coding",
            "What is the difference between a list and a tuple?",
            "A list is mutable; a tuple is immutable and can be hashed if its items are hashable.",
        ),
        _record(
            "coding",
            "How do you handle a missing dictionary key in Python?",
            "Use `data.get(key, default)`, or catch `KeyError` when absence is exceptional.",
        ),
    ]


def _conversation() -> list[dict[str, Any]]:
    return [
        _record("conversation", "Hello", "Hello. What would you like to work on?"),
        _record(
            "conversation",
            "How are you?",
            "I do not have feelings, but I am running well and ready to help.",
        ),
        _record(
            "conversation",
            "Thank you",
            "You are welcome. Tell me if you want the next step.",
        ),
        _record(
            "conversation",
            "Can you explain that more simply?",
            "Yes. I will keep the same answer but use shorter sentences and one example.",
        ),
        _record(
            "conversation",
            "I do not understand",
            "That is fair. Which part is unclear: the goal, the steps, or the result?",
        ),
        _record(
            "conversation",
            "Goodbye",
            "Goodbye. Your work is saved locally.",
        ),
        # Multi-turn intent repair. These teach the model to ask for the missing
        # slot instead of guessing when a request is ambiguous, and to carry the
        # clarified meaning into the next answer.
        {
            "system": SYSTEM,
            "messages": [
                {"role": "user", "content": "Move it to the other folder."},
                {
                    "role": "assistant",
                    "content": "Which file do you mean, and which folder should it move to?",
                },
                {"role": "user", "content": "the budget file, to reports"},
                {
                    "role": "assistant",
                    "content": "I understand: move the budget file into reports. I will show the exact move for approval before anything changes.",
                },
            ],
            "source": "claireai-curriculum:intent",
        },
        {
            "system": SYSTEM,
            "messages": [
                {"role": "user", "content": "Send it."},
                {
                    "role": "assistant",
                    "content": "What should I send, and to whom? I do not send anything without the exact text and an explicit approval.",
                },
                {"role": "user", "content": "the draft invoice to the client"},
                {
                    "role": "assistant",
                    "content": "Understood: the draft invoice to the client. I will show the recipient and body for approval first.",
                },
            ],
            "source": "claireai-curriculum:intent",
        },
        {
            "system": SYSTEM,
            "messages": [
                {"role": "user", "content": "What is the latest on that thing we discussed?"},
                {
                    "role": "assistant",
                    "content": "I do not know which topic you mean. Is it the news story from earlier, or a task in this conversation?",
                },
            ],
            "source": "claireai-curriculum:intent",
        },
    ]


def _intent() -> list[dict[str, Any]]:
    """Explicit request-shape examples, kept separate from small talk."""
    examples = [
        ("Find the biggest files in Downloads.", "list_files"),
        ("What is inside my Documents folder?", "list_files"),
        ("Move report.pdf from Downloads to Documents.", "move_file"),
        ("Rename the draft to final.", "rename_file"),
        ("Put this file in the bin; I do not need it any more.", "trash_file"),
        ("Add lunch with Ada tomorrow at 13:00.", "add_appointment"),
        ("What is on my calendar tomorrow?", "list_appointments"),
        ("Summarise what we did today.", "summarize"),
        ("Explain what a tokenizer does.", "chat"),
        ("I need help, but I have not decided what yet.", "clarify"),
    ]
    return [
        _record(
            "intent",
            "Route this request to one intent. Allowed intents: chat, list_files, "
            "move_file, rename_file, trash_file, add_appointment, list_appointments, "
            "summarize, clarify. Reply with the single intent word and nothing else.\n"
            f"Request: {request}",
            intent,
        )
        for request, intent in examples
    ]


def _hospitality() -> list[dict[str, Any]]:
    return [
        _record(
            "hospitality",
            "A room is 85 per night. What is the price for 3 nights with 10% tax?",
            "Rooms: 85 × 3 = 255. Tax: 25.50. Total: 280.50.",
        ),
        _record(
            "hospitality",
            "Write a short welcome message for a guest arriving late.",
            "Welcome! Your room is ready. Reception is open until 23:00; if you arrive "
            "later, call the number on your booking and we will arrange key collection.",
        ),
        _record(
            "hospitality",
            "A guest says the room is too noisy. What is a good response?",
            "Apologise, acknowledge the problem, offer a concrete option (a different room "
            "or a quiet-hours commitment), and confirm the next step before they leave.",
        ),
        _record(
            "hospitality",
            "How should a complaint be logged?",
            "Record the time, guest, room or booking, the exact issue, what was offered, "
            "what the guest accepted, and who will follow up.",
        ),
    ]


def _routines() -> list[dict[str, Any]]:
    return [
        _record(
            "routines",
            "Plan a morning with a 09:00 meeting, 20 minutes of email and a 30-minute review.",
            "08:10 email (20 min); 08:30 review (30 min); 09:00 meeting. Leave a 10-minute buffer.",
        ),
        _record(
            "routines",
            "How do you keep a daily routine realistic?",
            "Pick three must-do items, give each a time box, leave buffer between blocks, "
            "and move unfinished work explicitly instead of silently dropping it.",
        ),
        _record(
            "routines",
            "What belongs in an end-of-day shutdown routine?",
            "Close open loops, note the next action for each, save work, check tomorrow's "
            "first appointment, and stop rather than starting a large new task.",
        ),
    ]


def _email() -> list[dict[str, Any]]:
    return [
        _record(
            "email",
            "Draft a short email asking to move a meeting to Thursday.",
            "Subject: Moving our meeting to Thursday\n\nHello,\n\nCould we move our "
            "meeting to Thursday? I can make 10:00 or 14:00. If neither works, please "
            "send a time that suits you.\n\nThank you,\n[Your name]",
        ),
        _record(
            "email",
            "Draft a polite follow-up when no reply has arrived.",
            "Subject: Following up\n\nHello,\n\nI wanted to check whether you had a "
            "chance to look at my previous message. A one-line reply is enough.\n\nThank you,\n[Your name]",
        ),
        _record(
            "email",
            "How can an email be clearer?",
            "State the ask in the first two lines, use one topic per message, give a "
            "specific deadline, and make the next action obvious.",
        ),
    ]


def _social() -> list[dict[str, Any]]:
    return [
        _record(
            "social",
            "Draft a short post announcing that a shop opens on Saturday.",
            "We open this Saturday. Come by, say hello, and see what we have prepared. "
            "Opening hours: 09:00–18:00.",
        ),
        _record(
            "social",
            "Rewrite this for social media: We have a new product and it is good.",
            "New arrival: a product we are genuinely proud of. Here is what it does and "
            "who it is for — tell us what you would like to see next.",
        ),
        _record(
            "social",
            "What should be checked before posting for a business?",
            "Facts, prices and dates; permission to use any image or name; tone; spelling; "
            "whether the claim can be supported; and who is accountable for replies.",
        ),
        _record(
            "social",
            "Should an assistant post without approval?",
            "No. Drafting is safe; publishing to an account needs the owner's explicit "
            "approval and the platform's authorised API. Credentials are never guessed.",
        ),
    ]


def _research() -> list[dict[str, Any]]:
    return [
        _record(
            "research",
            "How do you judge whether a source is reliable?",
            "Check who wrote it, whether the claim is evidenced, the date, whether it is "
            "primary or repeating another source, and whether independent sources agree.",
        ),
        _record(
            "research",
            "How should a research answer handle uncertainty?",
            "Separate what is known from what is inferred, date the evidence, and say what "
            "would change the conclusion instead of inventing confidence.",
        ),
        _record(
            "research",
            "What is a useful way to compare two options?",
            "Define the criteria first, score each option against the same criteria, note "
            "the evidence for each score, and state the trade-off rather than only a winner.",
        ),
    ]


def _business() -> list[dict[str, Any]]:
    return [
        _record(
            "business",
            "A product sells for 50 and costs 30. What is the profit margin?",
            "Profit = 20. Margin = 20 / 50 = 40%.",
        ),
        _record(
            "business",
            "Fixed costs are 900, price is 30 and variable cost is 12. What is break-even?",
            "Contribution per unit = 18. Break-even = 900 / 18 = 50 units.",
        ),
        _record(
            "business",
            "1000 people saw an offer and 25 bought. What is the conversion rate?",
            "25 / 1000 = 2.5%.",
        ),
        _record(
            "business",
            "How do you improve a weak sales follow-up?",
            "Contact people while the need is fresh, reference their specific problem, "
            "offer one clear next step, and track replies instead of sending more volume.",
        ),
    ]


BUILDERS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "conversation": _conversation,
    "intent": _intent,
    "hospitality": _hospitality,
    "reports": _reports,
    "accounting": _accounting,
    "arithmetic": _arithmetic,
    "calculations": _calculations,
    "coding": _coding,
    "routines": _routines,
    "email": _email,
    "social": _social,
    "research": _research,
    "business": _business,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def online() -> bool:
    """A connectivity check that sends no application data."""
    try:
        with socket.create_connection(ONLINE_HOST, timeout=3):
            return True
    except OSError:
        return False


class TrainingError(Exception):
    """The curriculum or a training run could not be completed."""


class TrainingCurriculum:
    """Authored practice across the domains, plus real journal exchanges."""

    def __init__(self, data_dir: Path, journal: LearningJournal | None = None) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.directory = self.data_dir / "learning" / "curriculum"
        self.journal = journal or LearningJournal(self.data_dir)

    def examples(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for domain, builder in BUILDERS.items():
            records.extend(builder())
        try:
            for entry in self.journal.entries()[-200:]:
                sources = [
                    str(source)
                    for source in entry.get("sources") or []
                    if str(source).startswith(("http://", "https://"))
                ]
                if entry.get("verified") is not True or not sources:
                    # An ordinary model answer is not ground truth. It stays in
                    # the journal but never becomes a training target.
                    continue
                answer = str(entry.get("answer", ""))
                records.append(
                    _record(
                        "research",
                        str(entry.get("question", "")),
                        f"{answer}\n\nSource: {sources[0]}",
                    )
                )
        except (LearningError, OSError):
            pass
        # Deterministic split: every seventh record is validation.
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            question = record["messages"][0]["content"]
            if question and question not in seen:
                seen.add(question)
                unique.append(record)
        return unique

    def build(self) -> dict[str, Any]:
        records = self.examples()
        if len(records) < 20:
            raise TrainingError("not enough curriculum records to train on")
        self.directory.mkdir(parents=True, exist_ok=True)
        train = [record for index, record in enumerate(records) if index % 7]
        validation = [record for index, record in enumerate(records) if index % 7 == 0]
        _write_jsonl(self.directory / "train.jsonl", train)
        _write_jsonl(self.directory / "validation.jsonl", validation)
        counts: dict[str, int] = {}
        for record in records:
            domain = str(record.get("source", "")).split(":")[-1]
            counts[domain] = counts.get(domain, 0) + 1
        return {
            "total": len(records),
            "train": len(train),
            "validation": len(validation),
            "domains": counts,
            "path": str(self.directory),
        }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record_key(record: dict[str, Any]) -> str:
    payload = json.dumps(record.get("messages") or [], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise TrainingError(f"{path} line {number} is not valid JSON") from exc
        if isinstance(record, dict) and record.get("messages"):
            records.append(record)
    return records


def merge_training_corpus(
    curriculum_dir: Path,
    external_dir: Path,
    output_dir: Path,
) -> Path:
    """Merge authored/confirmed records into the external corpus, exactly once.

    The nightly worker used to choose the external corpus and silently drop the
    authored intent examples and the owner-confirmed lessons. This writes a
    combined corpus under the learning directory and rejects cross-split
    duplicates, so those records are actually trained on without contaminating
    validation.
    """
    authored_train = _read_jsonl(curriculum_dir / "train.jsonl")
    authored_validation = _read_jsonl(curriculum_dir / "validation.jsonl")
    external_train = _read_jsonl(external_dir / "train.jsonl")
    external_validation = _read_jsonl(external_dir / "validation.jsonl")

    train: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in [*external_train, *authored_train]:
        key = _record_key(record)
        if key not in seen:
            seen.add(key)
            train.append(record)
    validation: list[dict[str, Any]] = []
    for record in [*external_validation, *authored_validation]:
        key = _record_key(record)
        if key not in seen:
            seen.add(key)
            validation.append(record)
    if not train or not validation:
        raise TrainingError("the merged corpus has an empty train or validation split")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "validation.jsonl", validation)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "authored_train": len(authored_train),
                "external_train": len(external_train),
                "train": len(train),
                "validation": len(validation),
                "deduplicated": len(external_train)
                + len(authored_train)
                + len(external_validation)
                + len(authored_validation)
                - len(train)
                - len(validation),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


class TrainingWorker(threading.Thread):
    """Run one bounded local training job per day while the flag is on."""

    def __init__(
        self,
        data_dir: Path,
        *,
        enabled: Callable[[], bool],
        steps: int | None = None,
        run_interval_hours: int = DEFAULT_RUN_INTERVAL_HOURS,
        tick: float = TICK_SECONDS,
        online_check: Callable[[], bool] = online,
        runner: Callable[[list[str], Path, int], subprocess.CompletedProcess] | None = None,
        gate: Callable[[Path], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(name="claireai-training", daemon=True)
        self.data_dir = Path(data_dir).expanduser()
        self.enabled = enabled
        requested = steps if steps is not None else os.environ.get("CLAIREAI_TRAINING_STEPS")
        try:
            self.steps = max(20, min(MAX_STEPS, int(requested or DEFAULT_STEPS)))
        except (TypeError, ValueError):
            self.steps = DEFAULT_STEPS
        self.run_interval = timedelta(hours=max(1, run_interval_hours))
        self.tick_seconds = max(5.0, float(tick))
        self.online_check = online_check
        self.runner = runner
        self.gate = gate or CandidateGate(self.data_dir).review
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.state_path = self.data_dir / "learning" / "training.json"
        self.log_path = self.data_dir / "learning" / "training.log"
        self.last_error: str | None = None
        self.candidate: str | None = None
        self._state = self._read_state()

    # -- state -----------------------------------------------------------

    def _read_state(self) -> dict[str, Any]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def status(self) -> dict[str, Any]:
        state = dict(self._state)
        state.update(
            {
                "enabled": bool(self.enabled()),
                "steps": self.steps,
                "candidate": self.candidate or state.get("candidate"),
                "worker_error": self.last_error,
            }
        )
        return state

    def _write_state(self, **values: Any) -> None:
        self._state.update(values)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, self.state_path)

    def due(self) -> bool:
        last = self._state.get("last_attempt")
        if not isinstance(last, str):
            return True
        try:
            attempted = datetime.fromisoformat(last)
        except ValueError:
            return True
        return _now() - attempted >= self.run_interval

    # -- training --------------------------------------------------------

    def _review(
        self,
        candidate: Path,
        *,
        records: int | None = None,
        domains: dict[str, int] | None = None,
    ) -> bool:
        """Apply the gate, preserving the candidate when review is unavailable."""
        self.candidate = str(candidate)
        try:
            review = self.gate(candidate)
        except Exception as error:  # noqa: BLE001 - a broken harness must not kill the worker
            self._write_state(
                status="candidate needs review",
                candidate=str(candidate),
                records=records if records is not None else self._state.get("records"),
                domains=domains if domains is not None else self._state.get("domains"),
                error=None,
                gate={
                    "passed": None,
                    "promoted": False,
                    "reason": f"review unavailable: {error}",
                },
            )
            return True
        promoted = bool(review.get("promoted"))
        values: dict[str, Any] = {
            "status": "installed" if promoted else "candidate rejected",
            "candidate": str(candidate),
            "error": None,
            "gate": review,
        }
        if records is not None:
            values["records"] = records
        if domains is not None:
            values["domains"] = domains
        if records is not None or self._state.get("records") is not None:
            values["last_success"] = _now().isoformat(timespec="seconds")
        self._write_state(**values)
        return True

    def _retry_pending_review(self) -> bool:
        """Retry a candidate that could not be reviewed, without retraining."""
        if self._state.get("status") != "candidate needs review":
            return False
        candidate_value = self._state.get("candidate")
        if not candidate_value:
            return False
        candidate = Path(str(candidate_value))
        if not candidate.is_file():
            return False
        if self._state.get("status") != "candidate needs review":
            return False
        # The gate is local: a missing network must not strand a candidate.
        # Do not hold the status lock while the evaluation runs; the desktop
        # polls status() and would freeze for the whole review.
        return self._review(candidate)

    def tick(self) -> bool:
        """One attempt; returns whether training or a retried review happened."""
        if not self.enabled():
            return False
        if self._retry_pending_review():
            return True
        if not self.due():
            return False
        if not self.online_check():
            self.last_error = None
            self._write_state(status="offline")
            return False
        with self._lock:
            if not self.due():
                return False
            self._write_state(status="training", last_attempt=_now().isoformat(timespec="seconds"))
            try:
                curriculum = TrainingCurriculum(self.data_dir).build()
                candidate = self._train(curriculum)
            except (TrainingError, OSError, subprocess.SubprocessError) as error:
                self.last_error = str(error)
                self._write_state(status="error", error=str(error))
                return False
            self.last_error = None
            return self._review(candidate, records=curriculum["total"], domains=curriculum["domains"])

    def _train(self, curriculum: dict[str, Any]) -> Path:
        try:
            from claireai.update import checkout_root

            source = checkout_root()
        except Exception:  # noqa: BLE001 - optional discovery
            source = None
        if source is None:
            raise TrainingError("no source checkout found; set CLAIREAI_SOURCE or run --update once")
        trainer = source.parent / "brain-lab" / "train_chat.py"
        if not trainer.is_file():
            raise TrainingError(f"the trainer is missing: {trainer}")

        # Prefer the merged licensed corpus when the checkout has one. chat-v15
        # is the bilingual build from train_claireai.py; the v13 names remain
        # valid for older checkouts. The compact token arrays in train_chat.py
        # removed the old 15k-record memory ceiling, so a larger corpus is no
        # longer rejected just because it is large. Whatever is chosen, the
        # authored intent examples and confirmed lessons are merged in first;
        # selecting the external corpus directly silently discarded both.
        corpus = Path(curriculum["path"])
        for name in ("chat-v15", "chat-v13-small", "chat-v13"):
            candidate_corpus = source.parent / "brain-lab" / "data" / name
            if (candidate_corpus / "train.jsonl").is_file() and (candidate_corpus / "validation.jsonl").is_file():
                corpus = merge_training_corpus(
                    Path(curriculum["path"]),
                    candidate_corpus,
                    self.data_dir / "learning" / "curriculum-merged",
                )
                break

        stamp = _now().strftime("%Y%m%dT%H%M%SZ")
        output_dir = self.data_dir / "models" / "candidates"
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate = output_dir / f"claire-curriculum-{stamp}.pt"
        command = [
            sys.executable,
            str(trainer),
            "--corpus",
            str(corpus),
            "--steps",
            str(self.steps),
            "--output",
            str(candidate),
            "--name",
            f"claire-curriculum-{stamp}",
        ]
        # Continue from the installed weights with the installed architecture.
        # Training a fresh 256-token model every night made every candidate
        # incomparable with the installed one; this keeps the comparison real.
        installed = self.data_dir / "models" / "claire-scratch.pt"
        sidecar = installed.with_suffix(".json")
        if installed.is_file() and sidecar.is_file():
            try:
                architecture = json.loads(sidecar.read_text(encoding="utf-8"))["architecture"]
                for flag, key in (
                    ("--context", "context"),
                    ("--vocab-size", "vocab_size"),
                    ("--width", "width"),
                    ("--layers", "layers"),
                    ("--heads", "heads"),
                ):
                    command += [flag, str(int(architecture[key]))]
                position = str(architecture.get("position_encoding") or "learned")
                command += ["--position-encoding", position]
                command += ["--init-from", str(installed)]
            except (KeyError, OSError, TypeError, ValueError):
                # A malformed sidecar must not stop the nightly run; the trainer
                # will fall back to its own architecture, and the gate will reject
                # a candidate that does not fit.
                pass
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== {stamp} ({self.steps} steps) ===\n")
            log.flush()
            if self.runner is not None:
                completed = self.runner(command, source.parent, 3600)
            else:
                completed = subprocess.run(
                    command,
                    cwd=str(source.parent),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                    check=False,
                )
        if completed.returncode != 0 or not candidate.is_file():
            raise TrainingError(
                f"training exited {completed.returncode}; see {self.log_path}"
            )
        return candidate

    def run(self) -> None:  # pragma: no cover - thread loop
        while not self._stop.wait(self.tick_seconds):
            self.tick()

    def stop(self) -> None:
        self._stop.set()


def run_service_tick(settings: Any) -> dict[str, Any]:
    """One headless attempt for the systemd/cron entry point."""
    from claireai.runtime_control import RuntimeControl

    control = RuntimeControl(settings.control_state_path).read()
    allowed = bool(control.get("self_improvement")) and not bool(control.get("shutdown"))
    worker = TrainingWorker(
        settings.resolved_data_dir,
        enabled=lambda: allowed,
    )
    ran = worker.tick()
    result = worker.status()
    result["ran"] = ran
    return result


__all__ = [
    "DOMAINS",
    "TrainingCurriculum",
    "TrainingError",
    "TrainingWorker",
    "merge_training_corpus",
    "online",
    "run_service_tick",
]
