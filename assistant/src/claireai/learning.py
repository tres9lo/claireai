"""Opt-in background learning: journal what happened, then compile it.

The runtime does not rewrite model weights while it is in use. What it can do
honestly is keep the raw material for the next training run:

* every exchange is appended to ``<data>/learning/journal.jsonl`` when the owner
  has turned Background learning on;
* a background worker compiles new journal lines into
  ``<data>/learning/corpus.jsonl`` in the ``{"system", "messages"}`` shape
  ``brain-lab`` already trains on;
* a compiled checkpoint still has to pass the install gate in
  ``brain-lab/configs/`` before it replaces the installed one.

Recording is idempotent per exchange (a hash of question and answer), bounded in
size, written with mode 0600 like the rest of the data directory, and does
nothing at all while the flag is off. Turning the flag off stops both recording
and compilation; nothing is deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

MAX_LESSON_CHARS = 4000
COMPILE_INTERVAL_SECONDS = 30.0

CORPUS_SYSTEM = (
    "You are ClaireAI, a local-first personal assistant running on the owner's computer. "
    "Answer briefly and concretely, use tools when offered, and never claim an action you "
    "did not take."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clip(text: str) -> str:
    return text if len(text) <= MAX_LESSON_CHARS else text[:MAX_LESSON_CHARS]


def _write_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)


class LearningError(Exception):
    """The journal or its compiled form cannot be read."""


class LearningJournal:
    """The private record of what the assistant was asked and answered."""

    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir).expanduser()
        self.directory = self.root / "learning"
        self.journal_path = self.directory / "journal.jsonl"
        self.corpus_path = self.directory / "corpus.jsonl"
        self.state_path = self.directory / "state.json"
        self._lock = threading.Lock()

    # -- reading ---------------------------------------------------------

    def entries(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for number, line in enumerate(self.journal_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError as exc:
                raise LearningError(f"learning journal line {number} is not valid JSON") from exc
            if not isinstance(entry, dict) or not entry.get("id"):
                raise LearningError(f"learning journal line {number} is not an exchange")
            entries.append(entry)
        return entries

    def _compiled_count(self) -> int:
        if not self.state_path.exists():
            return 0
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise LearningError("learning state is not valid JSON") from exc
        compiled = state.get("compiled", 0) if isinstance(state, dict) else 0
        return compiled if isinstance(compiled, int) and compiled >= 0 else 0

    def status(self) -> dict[str, Any]:
        entries = self.entries()
        compiled = min(self._compiled_count(), len(entries))
        return {
            "total": len(entries),
            "compiled": compiled,
            "pending": max(0, len(entries) - compiled),
            "last_at": entries[-1].get("at") if entries else None,
            "corpus": str(self.corpus_path) if self.corpus_path.exists() else None,
        }

    # -- writing ---------------------------------------------------------

    def record(
        self,
        question: str,
        answer: str,
        *,
        model: str | None = None,
        route: str | None = None,
    ) -> bool:
        """Append one exchange. Returns ``False`` for an empty or duplicate one."""
        asked = _clip(" ".join(str(question).split()))
        replied = _clip(str(answer).strip())
        if not asked or not replied:
            return False
        digest = hashlib.sha256(f"{asked}\0{replied}".encode("utf-8")).hexdigest()[:24]
        with self._lock:
            if any(entry.get("id") == digest for entry in self.entries()):
                return False
            _write_line(
                self.journal_path,
                {
                    "id": digest,
                    "at": _now(),
                    "question": asked,
                    "answer": replied,
                    "model": model or "claireai-runtime",
                    "route": route,
                },
            )
        return True

    def compile(self) -> int:
        """Turn new journal lines into corpus records. Returns how many were added."""
        with self._lock:
            entries = self.entries()
            start = min(self._compiled_count(), len(entries))
            added = 0
            for entry in entries[start:]:
                _write_line(
                    self.corpus_path,
                    {
                        "system": CORPUS_SYSTEM,
                        "messages": [
                            {"role": "user", "content": entry["question"]},
                            {"role": "assistant", "content": entry["answer"]},
                        ],
                        "source": "claireai-learning",
                        "recorded_at": entry.get("at"),
                        "model": entry.get("model"),
                    },
                )
                added += 1
            if added:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.state_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"compiled": start + added, "updated_at": _now()}, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, self.state_path)
            return added


class BackgroundLearner(threading.Thread):
    """A quiet thread that compiles new lessons while the app is open.

    ``enabled`` is asked every tick rather than captured once, so flipping the
    setting takes effect without restarting the app. Compilation is append-only
    and idempotent, which is what makes it safe to run on a timer.
    """

    def __init__(
        self,
        journal: LearningJournal,
        *,
        enabled: Callable[[], bool],
        interval: float = COMPILE_INTERVAL_SECONDS,
    ) -> None:
        super().__init__(name="claireai-learning", daemon=True)
        self.journal = journal
        self.enabled = enabled
        self.interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self.compiled = 0
        self.last_error: str | None = None

    def tick(self) -> int:
        """One cycle; separated from the loop so it can be tested directly."""
        if not self.enabled():
            return 0
        try:
            added = self.journal.compile()
        except (LearningError, OSError) as error:
            self.last_error = str(error)
            return 0
        self.last_error = None
        self.compiled += added
        return added

    def run(self) -> None:  # pragma: no cover - a thread loop
        while not self._stop.wait(self.interval):
            self.tick()

    def stop(self) -> None:
        self._stop.set()


__all__ = [
    "BackgroundLearner",
    "LearningError",
    "LearningJournal",
    "COMPILE_INTERVAL_SECONDS",
]
