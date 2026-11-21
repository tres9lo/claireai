"""Background learning records only when enabled, and compiles once."""

from __future__ import annotations

import json
import os

import pytest

from claireai.learning import (
    REPLAY_LIMIT,
    BackgroundLearner,
    LearningError,
    LearningJournal,
)


def test_recording_is_opt_in_at_the_call_site_and_idempotent(tmp_path) -> None:
    journal = LearningJournal(tmp_path / "data")

    assert journal.record("What is a tokenizer?", "A tokenizer splits text.") is True
    assert journal.record("What is a tokenizer?", "A tokenizer splits text.") is False
    assert journal.record("", "empty question") is False

    status = journal.status()
    assert status["total"] == 1
    assert status["pending"] == 1
    assert status["last_at"]


def test_compile_is_append_only_and_resumable(tmp_path) -> None:
    journal = LearningJournal(tmp_path / "data")
    assert journal.record_lesson(
        "one", "first source-backed answer", sources=["https://example.test/one"]
    )
    assert journal.record_lesson(
        "two", "second source-backed answer", sources=["https://example.test/two"]
    )

    assert journal.compile() == 2
    assert journal.compile() == 0
    assert journal.status()["compiled"] == 2
    assert journal.status()["scanned"] == 2
    assert journal.status()["pending"] == 0

    assert journal.record_lesson(
        "three", "third source-backed answer", sources=["https://example.test/three"]
    )
    assert journal.compile() == 1
    assert journal.status()["compiled"] == 3
    assert journal.status()["scanned"] == 3

    lines = journal.corpus_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert '"content":"one"' in lines[0]
    assert '"role":"user"' in lines[0]
    assert "Source: https://example.test/one" in lines[0]


def test_unverified_answers_never_become_training_data(tmp_path) -> None:
    journal = LearningJournal(tmp_path / "data")
    assert journal.record("what is a tokenizer?", "a guess from the model") is True
    assert journal.compile() == 0
    assert journal.status()["verified"] == 0
    assert journal.status()["scanned"] == 1
    assert not journal.corpus_path.exists()


def test_a_verified_lesson_needs_a_source(tmp_path) -> None:
    journal = LearningJournal(tmp_path / "data")
    with pytest.raises(LearningError, match=r"http\(s\) source"):
        journal.record("q", "a", verified=True)


def test_compile_does_not_trust_a_forged_verified_entry(tmp_path) -> None:
    journal = LearningJournal(tmp_path / "data")
    journal.directory.mkdir(parents=True)
    journal.journal_path.write_text(
        json.dumps(
            {
                "id": "forged",
                "question": "q",
                "answer": "a",
                "verified": True,
                "sources": ["file:///etc/passwd"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert journal.compile() == 0
    assert not journal.corpus_path.exists()
    assert journal.status()["scanned"] == 1


def test_compile_keeps_a_bounded_replay_of_older_confirmed_lessons(tmp_path) -> None:
    journal = LearningJournal(tmp_path / "data")
    for index in range(270):
        journal.record_lesson(
            f"fact {index}",
            f"answer {index}",
            sources=[f"https://example.test/{index}"],
        )

    assert journal.compile() == 270
    replay = journal.replay_records()
    assert len(replay) == REPLAY_LIMIT
    questions = [record["messages"][0]["content"] for record in replay]
    assert all(int(question.rsplit(" ", 1)[-1]) < 70 for question in questions)
    assert all(record["source"] == "claireai-confirmed-research-replay" for record in replay)
    assert journal.replay_path.is_file()
    assert len(journal.replay_path.read_text(encoding="utf-8").splitlines()) == REPLAY_LIMIT
    # The same journal and cursor produce the same replay slice.
    assert journal.replay_records() == replay


def test_replay_never_teaches_an_unverified_guess(tmp_path) -> None:
    journal = LearningJournal(tmp_path / "data")
    journal.record("an unverified guess", "a model answer")
    for index in range(201):
        journal.record_lesson(
            f"confirmed {index}",
            f"answer {index}",
            sources=[f"https://example.test/{index}"],
        )

    assert journal.compile() == 201
    replay = journal.replay_records()
    assert len(replay) == 1
    assert replay[0]["messages"][0]["content"] == "confirmed 0"


def test_the_worker_does_nothing_while_disabled(tmp_path) -> None:
    journal = LearningJournal(tmp_path / "data")
    journal.record("q", "a")
    enabled = False
    worker = BackgroundLearner(journal, enabled=lambda: enabled, interval=1.0)

    assert worker.tick() == 0
    assert journal.status()["compiled"] == 0

    enabled = True
    assert worker.tick() == 0
    assert journal.status()["scanned"] == 1


def test_the_journal_is_private_to_the_owner(tmp_path) -> None:
    journal = LearningJournal(tmp_path / "data")
    journal.record("q", "a")
    journal.compile()

    if os.name != "nt":
        assert journal.journal_path.stat().st_mode & 0o777 == 0o600
