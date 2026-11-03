from __future__ import annotations

import sqlite3

import pytest

from claireai.storage import InvalidTransition, TaskRepository


def test_create_starts_queued_and_is_retrievable(conn: sqlite3.Connection) -> None:
    tasks = TaskRepository(conn)
    task = tasks.create(request="hello", kind="chat")
    assert task.status == "queued"
    assert task.finished_at is None
    assert tasks.get(task.id) == task


def test_happy_path_transitions_set_finished_at(conn: sqlite3.Connection) -> None:
    tasks = TaskRepository(conn)
    task = tasks.create(request="hi")
    tasks.update_status(task.id, "running")
    done = tasks.update_status(task.id, "succeeded")
    assert done.status == "succeeded"
    assert done.finished_at is not None


def test_invalid_transition_is_rejected(conn: sqlite3.Connection) -> None:
    tasks = TaskRepository(conn)
    task = tasks.create(request="hi")
    tasks.update_status(task.id, "running")
    tasks.update_status(task.id, "succeeded")
    with pytest.raises(InvalidTransition):
        tasks.update_status(task.id, "running")


def test_unknown_status_is_rejected(conn: sqlite3.Connection) -> None:
    tasks = TaskRepository(conn)
    task = tasks.create(request="hi")
    with pytest.raises(InvalidTransition):
        tasks.update_status(task.id, "teleported")


def test_missing_task_raises_lookup_error(conn: sqlite3.Connection) -> None:
    with pytest.raises(LookupError):
        TaskRepository(conn).update_status("nope", "running")


def test_failure_records_error(conn: sqlite3.Connection) -> None:
    tasks = TaskRepository(conn)
    task = tasks.create(request="hi")
    tasks.update_status(task.id, "running")
    failed = tasks.update_status(task.id, "failed", error="model unavailable")
    assert failed.error == "model unavailable"
    assert failed.status == "failed"


def test_recent_is_newest_first(conn: sqlite3.Connection) -> None:
    tasks = TaskRepository(conn)
    first = tasks.create(request="one")
    second = tasks.create(request="two")
    ids = [task.id for task in tasks.recent()]
    assert ids.index(second.id) < ids.index(first.id)
