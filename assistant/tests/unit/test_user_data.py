"""Persistent local conversations, directives, and UI-customized settings."""

from __future__ import annotations

import sqlite3
import os
from pathlib import Path

import pytest

from claireai.config import Settings
from claireai.storage import (
    ConversationRepository,
    DirectiveRepository,
    PreferenceRepository,
    TaskRepository,
    apply_preferences,
)
from claireai.storage.database import transaction


def test_conversation_messages_are_ordered_and_deleted_as_a_unit(conn: sqlite3.Connection) -> None:
    conversations = ConversationRepository(conn)
    with transaction(conn):
        conversation_id = conversations.create()
        task = TaskRepository(conn).create(request="redacted", kind="chat")
        conversations.add_message(conversation_id, role="user", content="private prompt", task_id=task.id)
        conversations.add_message(conversation_id, role="assistant", content="private answer", task_id=task.id)

    messages = conversations.messages(conversation_id)
    assert [(item.role, item.content) for item in messages] == [
        ("user", "private prompt"), ("assistant", "private answer")
    ]
    assert conversations.recent()[0]["title"] == "private prompt"
    assert conversations.delete(conversation_id) is True
    with pytest.raises(ValueError, match="conversation does not exist"):
        conversations.messages(conversation_id)


def test_directives_and_preferences_round_trip_and_validate(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    directives = DirectiveRepository(conn)
    with transaction(conn):
        directive_id = directives.save(
            title="Reports", content="Use a short summary and clear headings."
        )
    assert directives.list(enabled_only=True)[0]["id"] == directive_id
    with transaction(conn):
        directives.save(
            directive_id=directive_id,
            title="Reports",
            content="Use headings and list completed and blocked work.",
            enabled=False,
        )
    assert directives.list(enabled_only=True) == []

    preferences = PreferenceRepository(conn)
    with transaction(conn):
        saved = preferences.update({
            "owner_name": "Ange",
            "response_style": "detailed",
            "voice_enabled": True,
            "voice_pitch": 64,
            "voice_rate": 190,
            "allowed_folders": [str(tmp_path)],
        })
    effective = apply_preferences(Settings(_env_file=None, data_dir=tmp_path), conn)
    assert saved["owner_name"] == "Ange"
    assert effective.owner_name == "Ange"
    assert effective.response_style == "detailed"
    assert effective.voice_enabled is True
    assert effective.voice_pitch == 64 and effective.voice_rate == 190
    assert effective.allowed_folders == [tmp_path]

    with pytest.raises(ValueError, match="invalid preference"):
        preferences.update({"voice_pitch": 100})


def test_local_database_file_permissions_restrict_conversation_storage(database, tmp_path: Path) -> None:  # noqa: ANN001
    if os.name == "nt":
        pytest.skip("POSIX file mode check")
    path = database.path
    assert path.stat().st_mode & 0o777 == 0o600
