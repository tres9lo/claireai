"""Repositories for private conversation history and owner customization."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from claireai.config.settings import Settings
from claireai.storage.tasks import utc_now

MAX_DIRECTIVES = 20
MAX_DIRECTIVE_TITLE = 120
MAX_DIRECTIVE_CONTENT = 4000
MAX_CONVERSATION_MESSAGE = 64_000
EDITABLE_PREFERENCES = frozenset({
    "owner_name",
    "max_output_tokens",
    "agent_max_steps",
    "response_style",
    "temperature",
    "temperature_tools",
    "voice_enabled",
    "voice_pitch",
    "voice_rate",
    "allowed_folders",
    "tool_routing",
})


@dataclass(frozen=True)
class ConversationMessage:
    id: int
    conversation_id: str
    task_id: str | None
    sequence: int
    created_at: str
    role: str
    content: str


class ConversationRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, *, title: str = "New conversation") -> str:
        conversation_id = uuid.uuid4().hex
        now = utc_now()
        self.conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title[:MAX_DIRECTIVE_TITLE], now, now),
        )
        return conversation_id

    def exists(self, conversation_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone() is not None

    def add_message(
        self, conversation_id: str, *, role: str, content: str, task_id: str | None = None
    ) -> ConversationMessage:
        if role not in {"user", "assistant"}:
            raise ValueError("conversation role must be user or assistant")
        if not isinstance(content, str) or len(content) > MAX_CONVERSATION_MESSAGE:
            raise ValueError(f"conversation message exceeds {MAX_CONVERSATION_MESSAGE} characters")
        if not self.exists(conversation_id):
            raise ValueError("conversation does not exist")
        next_sequence = self.conn.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS n FROM conversation_messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()["n"]
        created_at = utc_now()
        cursor = self.conn.execute(
            "INSERT INTO conversation_messages "
            "(conversation_id, task_id, sequence, created_at, role, content) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, task_id, next_sequence, created_at, role, content),
        )
        self.conn.execute(
            "UPDATE conversations SET updated_at = ?, "
            "title = CASE WHEN title = 'New conversation' AND ? = 'user' "
            "THEN ? ELSE title END WHERE id = ?",
            (created_at, role, " ".join(content.split())[:MAX_DIRECTIVE_TITLE], conversation_id),
        )
        return ConversationMessage(
            id=int(cursor.lastrowid),
            conversation_id=conversation_id,
            task_id=task_id,
            sequence=next_sequence,
            created_at=created_at,
            role=role,
            content=content,
        )

    def messages(self, conversation_id: str, *, limit: int | None = None) -> list[ConversationMessage]:
        if not self.exists(conversation_id):
            raise ValueError("conversation does not exist")
        query = (
            "SELECT id, conversation_id, task_id, sequence, created_at, role, content "
            "FROM conversation_messages WHERE conversation_id = ? ORDER BY sequence"
        )
        params: tuple[Any, ...] = (conversation_id,)
        if limit is not None:
            query = (
                "SELECT * FROM (" + query + " DESC LIMIT ?) ORDER BY sequence"
            )
            params = (conversation_id, limit)
        return [ConversationMessage(**dict(row)) for row in self.conn.execute(query, params)]

    def recent(self, *, limit: int = 100) -> list[dict[str, str]]:
        rows = self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in rows]

    def delete(self, conversation_id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cursor.rowcount == 1


class DirectiveRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def list(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        query = (
            "SELECT id, created_at, updated_at, title, content, enabled, position "
            "FROM user_directives"
        )
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY position, created_at, id"
        return [
            {**dict(row), "enabled": bool(row["enabled"])}
            for row in self.conn.execute(query)
        ]

    def save(
        self,
        *,
        title: str,
        content: str,
        directive_id: str | None = None,
        enabled: bool = True,
    ) -> str:
        title, content = title.strip(), content.strip()
        if not title or len(title) > MAX_DIRECTIVE_TITLE:
            raise ValueError(f"directive title must contain 1–{MAX_DIRECTIVE_TITLE} characters")
        if not content or len(content) > MAX_DIRECTIVE_CONTENT:
            raise ValueError(f"directive text must contain 1–{MAX_DIRECTIVE_CONTENT} characters")
        if type(enabled) is not bool:
            raise ValueError("directive enabled must be a boolean")
        if directive_id is not None:
            cursor = self.conn.execute(
                "UPDATE user_directives SET updated_at = ?, title = ?, content = ?, enabled = ? "
                "WHERE id = ?",
                (utc_now(), title, content, int(enabled), directive_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("directive does not exist")
            return directive_id
        count = self.conn.execute("SELECT COUNT(*) AS n FROM user_directives").fetchone()["n"]
        if count >= MAX_DIRECTIVES:
            raise ValueError(f"at most {MAX_DIRECTIVES} directives can be stored")
        directive_id = uuid.uuid4().hex
        now = utc_now()
        self.conn.execute(
            "INSERT INTO user_directives (id, created_at, updated_at, title, content, enabled, position) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (directive_id, now, now, title, content, int(enabled), count),
        )
        return directive_id

    def delete(self, directive_id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM user_directives WHERE id = ?", (directive_id,))
        return cursor.rowcount == 1


class PreferenceRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get(self) -> dict[str, Any]:
        rows = self.conn.execute("SELECT key, value_json FROM user_preferences ORDER BY key")
        return {row["key"]: json.loads(row["value_json"]) for row in rows}

    def effective(self, settings: Settings) -> dict[str, Any]:
        values = {
            key: getattr(settings, key)
            for key in EDITABLE_PREFERENCES
        }
        values.update(self.get())
        return {
            key: [str(path) for path in value] if key == "allowed_folders" else value
            for key, value in values.items()
        }

    def update(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if not values or set(values) - EDITABLE_PREFERENCES:
            raise ValueError("provide one or more supported preference fields")
        try:
            # Reuse the runtime model's validation so UI values and environment
            # settings obey the same bounds and capability/folder normalization.
            validated = Settings(_env_file=None, **dict(values))
        except Exception as exc:
            raise ValueError(f"invalid preference: {exc}") from exc
        if "allowed_folders" in values and len(validated.allowed_folders) > 32:
            raise ValueError("at most 32 allowed folders can be stored")
        now = utc_now()
        for key in values:
            value = getattr(validated, key)
            if key == "allowed_folders":
                value = [str(path) for path in value]
            self.conn.execute(
                "INSERT INTO user_preferences (key, value_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, "
                "updated_at = excluded.updated_at",
                (key, json.dumps(value, ensure_ascii=False), now),
            )
        return self.get()


def apply_preferences(settings: Settings, conn: sqlite3.Connection) -> Settings:
    """Apply owner preferences saved in the local UI to one runtime turn."""
    preferences = PreferenceRepository(conn).get()
    return Settings(_env_file=None, **(settings.model_dump() | preferences))
