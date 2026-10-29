"""Shared fixtures: a migrated throwaway database and isolated settings."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from claireai.config import Settings
from claireai.storage import Database


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "db" / "claireai.sqlite3"


@pytest.fixture()
def database(db_path: Path) -> Database:
    db = Database(db_path)
    db.migrate()
    return db


@pytest.fixture()
def conn(database: Database):
    connection = database.connect()
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    # _env_file=None keeps ambient .env files out of unit tests.
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="test-model",
    )


@pytest.fixture()
def sqlite_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    return conn
