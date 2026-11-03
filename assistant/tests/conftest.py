"""Shared fixtures: a migrated throwaway database and isolated settings."""

from __future__ import annotations

import os
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
    )


@pytest.fixture()
def sqlite_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    return conn


@pytest.fixture()
def make_symlink():
    """Create a symlink, or skip the test where the OS refuses.

    Windows needs Developer Mode or an elevated process; a normal user session
    raises OSError (WinError 1314). Skipping with a reason is honest: the
    allow-list tests still run on Linux and on elevated CI runners, and a
    contributor on Windows sees why instead of a false failure.
    """

    def _make(link: Path, target: Path) -> None:
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(
                f"this OS refused to create a symlink ({exc}); "
                "on Windows enable Developer Mode or run pytest elevated"
            )

    return _make
