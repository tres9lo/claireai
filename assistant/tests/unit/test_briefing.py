"""The briefing is assembled from sources, not from the model."""

from __future__ import annotations

import sqlite3
from datetime import date

from claireai import briefing
from claireai.storage import TaskRepository, transaction
from claireai.storage.appointments import AppointmentRepository


def _pending(conn: sqlite3.Connection, request: str) -> str:
    tasks = TaskRepository(conn)
    with transaction(conn):
        task = tasks.create(request=request)
        tasks.update_status(task.id, "running")
    return task.id


def test_briefing_combines_news_calendar_and_pending_work(
    conn: sqlite3.Connection, monkeypatch
) -> None:
    day = date(2026, 11, 20)
    with transaction(conn):
        AppointmentRepository(conn).add(title="Dentist", day=day, starts_at="10:30")
    _pending(conn, "Send the quarterly report")
    seen: list[tuple[tuple[str, ...], int]] = []

    def fake_brief(countries, limit=3):  # noqa: ANN001, ANN202
        seen.append((tuple(countries), limit))
        return "- A cited headline — Publisher — https://example.test/story"

    monkeypatch.setattr(briefing, "brief", fake_brief)
    report = briefing.build_briefing(conn, countries=["Kenya", "France"], day=day)

    assert seen == [(("Kenya", "France"), 3)]
    assert "Dentist" in report
    assert "10:30" in report
    assert "Send the quarterly report" in report
    assert "https://example.test/story" in report
    assert "No model was involved." in report


def test_briefing_without_countries_does_not_invent_news(conn: sqlite3.Connection, monkeypatch) -> None:
    def explode(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("news must not be fetched when no country is selected")

    monkeypatch.setattr(briefing, "brief", explode)
    report = briefing.build_briefing(conn, countries=[], day=date(2026, 11, 20))
    assert "_No countries selected._" in report
    assert "_Nothing scheduled._" in report
    assert "_No pending tasks._" in report


def test_news_is_off_when_browser_access_is_off(conn: sqlite3.Connection, monkeypatch) -> None:
    def explode(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("news must not be fetched while browser access is off")

    monkeypatch.setattr(briefing, "brief", explode)
    report = briefing.build_briefing(
        conn, countries=["Kenya"], day=date(2026, 11, 20), news_enabled=False
    )
    assert "browser access is disabled" in report


def test_news_failure_is_shown_as_unavailable(conn: sqlite3.Connection, monkeypatch) -> None:
    def fail(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise briefing.NewsError("network unavailable")

    monkeypatch.setattr(briefing, "brief", fail)
    report = briefing.build_briefing(conn, countries=["Kenya"], day=date(2026, 11, 20))
    assert "News unavailable: network unavailable" in report


def test_the_briefing_scheduler_has_one_daily_job(conn: sqlite3.Connection) -> None:
    scheduler = briefing.build_scheduler(conn, countries=["Kenya"], when="07:15")
    jobs = scheduler.get_jobs()
    assert [job.id for job in jobs] == ["daily-briefing"]
    assert jobs[0].trigger.fields[5].expressions[0].first == 7
    assert jobs[0].trigger.fields[6].expressions[0].first == 15


def test_a_bad_briefing_time_is_refused(conn: sqlite3.Connection) -> None:
    for when in ("25:00", "07:99", "morning", "07", ""):
        try:
            briefing.build_scheduler(conn, countries=[], when=when)
        except ValueError:
            continue
        raise AssertionError(f"{when!r} should have been refused")
