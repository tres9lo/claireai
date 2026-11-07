"""Date phrases are resolved by code, with a fixed clock.

Every case passes ``today`` explicitly: a test that depends on the day it runs is
a test that fails on one particular Monday and nobody knows why.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from claireai.dates import answer_date_question, describe_day, parse_when, relative_label
from claireai.summary import SummaryError, parse_day

# 2026-11-06 is a Friday.
FRIDAY = date(2026, 11, 6)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("today", date(2026, 11, 6)),
        ("Today", date(2026, 11, 6)),
        ("now", date(2026, 11, 6)),
        ("yesterday", date(2026, 11, 5)),
        ("tomorrow", date(2026, 11, 7)),
        ("on friday", date(2026, 11, 6)),
        ("3 days ago", date(2026, 11, 3)),
        ("in 2 weeks", date(2026, 11, 20)),
        ("a week ago", date(2026, 10, 30)),
        ("last friday", date(2026, 10, 30)),
        ("next friday", date(2026, 11, 13)),
        ("monday", date(2026, 11, 2)),
        ("2026-11-03", date(2026, 11, 3)),
        ("2026-02-30", None),
        ("3 Nov 2026", date(2026, 11, 3)),
        ("Nov 3, 2026", date(2026, 11, 3)),
        ("6 Nov 2026", date(2026, 11, 6)),
        ("3 november", date(2026, 11, 3)),
    ],
)
def test_phrases_resolve_against_a_fixed_clock(phrase: str, expected: date | None) -> None:
    if expected is None:
        with pytest.raises(ValueError):
            parse_when(phrase, today=FRIDAY)
        return
    assert parse_when(phrase, today=FRIDAY) == expected


@pytest.mark.parametrize("phrase", ["", "   ", "someday", "next blursday", "6/11/2026", "yesterdays"])
def test_unreadable_phrases_are_refused_rather_than_guessed(phrase: str) -> None:
    with pytest.raises(ValueError):
        parse_when(phrase, today=FRIDAY)


def test_ambiguous_numeric_dates_are_refused() -> None:
    # Refusing is the point: 6/11 is 6 November in one country and 11 June in
    # another, and a summary dated to the wrong day is worse than a clear error.
    with pytest.raises(ValueError, match="not a date I understand"):
        parse_when("06/11/2026", today=FRIDAY)


def test_describe_day_reads_like_a_person_wrote_it() -> None:
    assert describe_day(FRIDAY) == "Friday, 6 November 2026"
    assert describe_day(date(2026, 1, 1)) == "Thursday, 1 January 2026"


def test_relative_labels() -> None:
    assert relative_label(FRIDAY, today=FRIDAY) == "today"
    assert relative_label(FRIDAY - timedelta(days=1), today=FRIDAY) == "yesterday"
    assert relative_label(FRIDAY + timedelta(days=1), today=FRIDAY) == "tomorrow"
    assert relative_label(date(2026, 10, 1), today=FRIDAY) is None


def _moment() -> datetime:
    return datetime(2026, 11, 6, 14, 32, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "question",
    [
        "what is the date",
        "what's the date today",
        "what date is it",
        "what day is it",
        "what's today's date",
        "hey claire, what is the date today please",
    ],
)
def test_date_questions_are_answered_from_the_clock(question: str) -> None:
    answer = answer_date_question(question, now=_moment())
    assert answer is not None
    assert "Friday, 6 November 2026" in answer


def test_a_time_question_reports_the_clock() -> None:
    moment = _moment()
    answer = answer_date_question("what time is it", now=moment)
    assert answer is not None
    # Wall-clock time in the host's zone, which is what an owner expects to read;
    # the assertion therefore converts, rather than pinning one timezone.
    assert moment.astimezone().strftime("%H:%M") in answer
    assert moment.astimezone().strftime("%Y") in answer


@pytest.mark.parametrize(
    "question",
    ["what time is it", "what's the time", "what time is it now", "what is the current time", "tell me the time"],
)
def test_every_time_phrasing_is_answered(question: str) -> None:
    assert answer_date_question(question, now=_moment()) is not None


@pytest.mark.parametrize(
    "question",
    [
        "hello",
        "what is the date of report.pdf",
        "list my Downloads folder",
        "summarise yesterday",
        "what is the biggest file",
    ],
)
def test_ordinary_requests_are_left_to_the_model(question: str) -> None:
    assert answer_date_question(question, now=_moment()) is None


def test_the_summary_accepts_the_same_phrases() -> None:
    day = parse_day("3 days ago")
    assert isinstance(day, date)
    with pytest.raises(SummaryError, match="not a date I understand"):
        parse_day("next blursday")
