"""Dates, resolved by code rather than guessed by the model.

The local model does not know today's date. Asked "what is the date", a model
either invents one or answers from training time, and both are wrong in a way the
owner cannot see. Everything here is deterministic: the assistant answers date
questions from the clock, and resolves "yesterday" before any model is involved.

Supported phrases (case-insensitive, an optional leading "on" is ignored):

- ``today``, ``now``, ``tonight``, ``yesterday``, ``tomorrow``
- ``3 days ago``, ``in 2 weeks``, ``a week ago``
- ``last friday``, ``next monday``, and a bare weekday (the most recent one)
- ``2026-11-06``, ``6 Nov 2026``, ``Nov 6 2026``, ``6 november``

Ambiguous numeric forms (``6/11/2026``) are refused rather than guessed: whether
that is 6 November or 11 June depends on the reader's country, and a wrong date
in a daily summary is worse than a clear refusal.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

WEEKDAYS = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DAY_MONTH_YEAR = re.compile(r"^(\d{1,2})\s+([a-z]+)\.?\s+(\d{4})$")
_MONTH_DAY_YEAR = re.compile(r"^([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})$")
_DAY_MONTH = re.compile(r"^(\d{1,2})\s+([a-z]+)\.?$")
_MONTH_DAY = re.compile(r"^([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?$")
_RELATIVE = re.compile(r"^(in\s+)?(\d+|a|an)\s+(day|days|week|weeks)\s*(ago|from now)?$")


def _normalise(text: str) -> str:
    value = " ".join(str(text or "").lower().replace(",", " ").split())
    value = value.strip(" .?!")
    if value.startswith("on "):
        value = value[3:]
    return value


def _build(year: int, month: int, day: int, *, source: str) -> date:
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"{source!r} is not a real date") from exc


def parse_when(text: str, *, today: date | None = None) -> date:
    """Resolve one date phrase. Raises ``ValueError`` rather than guessing."""
    now = today or datetime.now().date()
    value = _normalise(text)
    if not value:
        raise ValueError("no date given")

    if value in {"today", "now", "tonight", "this morning", "this evening"}:
        return now
    if value == "yesterday":
        return now - timedelta(days=1)
    if value == "tomorrow":
        return now + timedelta(days=1)

    if match := _ISO.match(value):
        return _build(int(match[1]), int(match[2]), int(match[3]), source=value)

    if match := _RELATIVE.match(value):
        amount = 1 if match[2] in {"a", "an"} else int(match[2])
        unit = timedelta(days=amount if match[3].startswith("day") else 7 * amount)
        # "in 2 weeks" looks forward; "3 days ago" and a bare "3 days" look back.
        forward = bool(match[1]) or match[4] == "from now"
        return now + unit if forward else now - unit

    words = value.split()
    if len(words) == 2 and words[0] in {"last", "next"} and words[1] in WEEKDAYS:
        target = WEEKDAYS[words[1]]
        delta = (now.weekday() - target) % 7
        if words[0] == "last":
            return now - timedelta(days=delta or 7)
        return now + timedelta(days=7 - delta if delta else 7)

    if value in WEEKDAYS:
        return now - timedelta(days=(now.weekday() - WEEKDAYS[value]) % 7)

    for pattern, day_first in ((_DAY_MONTH_YEAR, True), (_MONTH_DAY_YEAR, False)):
        if match := pattern.match(value):
            day = int(match[1]) if day_first else int(match[2])
            month_word = match[2] if day_first else match[1]
            year = int(match[3])
            if month_word not in MONTHS:
                raise ValueError(f"{value!r} names a month I do not know")
            return _build(year, MONTHS[month_word], day, source=value)

    for pattern, day_first in ((_DAY_MONTH, True), (_MONTH_DAY, False)):
        if match := pattern.match(value):
            day = int(match[1]) if day_first else int(match[2])
            month_word = match[2] if day_first else match[1]
            if month_word not in MONTHS:
                raise ValueError(f"{value!r} names a month I do not know")
            return date(now.year, MONTHS[month_word], day)

    raise ValueError(
        f"{value!r} is not a date I understand; try 2026-11-06, 'yesterday' or 'last friday'"
    )


def describe_day(day: date) -> str:
    """``Friday, 6 November 2026`` — the form a person reads back."""
    return f"{WEEKDAY_NAMES[day.weekday()]}, {day.day} {MONTH_NAMES[day.month - 1]} {day.year}"


def relative_label(day: date, *, today: date | None = None) -> str | None:
    """``today`` / ``yesterday`` / ``tomorrow``, or ``None``."""
    now = today or datetime.now().date()
    return {0: "today", 1: "yesterday", -1: "tomorrow"}.get((now - day).days)


_DATE_QUESTIONS = (
    re.compile(r"^(?:what(?:'s| is)? )?(?:the )?date(?: today| is it| is today)?$"),
    re.compile(r"^(?:what(?:'s| is)? )?today(?:'s)? date$"),
    re.compile(r"^what day (?:is it|is it today|is today)$"),
    re.compile(r"^which date is (?:it|today)$"),
    re.compile(r"^(?:what|which) day of the week is it$"),
)
_TIME_QUESTIONS = (
    re.compile(r"^(?:what(?:'s| is)? )?(?:the )?(?:current )?time(?: is it(?: now)?| now)?$"),
    re.compile(r"^tell me the time$"),
)
_POLITE_PREFIX = re.compile(r"^(?:hey|hi|hello|ok|okay|claire|claireai|please|so|and)[ ,]+")
_POLITE_SUFFIX = re.compile(r"[ ,]+(?:please|thanks|thank you)$")


def answer_date_question(text: str, *, now: datetime | None = None) -> str | None:
    """A direct answer to a date or time question, or ``None`` to let the model reply.

    Matching is deliberately exact after stripping politeness: a loose match would
    hijack questions a tool should answer ("what is the date of report.pdf").
    """
    value = _normalise(text)
    # "hey claire what is the date today please" carries two openings, not one.
    for _ in range(4):
        stripped = _POLITE_PREFIX.sub("", value)
        if stripped == value:
            break
        value = stripped
    value = _POLITE_SUFFIX.sub("", value)
    if not value:
        return None

    moment = now or datetime.now().astimezone()
    today = moment.date()
    if any(pattern.match(value) for pattern in _DATE_QUESTIONS):
        return f"Today is {describe_day(today)}."
    if any(pattern.match(value) for pattern in _TIME_QUESTIONS):
        clock = moment.astimezone().strftime("%H:%M")
        zone = moment.astimezone().tzname() or ""
        return f"It is {clock} {zone}".rstrip() + f" on {describe_day(today)}."
    return None


__all__ = [
    "MONTHS",
    "WEEKDAYS",
    "answer_date_question",
    "describe_day",
    "parse_when",
    "relative_label",
]
