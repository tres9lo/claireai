"""Arithmetic is answered by code, not sampled from the model."""

from __future__ import annotations

import pytest

from claireai.calculations import answer_calculation


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("what is 2 + 3?", "2 + 3 = 5"),
        ("calculate (12 * 3) / 4", "(12 * 3) / 4 = 9"),
        ("what's 20 percent of 250?", "20% of 250 = 50"),
        ("15% of 80", "15% of 80 = 12"),
        ("7.5 km to m", "7.5 km = 7500 m"),
        ("10 / 4", "10 / 4 = 2.5"),
        ("2^10", "2^10 = 1024"),
        ("what is -3 + 5", "-3 + 5 = 2"),
    ],
)
def test_plain_calculations_are_exact(question: str, expected: str) -> None:
    assert answer_calculation(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "I have 2 apples",
        "what is my name?",
        "2 + ",
        "1 / 0",
        "__import__('os').system('id')",
        "2026-11-20",
        "20/11/2026",
        "what is 5?",
        "",
    ],
)
def test_prose_and_unsafe_input_are_not_calculations(question: str) -> None:
    assert answer_calculation(question) is None
