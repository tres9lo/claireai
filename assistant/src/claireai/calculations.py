"""Deterministic arithmetic, percentages and common unit conversions.

A language model is not a calculator. This module handles the narrow cases where
the user is plainly asking for a calculation, so the answer is exact and does not
depend on sampling. It deliberately refuses prose that merely contains numbers;
everything else still goes to the model or the agent's tools.
"""

from __future__ import annotations

import ast
import re
from fractions import Fraction

MAX_INPUT = 300
MAX_EXPONENT = 1000
_PREFIX = re.compile(
    r"^(?:what\s+is|what's|whats|calculate|compute|evaluate|solve|how\s+much\s+is)\s+",
    re.IGNORECASE,
)
_EXPRESSION = re.compile(r"^[0-9\s.+\-*/()^]+$")
_PERCENT = re.compile(
    r"^([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)\s+of\s+([0-9]+(?:\.[0-9]+)?)$",
    re.IGNORECASE,
)
_CONVERSION = re.compile(
    r"^([0-9]+(?:\.[0-9]+)?)\s*([a-z]+)\s+(?:to|in|into)\s+([a-z]+)$",
    re.IGNORECASE,
)
_DATE = re.compile(r"^[0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{1,4}$")
_UNITS = {
    "mm": ("length", Fraction(1, 1000)),
    "millimeter": ("length", Fraction(1, 1000)),
    "millimeters": ("length", Fraction(1, 1000)),
    "cm": ("length", Fraction(1, 100)),
    "centimeter": ("length", Fraction(1, 100)),
    "centimeters": ("length", Fraction(1, 100)),
    "m": ("length", Fraction(1)),
    "meter": ("length", Fraction(1)),
    "meters": ("length", Fraction(1)),
    "km": ("length", Fraction(1000)),
    "kilometer": ("length", Fraction(1000)),
    "kilometers": ("length", Fraction(1000)),
    "in": ("length", Fraction(254, 10000)),
    "inch": ("length", Fraction(254, 10000)),
    "inches": ("length", Fraction(254, 10000)),
    "ft": ("length", Fraction(3048, 10000)),
    "foot": ("length", Fraction(3048, 10000)),
    "feet": ("length", Fraction(3048, 10000)),
    "mile": ("length", Fraction(1609344, 1000)),
    "miles": ("length", Fraction(1609344, 1000)),
    "mg": ("mass", Fraction(1, 1000000)),
    "milligram": ("mass", Fraction(1, 1000000)),
    "milligrams": ("mass", Fraction(1, 1000000)),
    "g": ("mass", Fraction(1, 1000)),
    "gram": ("mass", Fraction(1, 1000)),
    "grams": ("mass", Fraction(1, 1000)),
    "kg": ("mass", Fraction(1)),
    "kilogram": ("mass", Fraction(1)),
    "kilograms": ("mass", Fraction(1)),
    "lb": ("mass", Fraction(45359237, 100000000)),
    "lbs": ("mass", Fraction(45359237, 100000000)),
    "pound": ("mass", Fraction(45359237, 100000000)),
    "pounds": ("mass", Fraction(45359237, 100000000)),
    "ml": ("volume", Fraction(1, 1000)),
    "milliliter": ("volume", Fraction(1, 1000)),
    "milliliters": ("volume", Fraction(1, 1000)),
    "l": ("volume", Fraction(1)),
    "liter": ("volume", Fraction(1)),
    "liters": ("volume", Fraction(1)),
    "litre": ("volume", Fraction(1)),
    "litres": ("volume", Fraction(1)),
    "s": ("time", Fraction(1, 3600)),
    "sec": ("time", Fraction(1, 3600)),
    "second": ("time", Fraction(1, 3600)),
    "seconds": ("time", Fraction(1, 3600)),
    "min": ("time", Fraction(1, 60)),
    "minute": ("time", Fraction(1, 60)),
    "minutes": ("time", Fraction(1, 60)),
    "h": ("time", Fraction(1)),
    "hr": ("time", Fraction(1)),
    "hour": ("time", Fraction(1)),
    "hours": ("time", Fraction(1)),
    "day": ("time", Fraction(24)),
    "days": ("time", Fraction(24)),
    "week": ("time", Fraction(168)),
    "weeks": ("time", Fraction(168)),
}


class _Unsafe(ValueError):
    pass


class _Evaluator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> Fraction:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Fraction:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _Unsafe("not a number")
        if isinstance(node.value, float):
            return Fraction(str(node.value))
        return Fraction(node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Fraction:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise _Unsafe("unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> Fraction:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise _Unsafe("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise _Unsafe("division by zero")
            return Fraction(left // right)
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise _Unsafe("division by zero")
            return left % right
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > MAX_EXPONENT:
                raise _Unsafe("unsafe exponent")
            if left == 0 and right < 0:
                raise _Unsafe("division by zero")
            return left ** right.numerator
        raise _Unsafe("unsupported operator")

    def generic_visit(self, node: ast.AST) -> Fraction:
        raise _Unsafe(f"unsupported syntax: {type(node).__name__}")


def _evaluate(expression: str) -> tuple[Fraction, str]:
    if not _EXPRESSION.fullmatch(expression):
        raise _Unsafe("not a plain arithmetic expression")
    python_expression = expression.replace("^", "**")
    try:
        tree = ast.parse(python_expression, mode="eval")
    except SyntaxError as exc:
        raise _Unsafe("incomplete expression") from exc
    value = _Evaluator().visit(tree)
    if abs(value.numerator).bit_length() > 20_000 or value.denominator.bit_length() > 20_000:
        raise _Unsafe("result is too large")
    return value, expression.strip()


def _format(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text or "0"


def _answer_percent(match: re.Match[str]) -> str:
    percent = Fraction(match.group(1))
    whole = Fraction(match.group(2))
    result = percent / 100 * whole
    return f"{_format(percent)}% of {_format(whole)} = {_format(result)}"


def _answer_conversion(match: re.Match[str]) -> str | None:
    amount = Fraction(match.group(1))
    source = _UNITS.get(match.group(2).lower())
    target = _UNITS.get(match.group(3).lower())
    if source is None or target is None or source[0] != target[0]:
        return None
    result = amount * source[1] / target[1]
    return f"{_format(amount)} {match.group(2)} = {_format(result)} {match.group(3)}"


def answer_calculation(text: str) -> str | None:
    """Return a deterministic answer for a plain calculation, or ``None``."""
    if not isinstance(text, str):
        return None
    stripped = " ".join(text.strip().split())
    if not stripped or len(stripped) > MAX_INPUT:
        return None
    stripped = stripped.rstrip(" ?!.")
    stripped = _PREFIX.sub("", stripped).strip()
    if not stripped or _DATE.fullmatch(stripped):
        return None

    percent = _PERCENT.fullmatch(stripped)
    if percent:
        return _answer_percent(percent)
    conversion = _CONVERSION.fullmatch(stripped)
    if conversion:
        converted = _answer_conversion(conversion)
        if converted is not None:
            return converted

    if not any(operator in stripped for operator in "+-*/^"):
        return None
    try:
        value, expression = _evaluate(stripped)
    except _Unsafe:
        return None
    return f"{expression} = {_format(value)}"


__all__ = ["answer_calculation"]
