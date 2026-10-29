"""Redaction helpers for anything written to the task log or printed.

Audit rows must stay useful without becoming a secret store, so text is
scrubbed for common credential shapes and truncated before storage.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"
DEFAULT_MAX_LENGTH = 500

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # key=value / key: value credentials. The whole remainder of the line is
    # masked: over-redacting an audit line is cheaper than leaking a secret.
    re.compile(
        r"(?i)\b(api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|token|secret|"
        r"password|passwd|authorization|client[-_ ]?secret)\b\s*[:=]\s*[^\n]*"
    ),
    # Bearer / Basic auth headers
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    # Well-known token prefixes
    re.compile(r"\b(sk|sk-proj|ghp|gho|github_pat|hf|xox[baprs]|AIza)[-_][A-Za-z0-9_-]{8,}"),
)


def redact(text: str, *, max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """Return ``text`` with credentials masked and long content truncated."""
    if text is None:
        return ""
    cleaned = str(text)

    def _mask_label(match: re.Match[str]) -> str:
        return f"{match.group(1)}={REDACTED}"

    cleaned = _PATTERNS[0].sub(_mask_label, cleaned)
    cleaned = _PATTERNS[1].sub(lambda m: f"{m.group(1)} {REDACTED}", cleaned)
    cleaned = _PATTERNS[2].sub(REDACTED, cleaned)

    if max_length > 0 and len(cleaned) > max_length:
        dropped = len(cleaned) - max_length
        cleaned = f"{cleaned[:max_length]}…[truncated {dropped} chars]"
    return cleaned
