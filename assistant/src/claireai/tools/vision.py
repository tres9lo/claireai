"""Tools that let the assistant see a picture or the screen.

``identify_image`` reads a file inside an allow-listed folder, so it is ``safe``
like the other read-only tools. ``see_screen`` captures whatever is on the
display, which is the owner's private desktop: it always needs approval.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from claireai import vision
from claireai.tools.base import ToolContext, ToolError, ToolResult

MAX_LABELS = 40


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IdentifyImageArgs(_Args):
    path: str = Field(description="Absolute path of the image file to look at.")
    labels: list[str] | None = Field(
        default=None,
        description=(
            "Optional candidate descriptions to choose between, e.g. "
            "['a photo of a cat', 'a photo of a dog']. Omit to use a general set."
        ),
    )
    top: int = Field(default=3, ge=1, le=10, description="How many matches to return.")


class SeeScreenArgs(_Args):
    labels: list[str] | None = Field(
        default=None,
        description="Optional candidate descriptions to choose between.",
    )
    top: int = Field(default=3, ge=1, le=10, description="How many matches to return.")


def _clean_labels(labels: list[str] | None) -> list[str] | None:
    if labels is None:
        return None
    cleaned = [label.strip() for label in labels if label and label.strip()]
    if not cleaned:
        raise ToolError("labels were provided but all of them were empty")
    if len(cleaned) > MAX_LABELS:
        raise ToolError(f"at most {MAX_LABELS} labels can be compared at once")
    return cleaned


def _format(matches: list[tuple[str, float]]) -> str:
    lines = [f"{confidence * 100:.0f}%  {label}" for label, confidence in matches]
    return "Most likely (ranked, local model, no network):\n" + "\n".join(lines)


def _rank(data: bytes, labels: list[str] | None, top: int) -> list[tuple[str, float]]:
    try:
        return vision.describe(data, labels, limit=top)
    except vision.VisionUnavailable as error:
        raise ToolError(str(error)) from error


def identify_image(context: ToolContext, arguments: IdentifyImageArgs) -> ToolResult:
    """Identify what is in an image file inside an allow-listed folder."""
    path = context.sandbox.resolve(arguments.path)
    if not path.is_file():
        raise ToolError(f"{path} is not a regular file")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ToolError(f"cannot read {path}: {error}") from error
    if size > vision.MAX_IMAGE_BYTES:
        raise ToolError(
            f"{path} is {size} bytes; the limit is {vision.MAX_IMAGE_BYTES} bytes"
        )
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ToolError(f"cannot read {path}: {error}") from error

    matches = _rank(data, _clean_labels(arguments.labels), arguments.top)
    return ToolResult(
        text=_format(matches),
        detail={
            "path": str(path),
            "bytes": len(data),
            "model": f"{vision.MODEL_NAME}/{vision.PRETRAINED}",
            "matches": [{"label": label, "confidence": round(confidence, 4)} for label, confidence in matches],
        },
    )


def see_screen(context: ToolContext, arguments: SeeScreenArgs) -> ToolResult:
    """Capture the screen and identify what is on it."""
    try:
        data = vision.capture_screen()
    except vision.VisionUnavailable as error:
        raise ToolError(str(error)) from error

    matches = _rank(data, _clean_labels(arguments.labels), arguments.top)
    return ToolResult(
        text=_format(matches),
        detail={
            "capture_bytes": len(data),
            "model": f"{vision.MODEL_NAME}/{vision.PRETRAINED}",
            "matches": [{"label": label, "confidence": round(confidence, 4)} for label, confidence in matches],
        },
    )
