"""The vision layer without a GPU, a download, or a real screen.

Model inference is measured separately by ``brain-lab/vision_accuracy.py``
against a labelled benchmark. These tests cover the parts that decide whether the
assistant is *allowed* to look and whether what it got back is a usable frame,
because a black rectangle silently described as a photo is the failure that
matters most here.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from claireai import vision
from claireai.tools.base import ToolError
from claireai.tools.vision import IdentifyImageArgs, _clean_labels, identify_image


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _black(width: int = 64, height: int = 48) -> bytes:
    return _png(Image.new("RGB", (width, height), (0, 0, 0)))


def _noise(width: int = 64, height: int = 48) -> bytes:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (x * 4 % 256, y * 5 % 256, (x + y) * 3 % 256)
    return _png(image)


# -- decoding and blank detection ------------------------------------------


def test_a_solid_frame_is_recognised_as_blank() -> None:
    assert vision.looks_blank(vision.open_image(_black())) is True
    assert vision.looks_blank(vision.open_image(_png(Image.new("RGB", (8, 8), (255, 255, 255))))) is True


def test_a_real_frame_is_not_treated_as_blank() -> None:
    assert vision.looks_blank(vision.open_image(_noise())) is False


def test_empty_bytes_are_refused() -> None:
    with pytest.raises(vision.VisionUnavailable, match="empty"):
        vision.open_image(b"")


def test_an_oversized_image_is_refused_before_decoding(monkeypatch) -> None:
    monkeypatch.setattr(vision, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(vision.VisionUnavailable, match="larger than"):
        vision.open_image(b"x" * 11)


def test_a_non_image_file_is_refused_clearly() -> None:
    with pytest.raises(vision.VisionUnavailable, match="not a readable image"):
        vision.open_image(b"this is a text file, not a picture")


# -- screen capture --------------------------------------------------------


def test_capture_refuses_when_no_tool_is_installed(monkeypatch) -> None:
    monkeypatch.setattr(vision.shutil, "which", lambda _name: None)

    with pytest.raises(vision.VisionUnavailable, match="no screen capture tool"):
        vision.capture_screen()


def _fake_capture(monkeypatch, payload: bytes, *, name: str = "grim") -> None:
    monkeypatch.setattr(vision.shutil, "which", lambda tool: f"/usr/bin/{tool}" if tool == name else None)

    def fake_run(command, **_kwargs):  # noqa: ANN001
        Path(command[-1]).write_bytes(payload)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(vision.subprocess, "run", fake_run)


def test_capture_returns_a_real_frame(monkeypatch) -> None:
    payload = _noise()
    _fake_capture(monkeypatch, payload)

    assert vision.capture_screen() == payload


def test_a_blank_capture_is_refused_rather_than_described(monkeypatch) -> None:
    """A blocked compositor returns black. Describing that as a screenshot would lie."""
    _fake_capture(monkeypatch, _black())

    with pytest.raises(vision.VisionUnavailable) as caught:
        vision.capture_screen()
    message = str(caught.value)
    assert "blank frame" in message
    assert "grim" in message, "the refusal must name which tool failed and why"


def test_a_capture_tool_that_writes_nothing_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(vision.shutil, "which", lambda tool: f"/usr/bin/{tool}" if tool == "grim" else None)
    monkeypatch.setattr(
        vision.subprocess, "run", lambda command, **kwargs: SimpleNamespace(returncode=1, stdout=b"", stderr=b"")
    )

    with pytest.raises(vision.VisionUnavailable, match="produced no file"):
        vision.capture_screen()


def test_capture_tools_are_looked_for_on_this_host() -> None:
    # Whatever is installed, the probe must not raise and must only name known tools.
    known = {name for name, _argv in vision.capture_commands()}
    assert set(vision.available_capture_tools()) <= known


# -- tool arguments --------------------------------------------------------


class _Sandbox:
    def __init__(self, root: Path) -> None:
        self.root = root

    def resolve(self, path: str) -> Path:
        return Path(path)


class _Context:
    def __init__(self, root: Path) -> None:
        self.sandbox = _Sandbox(root)


def test_labels_are_trimmed_and_empties_dropped() -> None:
    assert _clean_labels(["  a cat ", "", "a dog  "]) == ["a cat", "a dog"]


def test_a_label_list_of_nothing_is_refused() -> None:
    with pytest.raises(ToolError, match="all of them were empty"):
        _clean_labels(["   ", ""])


def test_too_many_labels_are_refused() -> None:
    with pytest.raises(ToolError, match="at most"):
        _clean_labels([f"label {index}" for index in range(41)])


def test_no_labels_means_use_the_default_set() -> None:
    assert _clean_labels(None) is None


def test_identifying_a_text_file_fails_as_an_image_not_a_crash(tmp_path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("just words", encoding="utf-8")

    with pytest.raises(ToolError, match="not a readable image"):
        identify_image(_Context(tmp_path), IdentifyImageArgs(path=str(document)))


def test_a_missing_path_is_refused(tmp_path) -> None:
    with pytest.raises(ToolError, match="not a regular file"):
        identify_image(_Context(tmp_path), IdentifyImageArgs(path=str(tmp_path / "nope.png")))


def test_the_default_label_set_covers_screens_and_photos() -> None:
    joined = " ".join(vision.DEFAULT_LABELS).lower()
    assert "screenshot" in joined
    assert "photo" in joined
    assert len(set(vision.DEFAULT_LABELS)) == len(vision.DEFAULT_LABELS), "labels must be distinct"
