from __future__ import annotations

from pathlib import Path

import pytest

from claireai.config import Settings
from claireai.permissions import Sandbox
from claireai.tools.base import ToolContext, ToolError
from claireai.tools.canvas import (
    CanvasArgs,
    CanvasMedia,
    CanvasSection,
    create_canvas,
    list_canvases,
)


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        sandbox=Sandbox.from_settings(Settings(_env_file=None, allowed_folders=[])),
        data_dir=tmp_path / "data",
    )


def test_canvas_is_a_self_contained_local_page(tmp_path: Path) -> None:
    result = create_canvas(
        _context(tmp_path),
        CanvasArgs(
            title="How a bike gear works",
            summary="A small mechanism with a big effect.",
            steps=["The chain turns the rear cog.", "The cog turns the wheel."],
            sections=[
                CanvasSection(
                    heading="Why it helps",
                    points=["Lower gear: easier climbing.", "Higher gear: faster flat riding."],
                )
            ],
            sources=["https://example.com/gears"],
        ),
    )
    page = Path(result.detail["path"])
    markup = page.read_text(encoding="utf-8")

    assert page.is_file()
    assert page.parent.name == "canvases"
    assert "How a bike gear works" in markup
    assert "The chain turns the rear cog." in markup
    assert "<script" not in markup.lower()
    assert "https://example.com/gears" in markup
    assert result.detail["title"] == "How a bike gear works"

    rows = list_canvases(tmp_path / "data")
    assert rows[0]["title"] == "How a bike gear works"
    assert rows[0]["path"] == page


def test_canvas_escapes_model_text_instead_of_executing_it(tmp_path: Path) -> None:
    result = create_canvas(
        _context(tmp_path),
        CanvasArgs(
            title="<script>alert(1)</script>",
            sections=[CanvasSection(heading="A & B", points=["<img src=x onerror=alert(1)>"])],
        ),
    )
    markup = Path(result.detail["path"]).read_text(encoding="utf-8")

    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup
    assert "<img" not in markup
    assert "&lt;img" in markup


def test_canvas_renders_news_video_links_and_embeds_only_known_providers(tmp_path: Path) -> None:
    result = create_canvas(
        _context(tmp_path),
        CanvasArgs(
            title="World briefing",
            media=[
                CanvasMedia(
                    kind="news",
                    title="A real headline",
                    url="https://news.example/story",
                    summary="The publisher's standfirst.",
                    publisher="Example News",
                ),
                CanvasMedia(
                    kind="video",
                    title="How it works",
                    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                ),
                CanvasMedia(
                    kind="video",
                    title="Do not embed",
                    url="javascript:alert(1)",
                ),
                CanvasMedia(
                    kind="link",
                    title="Full report",
                    url="https://example.org/report",
                ),
            ],
        ),
    )
    markup = Path(result.detail["path"]).read_text(encoding="utf-8")

    assert "news-card" in markup
    assert "https://news.example/story" in markup
    assert "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ" in markup
    assert "javascript:alert" not in markup
    assert "Full report" in markup
    assert "canvas file is local" in markup


def test_canvas_needs_content_and_honours_dry_run(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="section"):
        create_canvas(_context(tmp_path), CanvasArgs(title="Empty"))

    result = create_canvas(
        ToolContext(
            sandbox=Sandbox.from_settings(Settings(_env_file=None, allowed_folders=[])),
            data_dir=tmp_path / "data",
            dry_run=True,
        ),
        CanvasArgs(title="Preview", steps=["One step"]),
    )
    assert result.text.startswith("[dry run]")
    assert list_canvases(tmp_path / "data") == []
