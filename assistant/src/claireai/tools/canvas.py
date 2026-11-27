"""Local visual canvases: self-contained HTML from structured model output.

The model supplies a title, a short lead, sections and optional steps; this
module owns the markup and styling. That keeps a small model from having to
write (or smuggle in) scripts, and makes every canvas a plain local file the
owner can open in the default browser. No network asset is loaded.
"""

from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, Field

from claireai.tools.base import ToolContext, ToolError, ToolResult

MAX_SECTIONS = 6
MAX_POINTS = 8
MAX_STEPS = 10


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CanvasSection(_Args):
    heading: str = Field(min_length=1, max_length=120)
    points: list[str] = Field(default_factory=list, max_length=MAX_POINTS)


class CanvasMedia(_Args):
    """One internet item shown on the canvas.

    ``kind=news`` is for a headline returned by ``world_news`` (with its real
    publisher URL), ``video`` embeds only a recognised provider, ``image``
    points at a public https image, and ``link`` is a source card. The canvas
    never fetches or executes remote scripts.
    """

    kind: Literal["news", "video", "image", "link"] = "link"
    title: str = Field(min_length=1, max_length=160)
    url: str = Field(min_length=1, max_length=2000)
    summary: str = Field(default="", max_length=500)
    publisher: str = Field(default="", max_length=120)


class ListCanvasesArgs(_Args):
    limit: int = Field(default=12, ge=1, le=50, description="How many recent canvases to list.")


class CanvasArgs(_Args):
    title: str = Field(min_length=1, max_length=120)
    summary: str = Field(default="", max_length=600)
    sections: list[CanvasSection] = Field(default_factory=list, max_length=MAX_SECTIONS)
    steps: list[str] = Field(default_factory=list, max_length=MAX_STEPS)
    sources: list[str] = Field(default_factory=list, max_length=6)
    media: list[CanvasMedia] = Field(default_factory=list, max_length=8)


def _clean(value: str, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _slug(title: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return value[:48] or "canvas"


def _link(value: str) -> str:
    text = html.escape(value)
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return (
            f'<a href="{html.escape(value, quote=True)}" '
            f'target="_blank" rel="noreferrer noopener">{text}</a>'
        )
    return text


def _safe_url(value: str) -> str | None:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return value.strip()


def _youtube_id(parsed: Any) -> str | None:
    host = parsed.netloc.lower().removeprefix("www.")
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/")):
            candidate = parsed.path.strip("/").split("/")[1]
        else:
            candidate = None
    else:
        candidate = None
    if candidate and re.fullmatch(r"[A-Za-z0-9_-]{6,20}", candidate):
        return candidate
    return None


def _video_embed(value: str) -> str | None:
    value = _safe_url(value) or ""
    if not value:
        return None
    parsed = urlparse(value)
    video_id = _youtube_id(parsed)
    if video_id:
        return (
            f"https://www.youtube-nocookie.com/embed/{video_id}"
            "?rel=0&modestbranding=1"
        )
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "vimeo.com" or host == "player.vimeo.com":
        parts = [part for part in parsed.path.split("/") if part.isdigit()]
        if parts:
            return f"https://player.vimeo.com/video/{parts[-1]}"
    if parsed.path.lower().endswith((".mp4", ".webm", ".ogg")):
        return value
    return None


def _media_card(item: CanvasMedia) -> str:
    url = _safe_url(item.url)
    if url is None:
        return ""
    title = html.escape(_clean(item.title, 160))
    summary = html.escape(_clean(item.summary, 500))
    publisher = html.escape(_clean(item.publisher, 120))
    safe_url = html.escape(url, quote=True)
    source = f'<small>{publisher}</small>' if publisher else ""
    summary_html = f'<p>{summary}</p>' if summary else ""
    if item.kind == "video":
        embed = _video_embed(url)
        if embed:
            if embed == url and url.lower().endswith((".mp4", ".webm", ".ogg")):
                player = (
                    f'<video controls preload="metadata" src="{safe_url}"></video>'
                )
            else:
                player = (
                    f'<iframe src="{html.escape(embed, quote=True)}" '
                    f'title="{title}" loading="lazy" allowfullscreen '
                    'allow="accelerometer; clipboard-write; encrypted-media; '
                    'gyroscope; picture-in-picture"></iframe>'
                )
            return (
                '<article class="media-card video-card">'
                f'<div class="embed">{player}</div>'
                f'<h3>{title}</h3>{summary_html}{source}</article>'
            )
        return (
            f'<a class="media-card video-card" href="{safe_url}" target="_blank" '
            f'rel="noreferrer noopener"><div class="play">▶</div>'
            f'<h3>{title}</h3>{summary_html}{source}</a>'
        )
    if item.kind == "image":
        return (
            f'<figure class="media-card image-card"><img src="{safe_url}" '
            f'alt="{title}" loading="lazy" referrerpolicy="no-referrer">'
            f'<figcaption><h3>{title}</h3>{summary_html}{source}</figcaption></figure>'
        )
    if item.kind == "news":
        return (
            '<article class="media-card news-card"><div class="tag">NEWS</div>'
            f'<h3><a href="{safe_url}" target="_blank" rel="noreferrer noopener">'
            f'{title}</a></h3>{summary_html}{source}</article>'
        )
    return (
        f'<a class="media-card link-card" href="{safe_url}" target="_blank" '
        f'rel="noreferrer noopener"><div class="tag">LINK</div>'
        f'<h3>{title}</h3>{summary_html}{source}</a>'
    )


def _render(arguments: CanvasArgs, *, created: str) -> str:
    title = html.escape(_clean(arguments.title, 120))
    summary = html.escape(_clean(arguments.summary, 600))
    sections = []
    for section in arguments.sections:
        points = "".join(
            f"<li>{html.escape(_clean(point, 400))}</li>"
            for point in section.points
            if str(point).strip()
        )
        sections.append(
            '<article class="card">'
            f"<h2>{html.escape(_clean(section.heading, 120))}</h2>"
            f"<ul>{points}</ul></article>"
        )
    steps = "".join(
        f'<div class="step"><span>{index}</span><p>{html.escape(_clean(step, 400))}</p></div>'
        for index, step in enumerate(arguments.steps, 1)
        if str(step).strip()
    )
    sources = "".join(
        f"<li>{_link(_clean(source, 500))}</li>"
        for source in arguments.sources
        if str(source).strip()
    )
    media = "".join(_media_card(item) for item in arguments.media)
    media_html = (
        f'<section class="media-grid">{media}</section>' if media else ""
    )
    steps_html = f'<section class="flow">{steps}</section>' if steps else ""
    sections_html = f'<section class="grid">{"".join(sections)}</section>' if sections else ""
    sources_html = (
        f'<section class="sources"><h2>Sources</h2><ul>{sources}</ul></section>'
        if sources
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · ClaireAI canvas</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #101417; --panel: #171d21; --line: #2b363d;
    --text: #eef3f5; --muted: #9aabb4; --accent: #68d8c4;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: radial-gradient(circle at 12% 0%, #1b2b2d, var(--bg) 45%);
    color: var(--text); font: 16px/1.55 "Segoe UI", system-ui, sans-serif;
  }}
  main {{ width: min(1080px, 92vw); margin: 0 auto; padding: 56px 0 72px; }}
  .kicker {{ color: var(--accent); font-size: 12px; letter-spacing: .14em; text-transform: uppercase; }}
  h1 {{ font-size: clamp(32px, 5vw, 58px); line-height: 1.05; margin: 12px 0 16px; max-width: 900px; }}
  .lead {{ color: var(--muted); font-size: 19px; max-width: 780px; margin: 0 0 32px; }}
  .flow {{ display: grid; gap: 10px; margin: 26px 0 34px; }}
  .step {{
    display: grid; grid-template-columns: 38px 1fr; gap: 14px; align-items: start;
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px;
  }}
  .step span {{
    display: grid; place-items: center; width: 30px; height: 30px; border-radius: 50%;
    background: var(--accent); color: #0d1917; font-weight: 800;
  }}
  .step p {{ margin: 3px 0 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
  .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 20px; }}
  .card h2, .sources h2 {{ font-size: 17px; margin: 0 0 10px; color: var(--accent); }}
  ul {{ margin: 0; padding-left: 20px; }}
  li {{ margin: 6px 0; }}
  .sources {{ margin-top: 30px; border-top: 1px solid var(--line); padding-top: 20px; }}
  .sources li {{ color: var(--muted); font-size: 14px; overflow-wrap: anywhere; }}
  .media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin: 26px 0 30px; }}
  .media-card {{ display: block; background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 18px; color: var(--text); text-decoration: none; overflow: hidden; }}
  .media-card h3 {{ margin: 8px 0 6px; font-size: 17px; line-height: 1.3; }}
  .media-card h3 a {{ text-decoration: none; }}
  .media-card p {{ color: var(--muted); font-size: 14px; margin: 0 0 8px; }}
  .media-card small {{ color: var(--muted); font-size: 11px; }}
  .tag {{ display: inline-block; color: var(--accent); font-size: 9px; font-weight: 800; letter-spacing: .14em; }}
  .embed {{ aspect-ratio: 16 / 9; background: #0a0e10; border-radius: 10px; overflow: hidden; }}
  .embed iframe, .embed video {{ width: 100%; height: 100%; border: 0; display: block; }}
  .image-card {{ padding: 0; }}
  .image-card img {{ width: 100%; height: 180px; object-fit: cover; display: block; }}
  .image-card figcaption {{ padding: 14px 18px 18px; }}
  .play {{ display: grid; place-items: center; height: 120px; border-radius: 10px; background: radial-gradient(circle, #23413d, #11181b); color: var(--accent); font-size: 30px; }}
  a {{ color: var(--accent); }}
  footer {{ margin-top: 40px; color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<main>
  <div class="kicker">ClaireAI canvas · made locally</div>
  <h1>{title}</h1>
  {f'<p class="lead">{summary}</p>' if summary else ''}
  {media_html}
  {steps_html}
  {sections_html}
  {sources_html}
  <footer>Created {html.escape(created)} on this PC. The canvas file is local; videos, images and article links load from their source when online.</footer>
</main>
</body>
</html>
"""


def _canvas_dir(context: ToolContext) -> Path:
    if context.data_dir is None:
        raise ToolError("canvas storage is not available in this run")
    folder = Path(context.data_dir) / "canvases"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def create_canvas(context: ToolContext, arguments: CanvasArgs) -> ToolResult:
    """Write one self-contained canvas and return its path."""
    if not arguments.sections and not arguments.steps and not arguments.media:
        raise ToolError("a canvas needs at least one section, step or media item")
    folder = _canvas_dir(context)
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    stamp = created_at.strftime("%Y%m%d-%H%M%S")
    token = f"{time.time_ns() % 1_000_000:06d}"
    name = f"{stamp}-{_slug(arguments.title)}-{token}"
    page = folder / f"{name}.html"
    meta = folder / f"{name}.json"
    created = created_at.strftime("%Y-%m-%d %H:%M UTC")
    if context.dry_run:
        return ToolResult(
            text=f"[dry run] would create canvas {arguments.title!r} at {page}",
            detail={"path": str(page), "title": arguments.title},
        )
    page.write_text(_render(arguments, created=created), encoding="utf-8")
    page.chmod(0o600)
    payload: dict[str, Any] = {
        "id": name,
        "title": arguments.title,
        "summary": arguments.summary,
        "created": created_at.isoformat(),
        "html": page.name,
        "sections": [
            {"heading": section.heading, "points": list(section.points)}
            for section in arguments.sections
        ],
        "steps": list(arguments.steps),
        "sources": list(arguments.sources),
        "media": [
            {
                "kind": item.kind,
                "title": item.title,
                "url": item.url,
                "summary": item.summary,
                "publisher": item.publisher,
            }
            for item in arguments.media
        ],
    }
    meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    meta.chmod(0o600)
    return ToolResult(
        text=(
            f"Created canvas {arguments.title!r}. Open it from the Canvas page "
            f"({page.name})."
        ),
        detail={"path": str(page), "title": arguments.title, "id": name},
    )


def list_canvases(data_dir: str | Path | None, *, limit: int = 30) -> list[dict[str, Any]]:
    """Read the local canvas index for the desktop page, newest first."""
    if data_dir is None:
        return []
    folder = Path(data_dir) / "canvases"
    if not folder.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for meta in folder.glob("*.json"):
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        page = folder / str(payload.get("html") or "")
        if not page.is_file() or page.suffix != ".html":
            continue
        rows.append(
            {
                "id": str(payload.get("id") or meta.stem),
                "title": str(payload.get("title") or "Untitled canvas"),
                "summary": str(payload.get("summary") or ""),
                "created": str(payload.get("created") or ""),
                "path": page,
            }
        )
    rows.sort(key=lambda row: row["path"].stat().st_mtime, reverse=True)
    return rows[:limit]


def list_canvases_tool(context: ToolContext, arguments: ListCanvasesArgs) -> ToolResult:
    """Read the local canvas index without creating anything."""
    rows = list_canvases(context.data_dir, limit=arguments.limit)
    if not rows:
        return ToolResult(text="No local canvases have been created yet.")
    lines = "\n".join(
        f"{row['title']} — {row['id']} ({row['created']})" for row in rows
    )
    return ToolResult(
        text=f"Local canvases ({len(rows)}):\n{lines}",
        detail={"canvases": [row["id"] for row in rows]},
    )
