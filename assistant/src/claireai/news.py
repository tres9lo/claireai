"""Country-aware headlines from public RSS, with no model in the loop.

The assistant is allowed to say it could not fetch the news. It is never allowed
to invent a headline. Google News publishes per-country RSS editions for personal
feed-reader use; every item here keeps its publisher and link so the owner can
check the original.
"""

from __future__ import annotations

import html
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable

import httpx

MAX_COUNTRIES = 5
MAX_HEADLINES = 10
MAX_FEED_BYTES = 3_000_000
CACHE_SECONDS = 15 * 60
USER_AGENT = "ClaireAI/0.1 (personal local news reader)"


class NewsError(Exception):
    """A country is unsupported or the feed could not be read."""


# Country editions. The value is (Google News gl, hl). Only editions that Google
# actually publishes are listed; adding a country is a data change, not a code
# change. English editions are preferred where they exist, otherwise the local
# language is kept rather than pretending an English translation exists.
_COUNTRIES: dict[str, tuple[str, str]] = {
    "united states": ("US", "en-US"), "usa": ("US", "en-US"),
    "united kingdom": ("GB", "en-GB"), "uk": ("GB", "en-GB"), "britain": ("GB", "en-GB"),
    "canada": ("CA", "en-CA"), "australia": ("AU", "en-AU"), "new zealand": ("NZ", "en-NZ"),
    "ireland": ("IE", "en-IE"), "india": ("IN", "en-IN"), "singapore": ("SG", "en-SG"),
    "south africa": ("ZA", "en-ZA"), "nigeria": ("NG", "en-NG"), "ghana": ("GH", "en-GH"),
    "kenya": ("KE", "en-KE"), "uganda": ("UG", "en-UG"), "tanzania": ("TZ", "en-TZ"),
    "rwanda": ("RW", "en-RW"), "ethiopia": ("ET", "en-ET"), "zimbabwe": ("ZW", "en-ZW"),
    "zambia": ("ZM", "en-ZM"), "botswana": ("BW", "en-BW"), "namibia": ("NA", "en-NA"),
    "pakistan": ("PK", "en-PK"), "bangladesh": ("BD", "en-BD"), "sri lanka": ("LK", "en-LK"),
    "philippines": ("PH", "en-PH"), "malaysia": ("MY", "en-MY"),
    "france": ("FR", "fr"), "belgium": ("BE", "fr"), "switzerland": ("CH", "de"),
    "germany": ("DE", "de"), "austria": ("AT", "de"),
    "spain": ("ES", "es"), "mexico": ("MX", "es"), "argentina": ("AR", "es"),
    "colombia": ("CO", "es"), "chile": ("CL", "es"), "peru": ("PE", "es"),
    "brazil": ("BR", "pt-BR"), "portugal": ("PT", "pt-PT"),
    "italy": ("IT", "it"), "netherlands": ("NL", "nl"),
    "egypt": ("EG", "ar"), "saudi arabia": ("SA", "ar"), "united arab emirates": ("AE", "ar"),
    "qatar": ("QA", "ar"), "jordan": ("JO", "ar"), "lebanon": ("LB", "ar"),
    "morocco": ("MA", "fr"), "senegal": ("SN", "fr"), "ivory coast": ("CI", "fr"),
    "cameroon": ("CM", "fr"), "democratic republic of the congo": ("CD", "fr"),
    "russia": ("RU", "ru"), "turkey": ("TR", "tr"), "japan": ("JP", "ja"),
    "south korea": ("KR", "ko"), "china": ("CN", "zh-CN"), "taiwan": ("TW", "zh-TW"),
    "indonesia": ("ID", "id"), "thailand": ("TH", "th"), "vietnam": ("VN", "vi"),
}

_CANONICAL: dict[str, str] = {}
for _name, (_code, _hl) in _COUNTRIES.items():
    if _code not in _CANONICAL or len(_name) > len(_CANONICAL[_code]):
        _CANONICAL[_code] = _name
_CODES = {code.lower(): name for code, name in _CANONICAL.items()}
_CACHE: dict[tuple[str, int], tuple[float, list[dict[str, str]]]] = {}
_NEWS_WORD = re.compile(r"\b(news|headlines?|briefing|top stories|what(?:'s| is) happening)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Headline:
    title: str
    link: str
    publisher: str
    published: str


def supported_countries() -> tuple[str, ...]:
    """Canonical names accepted by :func:`brief`, suitable for an error message."""
    return tuple(sorted({name for name in _COUNTRIES if len(name.split()) > 1}))


def countries_in(text: str) -> list[str]:
    """Find supported country names or two-letter codes in a request."""
    lowered = " ".join(text.lower().split())
    found: list[tuple[int, str]] = []
    for alias in sorted(_COUNTRIES, key=len, reverse=True):
        match = re.search(rf"\b{re.escape(alias)}\b", lowered)
        if match:
            found.append((match.start(), _COUNTRIES[alias][0]))
    for code in re.findall(r"\b([A-Z]{2})\b", text):
        if code.lower() in _CODES:
            found.append((text.find(code), code.upper()))
    ordered: list[str] = []
    for _, code in sorted(found):
        name = _CANONICAL.get(code, "")
        if name and name not in ordered:
            ordered.append(name)
        if len(ordered) >= MAX_COUNTRIES:
            break
    return ordered


def is_news_request(text: str) -> bool:
    """True only when the request asks for news *and* names a supported country."""
    return bool(_NEWS_WORD.search(text) and countries_in(text))


def _feed_url(gl: str, hl: str) -> str:
    return f"https://news.google.com/rss?hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0]}"


def _fetch_feed(url: str) -> bytes:
    response = httpx.get(
        url,
        timeout=httpx.Timeout(12.0, connect=5.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    if response.status_code >= 400:
        raise NewsError(f"the news feed answered HTTP {response.status_code}")
    if len(response.content) > MAX_FEED_BYTES:
        raise NewsError("the news feed was larger than the safety limit")
    return response.content


def _published(value: str) -> str:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _parse(payload: bytes) -> list[Headline]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise NewsError("the news feed was not readable XML") from exc
    items: list[Headline] = []
    for item in root.findall(".//item"):
        title = html.unescape((item.findtext("title") or "").strip())
        link = (item.findtext("link") or "").strip()
        if not title or not link.startswith(("https://", "http://")):
            continue
        source = item.find("source")
        publisher = html.unescape((source.text if source is not None else "") or "").strip()
        publisher = publisher or "unknown publisher"
        # Google often appends " - Publisher" to the title; keep the headline,
        # not a duplicate of the source column.
        suffix = f" - {publisher}"
        if title.endswith(suffix):
            title = title[: -len(suffix)].rstrip()
        items.append(Headline(title, link, publisher, _published(item.findtext("pubDate") or "")))
    return items


def fetch_country(country: str, *, limit: int = 5) -> list[Headline]:
    """Return up to ``limit`` current headlines for one country."""
    key = " ".join(country.lower().split())
    edition = _COUNTRIES.get(key)
    if edition is None:
        code = _CODES.get(key)
        if code:
            edition = _COUNTRIES[code]
    if edition is None:
        raise NewsError(
            f"unsupported country {country!r}. Try one of: {', '.join(supported_countries()[:12])}, ..."
        )
    limit = max(1, min(MAX_HEADLINES, int(limit)))
    gl, hl = edition
    cache_key = (gl, limit)
    cached = _CACHE.get(cache_key)
    if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return list(cached[1])
    headlines = _parse(_fetch_feed(_feed_url(gl, hl)))[:limit]
    _CACHE[cache_key] = (time.monotonic(), headlines)
    return list(headlines)


def brief(countries: Iterable[str], *, limit: int = 5) -> str:
    """Render a cited headline brief for one or more countries."""
    names = [str(name).strip() for name in countries if str(name).strip()]
    if not names:
        raise NewsError("name at least one country")
    if len(names) > MAX_COUNTRIES:
        raise NewsError(f"at most {MAX_COUNTRIES} countries per brief")
    lines = [f"Headlines retrieved {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}."]
    failures: list[str] = []
    for name in names:
        try:
            headlines = fetch_country(name, limit=limit)
        except (NewsError, httpx.HTTPError) as exc:
            failures.append(f"{name}: {exc}")
            continue
        lines.append(f"\n{name.title()}")
        if not headlines:
            lines.append("  (the feed returned no headlines)")
            continue
        for index, item in enumerate(headlines, 1):
            when = f" ({item.published})" if item.published else ""
            lines.append(f"  {index}. {item.title} — {item.publisher}{when}\n     {item.link}")
    if failures:
        lines.append("\nCould not retrieve: " + "; ".join(failures))
    lines.append(
        "\nSource: Google News country editions (public RSS). Headlines are links, "
        "not verified facts; check the publisher before acting."
    )
    return "\n".join(lines)


def answer_news(text: str) -> str | None:
    """Answer a plain country-news request, or return ``None`` for another task."""
    if not is_news_request(text):
        return None
    return brief(countries_in(text))


__all__ = [
    "Headline",
    "NewsError",
    "answer_news",
    "brief",
    "countries_in",
    "fetch_country",
    "is_news_request",
    "supported_countries",
]
