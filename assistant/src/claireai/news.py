"""Country-aware headlines and summaries from public RSS, with no model.

The assistant is allowed to say it could not fetch the news. It is never allowed
to invent a headline or a summary. Bing's public news RSS carries a short
description and the real publisher URL, which is what makes a briefing more than
a list of links. For Rwanda the local outlets are read directly first: Google has
been redirecting this host's ``RW`` edition to the US edition, so a country code
alone was not a reliable promise.
"""

from __future__ import annotations

import html
import re
import socket
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from itertools import zip_longest
from typing import Iterable
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

MAX_COUNTRIES = 5
MAX_HEADLINES = 10
MAX_FEED_BYTES = 3_000_000
CACHE_SECONDS = 15 * 60
FETCH_ATTEMPTS = 3
RETRY_DELAYS = (0.4, 1.0)
USER_AGENT = "ClaireAI/0.1 (personal local news reader)"

# Direct publisher feeds. These are used before an index so a request for a
# country's news is answered by that country's newsroom, not by a world feed
# wearing the country's edition code.
_LOCAL_FEEDS: dict[str, tuple[str, ...]] = {
    "RW": (
        "https://www.ktpress.rw/feed/",
        "https://taarifa.rw/feed/",
    ),
}


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
    summary: str = ""


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


def _feed_url(gl: str, hl: str, name: str) -> str:
    """Bing's news RSS for one country.

    ``cc`` is a locale hint, while the query is what actually keeps the result
    set on-topic; Bing's RSS also supplies the descriptions that Google's
    country feed omits.
    """
    query = urlencode({"q": name, "format": "RSS", "cc": gl, "setlang": hl})
    return f"https://www.bing.com/news/search?{query}"


def _google_feed_url(gl: str, hl: str) -> str:
    """Fallback top-stories edition, used only when Bing cannot be read."""
    return f"https://news.google.com/rss?hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0]}"


def _real_link(link: str) -> str:
    """Unwrap Bing's ``apiclick`` redirect so the owner gets the publisher."""
    parsed = urlparse(link)
    if parsed.netloc.endswith("bing.com") and parsed.path.endswith("/apiclick.aspx"):
        inner = parse_qs(parsed.query).get("url", [""])[0]
        if inner.startswith(("https://", "http://")):
            return inner
    return link


def _plain_text(value: str) -> str:
    """Turn a feed's HTML description into one readable paragraph."""
    text = html.unescape(re.sub(r"(?is)<(script|style).*?</\1>", " ", value))
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    # WordPress feeds append a boilerplate footer after the real standfirst.
    text = re.sub(r"\s*The post .*? appeared first on .*?$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Continue reading.*?$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _dns_failure(error: BaseException) -> bool:
    """True when an exception chain is a resolver failure rather than an HTTP one."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, socket.gaierror):
            return True
        message = str(current).lower()
        if "name resolution" in message or "name or service not known" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _fetch_feed(url: str) -> bytes:
    """Fetch one feed, retrying transient DNS/timeout/5xx failures.

    A single resolver blip used to discard a country from the whole briefing.
    Three bounded attempts cost at most a couple of seconds and keep a partial
    outage from looking like an unsupported country. Permanent HTTP errors are
    not retried, and the final error names DNS explicitly when that was the
    cause.
    """
    last_error: BaseException | None = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            response = httpx.get(
                url,
                timeout=httpx.Timeout(12.0, connect=5.0),
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
        except (httpx.TransportError, OSError) as exc:
            last_error = exc
            if attempt + 1 < FETCH_ATTEMPTS:
                time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                continue
            break
        if response.status_code in {408, 429} or response.status_code >= 500:
            last_error = NewsError(
                f"the news feed answered HTTP {response.status_code}"
            )
            if attempt + 1 < FETCH_ATTEMPTS:
                time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                continue
            break
        if response.status_code >= 400:
            raise NewsError(f"the news feed answered HTTP {response.status_code}")
        if len(response.content) > MAX_FEED_BYTES:
            raise NewsError("the news feed was larger than the safety limit")
        return response.content

    if last_error is not None and _dns_failure(last_error):
        raise NewsError(
            "temporary DNS failure while resolving the news feed; check the "
            "network or DNS connection and try again"
        ) from last_error
    detail = str(last_error) if last_error is not None else "no response"
    raise NewsError(
        f"the news feed could not be reached after {FETCH_ATTEMPTS} attempts ({detail})"
    ) from last_error


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
        link = _real_link(link)
        # Google puts the publisher in <source>; Bing uses a namespaced
        # <News:Source>, so match the local name instead of one namespace.
        source = item.find("source")
        if source is None:
            source = item.find("{*}Source")
        publisher = html.unescape((source.text if source is not None else "") or "").strip()
        # Publisher feeds (KT Press, Taarifa) omit <source>; the site is still a
        # better citation than "unknown publisher".
        publisher = publisher or urlparse(link).netloc.removeprefix("www.") or "unknown publisher"
        # Google often appends " - Publisher" to the title; keep the headline,
        # not a duplicate of the source column.
        suffix = f" - {publisher}"
        if title.endswith(suffix):
            title = title[: -len(suffix)].rstrip()
        summary = _plain_text(item.findtext("description") or "")
        normalized_title = " ".join(title.lower().split())
        normalized_summary = " ".join(summary.lower().split())
        # Google's description is just the headline repeated inside a link list;
        # keeping that would look like a summary while adding nothing.
        if normalized_summary.startswith(normalized_title) and len(normalized_summary) <= len(normalized_title) + 100:
            summary = ""
        items.append(
            Headline(
                title,
                link,
                publisher,
                _published(item.findtext("pubDate") or ""),
                summary,
            )
        )
    return items


def _merge(items: Iterable[Headline], limit: int) -> list[Headline]:
    """Deduplicate the local and indexed copies of the same story."""
    merged: list[Headline] = []
    seen: set[str] = set()
    for item in items:
        key = " ".join(re.sub(r"[^a-z0-9 ]", " ", item.title.lower()).split())
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


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

    failures: list[str] = []
    local_batches: list[list[Headline]] = []
    for url in _LOCAL_FEEDS.get(gl, ()):
        try:
            batch = _parse(_fetch_feed(url))
        except (NewsError, httpx.HTTPError) as exc:
            failures.append(f"{urlparse(url).netloc}: {exc}")
            continue
        if batch:
            local_batches.append(batch)

    # Round-robin the publisher feeds, so a prolific outlet cannot fill every
    # slot while the second local newsroom is present but later in its own feed.
    local = [item for group in zip_longest(*local_batches) for item in group if item is not None]
    canonical = _CANONICAL.get(gl, country)
    country_words = {word for word in re.findall(r"[a-z]+", canonical.lower()) if len(word) > 2}

    def about_country(item: Headline) -> bool:
        haystack = f"{item.title} {item.summary}".lower()
        return bool(country_words) and country_words <= set(re.findall(r"[a-z]+", haystack))

    # The index adds summaries and fills the list when a publisher feed is
    # temporarily down. If it fails, use Google's edition only as a last resort
    # and only when no local story was available.
    indexed: list[Headline] = []
    try:
        indexed = _parse(_fetch_feed(_feed_url(gl, hl, canonical)))
    except (NewsError, httpx.HTTPError) as exc:
        failures.append(f"Bing News: {exc}")
        if not local and not _dns_failure(exc):
            try:
                indexed = _parse(_fetch_feed(_google_feed_url(gl, hl)))
            except (NewsError, httpx.HTTPError) as fallback_exc:
                failures.append(f"Google News: {fallback_exc}")

    # Keep the country the owner asked for in front: a local outlet's regional
    # story is still local, but it should not crowd out the country itself.
    focused = [item for item in local if about_country(item)]
    regional = [item for item in local if not about_country(item)]
    headlines = _merge([*focused, *indexed, *regional], limit)
    if not headlines:
        detail = "; ".join(failures) or "the feeds returned no headlines"
        raise NewsError(f"could not retrieve news for {country}: {detail}")
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
            failures.append(f"{name.title()}: {exc}")
            continue
        lines.append(f"\n{name.title()}")
        if not headlines:
            lines.append("  (the feed returned no headlines)")
            continue
        for index, item in enumerate(headlines, 1):
            when = f" ({item.published})" if item.published else ""
            lines.append(f"  {index}. {item.title} — {item.publisher}{when}")
            if item.summary:
                lines.append(f"     {item.summary}")
            lines.append(f"     {item.link}")
    if failures:
        lines.append("\nCould not retrieve: " + "; ".join(failures))
    lines.append(
        "\nSource: public RSS news feeds (Bing News index; KT Press and Taarifa "
        "are read directly for Rwanda). Summaries are the publisher's standfirst, "
        "not verified facts; open the link before acting."
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
