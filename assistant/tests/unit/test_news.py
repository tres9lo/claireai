"""News must come from a feed and carry a publisher/link, never from the model."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from claireai import news

FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Top stories</title>
<item><title>Kenya launches a new rail line - Example News</title>
<link>https://example.test/kenya-rail</link>
<pubDate>Fri, 20 Nov 2026 08:00:00 GMT</pubDate>
<source url="https://example.test">Example News</source></item>
<item><title>Second story</title><link>https://example.test/second</link>
<pubDate>Fri, 20 Nov 2026 07:00:00 GMT</pubDate><source>Wire</source></item>
</channel></rss>"""


def test_country_detection_is_specific() -> None:
    assert news.countries_in("What is the latest news from Kenya and France?") == [
        "kenya",
        "france",
    ]
    assert news.countries_in("Tell us about the weather") == []
    assert news.is_news_request("Give me headlines for Rwanda") is True
    assert news.is_news_request("Give me a recipe for Rwanda beans") is False


def test_brief_keeps_publisher_date_and_link(monkeypatch) -> None:
    news._CACHE.clear()
    monkeypatch.setattr(news, "_fetch_feed", lambda url: FEED)
    text = news.brief(["Kenya"], limit=2)

    assert "Kenya launches a new rail line" in text
    assert "Example News" in text
    assert "https://example.test/kenya-rail" in text
    assert "2026-11-20 08:00 UTC" in text
    assert "Source: Google News" in text


def test_feed_failure_is_reported_not_invented(monkeypatch) -> None:
    news._CACHE.clear()

    def fail(url):  # noqa: ANN001, ANN202
        raise news.NewsError("network unavailable")

    monkeypatch.setattr(news, "_fetch_feed", fail)
    text = news.brief(["Kenya"])
    assert "Could not retrieve" in text
    assert "network unavailable" in text
    assert "Kenya launches" not in text


def test_unsupported_country_is_clear() -> None:
    with pytest.raises(news.NewsError, match="unsupported country"):
        news.fetch_country("Atlantis")


def test_unknown_request_is_not_forced_through_news(monkeypatch) -> None:
    news._CACHE.clear()
    monkeypatch.setattr(news, "_fetch_feed", lambda url: FEED)
    assert news.answer_news("What is 2 + 2?") is None
    assert "Kenya" in (news.answer_news("news from Kenya") or "")
