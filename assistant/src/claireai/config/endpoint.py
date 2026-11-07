"""Endpoint rules shared by configuration and the connector brains.

The assistant is local by design; a remote brain is an explicit opt-in. These
rules are enforced at configuration time *and* again by the adapter, so a
hand-built settings object cannot bypass them:

- the URL must be absolute http(s);
- plain ``http`` is allowed only on loopback, because an API key sent in clear
  text to another host on the network is a credential leak, not a preference.
"""

from __future__ import annotations

from urllib.parse import urlparse

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "[::1]"})

# Where each connector points when the owner names no base URL. The scratch
# provider has none: it reads a checkpoint from disk.
DEFAULT_BASE_URLS = {
    "scratch": "",
    "ollama": "http://127.0.0.1:11434/api",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}


def endpoint_is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    return host in LOOPBACK_HOSTS or host.endswith(".localhost")


def check_endpoint(url: str) -> str:
    """Validate and normalise a brain endpoint, or raise ``ValueError``."""
    text = str(url or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"a brain endpoint must be an absolute http(s) URL, got {url!r}")
    if parsed.scheme == "http" and not endpoint_is_local(text):
        raise ValueError(
            f"refusing to send credentials over plain http to {parsed.hostname!r}; "
            "use https, or a loopback address for a local runtime"
        )
    return text.rstrip("/")
