"""Command-line entry point for the local API service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from claireai.api.app import TOKEN_FILE, create_app, load_or_create_token
from claireai.config import Settings
from claireai.config.endpoint import endpoint_is_local


def run_api(
    *,
    host: str | None = None,
    port: int | None = None,
    data_dir: Path | None = None,
) -> int:
    overrides = {}
    if host:
        overrides["api_host"] = host
    if port is not None:
        overrides["api_port"] = port
    if data_dir is not None:
        overrides["data_dir"] = data_dir
    settings = Settings(**overrides)
    settings.ensure_directories()
    token = settings.api_token_value or load_or_create_token(settings.resolved_data_dir)
    app = create_app(settings, token=token)

    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: the API service needs the optional api extra: "
            "pip install 'claireai[api]'",
            file=sys.stderr,
        )
        return 2

    endpoint = f"http://{settings.api_host}:{settings.api_port}"
    print(f"ClaireAI API listening on {endpoint}")
    print(f"token file: {settings.resolved_data_dir / TOKEN_FILE}")
    if not endpoint_is_local(endpoint):
        print(
            "WARNING: this host is not loopback. The bearer token is protecting "
            "the API, but there is no TLS here; put it behind an HTTPS reverse proxy "
            "before exposing it beyond a trusted network.",
            file=sys.stderr,
        )
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level=settings.log_level.lower())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="claireai-api", description="Run the local ClaireAI HTTP API."
    )
    parser.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="port (default 8765)")
    parser.add_argument("--data-dir", type=Path, default=None, help="data directory")
    args = parser.parse_args(argv)
    return run_api(host=args.host, port=args.port, data_dir=args.data_dir)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
