"""Loopback HTTP service around the existing ClaireAI runtime."""

from claireai.api.app import create_app

__all__ = ["create_app"]
