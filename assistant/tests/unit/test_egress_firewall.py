"""The optional host firewall helper emits safe, scoped nftables rules."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "egress_firewall.py"
SPEC = importlib.util.spec_from_file_location("egress_firewall", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
egress_firewall = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(egress_firewall)


def test_ruleset_is_scoped_to_the_uid_and_web_ports() -> None:
    text = egress_firewall.ruleset(1000)

    assert "table inet claireai_egress" in text
    assert "meta skuid 1000" in text
    assert "tcp dport { 80, 443 }" in text
    assert "10.0.0.0/8" in text
    assert "fc00::/7" in text
    assert "::ffff:a00:0/104" in text


def test_ruleset_keeps_loopback_available() -> None:
    text = egress_firewall.ruleset(1000)

    assert "127.0.0.0/8" not in text
    assert "::1" not in text


def test_ruleset_rejects_root() -> None:
    with pytest.raises(ValueError, match="non-root"):
        egress_firewall.ruleset(0)
