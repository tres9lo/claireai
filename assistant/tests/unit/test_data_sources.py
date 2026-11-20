"""The training catalog must be explicit about licence and access."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ALLOWED_ACCESS = {"public", "request", "credentials", "do-not-use"}


def test_every_source_has_a_licence_link_and_access_state() -> None:
    payload = json.loads((ROOT / "brain-lab" / "data-sources.json").read_text(encoding="utf-8"))
    sources = payload["sources"]

    assert len(sources) >= 10
    assert len({source["id"] for source in sources}) == len(sources)
    for source in sources:
        assert source["name"]
        assert source["url"].startswith("https://")
        assert source["license"]
        assert source["access"] in ALLOWED_ACCESS
        assert source["domains"]


def test_sources_with_restrictions_are_not_marked_for_training() -> None:
    payload = json.loads((ROOT / "brain-lab" / "data-sources.json").read_text(encoding="utf-8"))
    by_id = {source["id"]: source for source in payload["sources"]}

    assert by_id["stack-exchange"]["access"] == "do-not-use"
    assert by_id["enron-email"]["access"] == "do-not-use"
    assert by_id["fineweb-edu"]["license"].startswith("ODC-By")
