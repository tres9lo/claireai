from __future__ import annotations

import pytest

from claireai.improvements import ImprovementError, ImprovementLedger


def test_candidate_requires_evaluation_and_owner_approval_before_promotion(tmp_path):
    data_dir = tmp_path / "data"
    candidate_dir = data_dir / "models" / "candidates"
    candidate_dir.mkdir(parents=True)
    checkpoint = candidate_dir / "candidate.pt"
    checkpoint.write_bytes(b"candidate checkpoint")
    active = data_dir / "models" / "active.pt"
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_bytes(b"active checkpoint")

    ledger = ImprovementLedger(data_dir)
    candidate_id = ledger.record_candidate(
        checkpoint=checkpoint,
        parent=active,
        manifest={"training_loss": 1.0, "validation_loss": 1.1, "corpus": []},
    )
    with pytest.raises(ImprovementError, match="evaluate"):
        ledger.promote(candidate_id, active_checkpoint=active)

    ledger.record_evaluation(candidate_id, {"all_stable": False, "cases_stable": 0}, eligible=True)
    with pytest.raises(ImprovementError, match="fully passing"):
        ledger.decide(candidate_id, "approved")
    ledger.record_evaluation(candidate_id, {"all_stable": True, "cases_stable": 1}, eligible=False)
    with pytest.raises(ImprovementError, match="fully passing"):
        ledger.decide(candidate_id, "approved")
    ledger.record_evaluation(candidate_id, {"all_stable": True, "cases_stable": 8}, eligible=True)
    ledger.decide(candidate_id, "approved")
    ledger.promote(candidate_id, active_checkpoint=active)

    assert active.read_bytes() == b"candidate checkpoint"
    assert list((data_dir / "models" / "archive").glob("*.pt"))[0].read_bytes() == b"active checkpoint"
    assert ledger.list()[0]["status"] == "promoted"
