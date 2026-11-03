"""Append-only review ledger for opt-in scratch-model training candidates."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ImprovementError(Exception):
    """An improvement candidate is missing, invalid, or not ready to promote."""


class ImprovementLedger:
    """Track training, evaluation, owner decisions, and checkpoint promotion."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.path = self.data_dir / "improvements" / "ledger.jsonl"

    def _append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, line.encode("utf-8"))
        finally:
            os.close(descriptor)

    def _events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise ImprovementError(f"improvement ledger line {number} is invalid") from exc
            if not isinstance(event, dict) or not isinstance(event.get("id"), str):
                raise ImprovementError(f"improvement ledger line {number} has an invalid event")
            events.append(event)
        return events

    def list(self) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        for event in self._events():
            candidate_id = event["id"]
            kind = event.get("event")
            if kind == "candidate":
                candidates[candidate_id] = {**event, "status": "needs_evaluation"}
            elif candidate_id in candidates and kind == "evaluation":
                candidates[candidate_id]["evaluation"] = event.get("summary", {})
                candidates[candidate_id]["status"] = "ready_for_review" if event.get("all_stable") is True else "evaluation_failed"
            elif candidate_id in candidates and kind == "decision":
                candidates[candidate_id]["status"] = event.get("status", "needs_evaluation")
            elif candidate_id in candidates and kind == "promoted":
                candidates[candidate_id]["status"] = "promoted"
                candidates[candidate_id]["promoted_at"] = event.get("at")
        return sorted(candidates.values(), key=lambda item: item.get("created_at", ""), reverse=True)

    def record_candidate(self, *, checkpoint: Path, parent: Path, manifest: dict[str, Any]) -> str:
        candidate = Path(checkpoint).resolve()
        try:
            relative = candidate.relative_to(self.data_dir)
        except ValueError as exc:
            raise ImprovementError("candidate checkpoint must be inside the ClaireAI data directory") from exc
        candidate_id = uuid.uuid4().hex
        self._append({
            "event": "candidate",
            "id": candidate_id,
            "created_at": _now(),
            "checkpoint": str(relative),
            "parent_checkpoint": str(Path(parent).resolve()),
            "training_loss": manifest.get("training_loss"),
            "validation_loss": manifest.get("validation_loss"),
            "corpus": manifest.get("corpus", []),
        })
        return candidate_id

    def record_evaluation(self, candidate_id: str, summary: dict[str, Any], *, eligible: bool) -> None:
        candidate = self._candidate(candidate_id)
        self._append({
            "event": "evaluation",
            "id": candidate_id,
            "at": _now(),
            "all_stable": eligible and summary.get("all_stable") is True,
            "eligible": eligible,
            "summary": summary,
        })

    def decide(self, candidate_id: str, status: str) -> None:
        if status not in {"approved", "rejected"}:
            raise ImprovementError("decision must be approved or rejected")
        candidate = self._candidate(candidate_id)
        if candidate.get("status") not in {"ready_for_review", "evaluation_failed", "approved"}:
            raise ImprovementError("evaluate this candidate before reviewing it")
        if status == "approved" and candidate.get("status") != "ready_for_review":
            raise ImprovementError("only a candidate with a fully passing evaluation can be approved")
        self._append({"event": "decision", "id": candidate_id, "at": _now(), "status": status})

    def promote(self, candidate_id: str, *, active_checkpoint: Path) -> Path:
        candidate = self._candidate(candidate_id)
        if candidate.get("status") != "approved":
            raise ImprovementError("approve a fully evaluated candidate before promotion")
        source = (self.data_dir / candidate["checkpoint"]).resolve()
        candidate_root = (self.data_dir / "models" / "candidates").resolve()
        if candidate_root not in source.parents or not source.is_file():
            raise ImprovementError("candidate file is missing or outside the candidates directory")
        target = Path(active_checkpoint).expanduser().resolve()
        if source == target:
            raise ImprovementError("candidate already is the active checkpoint")
        target.parent.mkdir(parents=True, exist_ok=True)
        archive_dir = self.data_dir / "models" / "archive"
        if target.is_file():
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive = archive_dir / f"{target.stem}-{_now().replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:8]}.pt"
            shutil.copy2(target, archive)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self._append({"event": "promoted", "id": candidate_id, "at": _now(), "active_checkpoint": str(target)})
        return target

    def _candidate(self, candidate_id: str) -> dict[str, Any]:
        for candidate in self.list():
            if candidate["id"] == candidate_id:
                return candidate
        raise ImprovementError(f"unknown improvement candidate {candidate_id!r}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
