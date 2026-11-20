"""Strict review of a trained candidate before it can replace the local model.

The training worker is allowed to *produce* checkpoints; this module is the only
thing allowed to put one in front of the user. It runs the same fixed suites on
the candidate and on the installed checkpoint, with the same run count, then
applies the written gate:

* tool checks must be strictly better;
* conversation cases must not regress;
* both suites must have completed on the same case counts.

The capability suite is recorded as a diagnostic but is not part of the install
decision. A failed or unavailable suite rejects the candidate; nothing is
promoted on a guess, and the candidate is never deleted. The previous installed
pair is archived under ``models/archive`` before a promotion so it can be
restored.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

DEFAULT_RUNS = 3
MAX_RUNS = 5
SUITE_TIMEOUT_SECONDS = 1_800
ACCEPTABLE_EXITS = {0, 1}

TOOLS_SUITE = "tool_call_eval.py"
CONVERSATION_SUITE = "conversation_eval.py"
CAPABILITY_SUITE = "eval_model.py"


class GateError(Exception):
    """The gate could not produce a trustworthy comparison."""


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    os.replace(temporary, path)


def _score_tools(payload: Mapping[str, Any]) -> dict[str, int]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else payload
    try:
        return {
            "checks_passed": int(summary["checks_passed"]),
            "checks": int(summary["checks"]),
            "cases_stable": int(summary.get("cases_stable", 0)),
            "cases": int(summary.get("cases", 0)),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError("tool-call suite returned no usable summary") from exc


def _score_conversation(payload: Mapping[str, Any]) -> dict[str, int]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else payload
    try:
        return {
            "passed": int(summary["passed"]),
            "total": int(summary["total"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError("conversation suite returned no usable summary") from exc


def _score_capability(payload: Mapping[str, Any]) -> dict[str, int]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else payload
    passed = summary.get("passed", summary.get("checks_passed"))
    total = summary.get("total", summary.get("checks"))
    try:
        return {"passed": int(passed), "total": int(total)}
    except (TypeError, ValueError) as exc:
        raise GateError("capability suite returned no usable summary") from exc


class CandidateGate:
    """Evaluate a candidate and, only on a pass, install it with a backup."""

    def __init__(
        self,
        data_dir: Path,
        *,
        checkout: Path | None = None,
        runs: int = DEFAULT_RUNS,
        installed_checkpoint: Path | None = None,
        runner: Callable[[list[str], Path], subprocess.CompletedProcess] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.checkout = Path(checkout).expanduser() if checkout is not None else None
        self.runs = max(1, min(MAX_RUNS, int(runs)))
        self.installed_checkpoint = (
            Path(installed_checkpoint).expanduser()
            if installed_checkpoint is not None
            else self.data_dir / "models" / "claire-scratch.pt"
        )
        self.runner = runner

    # -- execution -------------------------------------------------------

    def _source(self) -> Path:
        if self.checkout is not None:
            return self.checkout
        from claireai.update import checkout_root

        source = checkout_root(self.data_dir)
        if source is None:
            raise GateError("no source checkout found for the evaluation scripts")
        return source

    def _execute(self, command: list[str], cwd: Path, log) -> subprocess.CompletedProcess:  # noqa: ANN001
        if self.runner is not None:
            return self.runner(command, cwd)
        environment = os.environ.copy()
        environment["CLAIREAI_DATA_DIR"] = str(self.data_dir)
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            capture_output=True,
            text=True,
            timeout=SUITE_TIMEOUT_SECONDS,
            check=False,
        )
        log.write(completed.stdout or "")
        log.write(completed.stderr or "")
        log.flush()
        return completed

    def _suite(
        self,
        *,
        script: Path,
        checkpoint: Path,
        out: Path,
        cwd: Path,
        log,
        sandbox: bool = False,
    ) -> dict[str, int]:  # noqa: ANN001
        command = [
            sys.executable,
            str(script),
            "--model-checkpoint",
            str(checkpoint),
            "--runs",
            str(self.runs),
            "--out",
            str(out),
        ]
        if sandbox:
            command += ["--sandbox", "home"]
        try:
            completed = self._execute(command, cwd, log)
        except (OSError, subprocess.SubprocessError) as exc:
            raise GateError(f"{script.name} could not run: {exc}") from exc
        if completed.returncode not in ACCEPTABLE_EXITS:
            raise GateError(
                f"{script.name} exited {completed.returncode}; see {out.parent / 'gate.log'}"
            )
        try:
            payload = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GateError(f"{script.name} wrote no readable report") from exc
        if script.name == TOOLS_SUITE:
            return _score_tools(payload)
        if script.name == CONVERSATION_SUITE:
            return _score_conversation(payload)
        return _score_capability(payload)

    # -- decision --------------------------------------------------------

    @staticmethod
    def decide(
        candidate: Mapping[str, Mapping[str, int]],
        installed: Mapping[str, Mapping[str, int]],
    ) -> tuple[bool, str]:
        candidate_tools, installed_tools = candidate["tools"], installed["tools"]
        candidate_chat, installed_chat = candidate["conversation"], installed["conversation"]
        if candidate_tools["checks"] != installed_tools["checks"]:
            return False, "tool case counts differ, so the comparison is not valid"
        if candidate_chat["total"] != installed_chat["total"]:
            return False, "conversation case counts differ, so the comparison is not valid"
        if not candidate_tools["checks"] or not candidate_chat["total"]:
            return False, "a suite returned an empty result"

        tools_win = candidate_tools["checks_passed"] > installed_tools["checks_passed"]
        chat_ok = candidate_chat["passed"] >= installed_chat["passed"]
        detail = (
            f"tools {candidate_tools['checks_passed']}/{candidate_tools['checks']} "
            f"vs {installed_tools['checks_passed']}/{installed_tools['checks']}; "
            f"conversation {candidate_chat['passed']}/{candidate_chat['total']} "
            f"vs {installed_chat['passed']}/{installed_chat['total']}"
        )
        if tools_win and chat_ok:
            return True, f"passed: {detail}"
        if not tools_win:
            return False, f"tool checks did not improve: {detail}"
        return False, f"conversation regressed: {detail}"

    def _promote(self, candidate: Path) -> dict[str, str]:
        installed = self.installed_checkpoint
        if not installed.is_file():
            raise GateError(f"installed checkpoint is missing: {installed}")
        candidate_sidecar = candidate.with_suffix(".json")
        installed_sidecar = installed.with_suffix(".json")
        if not candidate_sidecar.is_file() or not installed_sidecar.is_file():
            raise GateError("checkpoint sidecars are required for promotion")

        archive = installed.parent / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        stamp = _stamp()
        backup_pt = archive / f"{installed.stem}-{stamp}.pt"
        backup_json = archive / f"{installed.stem}-{stamp}.json"
        shutil.copy2(installed, backup_pt)
        shutil.copy2(installed_sidecar, backup_json)
        for path in (backup_pt, backup_json):
            try:
                path.chmod(0o600)
            except OSError:
                pass

        pending = []
        try:
            for source, target in (
                (candidate, installed),
                (candidate_sidecar, installed_sidecar),
            ):
                temporary = target.with_name(target.name + ".promoting")
                shutil.copyfile(source, temporary)
                try:
                    temporary.chmod(0o600)
                except OSError:
                    pass
                pending.append((temporary, target))
            for temporary, target in pending:
                os.replace(temporary, target)
        except OSError as exc:
            for temporary, _target in pending:
                try:
                    temporary.unlink()
                except OSError:
                    pass
            # Best effort: put the archived pair back rather than leave a
            # mismatched model and sidecar installed.
            try:
                shutil.copyfile(backup_pt, installed)
                shutil.copyfile(backup_json, installed_sidecar)
            except OSError:
                pass
            raise GateError(f"could not install the candidate: {exc}") from exc
        return {
            "backup": str(backup_pt),
            "backup_sidecar": str(backup_json),
        }

    def review(self, candidate: Path) -> dict[str, Any]:
        candidate = Path(candidate).expanduser()
        if not candidate.is_file():
            raise GateError(f"candidate is missing: {candidate}")
        if not candidate.with_suffix(".json").is_file():
            raise GateError(f"candidate sidecar is missing: {candidate.with_suffix('.json')}")
        installed = self.installed_checkpoint
        if not installed.is_file() or not installed.with_suffix(".json").is_file():
            raise GateError("the installed checkpoint has no sidecar, so it cannot be compared")

        source = self._source()
        scripts = source.parent / "scripts"
        required = [scripts / TOOLS_SUITE, scripts / CONVERSATION_SUITE, scripts / CAPABILITY_SUITE]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise GateError(f"evaluation scripts are missing: {', '.join(missing)}")

        report_dir = self.data_dir / "learning" / "gate" / _stamp()
        report_dir.mkdir(parents=True, exist_ok=True)
        scores: dict[str, dict[str, dict[str, int]]] = {}
        with (report_dir / "gate.log").open("a", encoding="utf-8") as log:
            log.write(f"candidate: {candidate}\ninstalled: {installed}\nruns: {self.runs}\n")
            for label, checkpoint in (("candidate", candidate), ("installed", installed)):
                scores[label] = {
                    "tools": self._suite(
                        script=scripts / TOOLS_SUITE,
                        checkpoint=checkpoint,
                        out=report_dir / f"{label}-tools.json",
                        cwd=source.parent,
                        log=log,
                        sandbox=True,
                    ),
                    "conversation": self._suite(
                        script=scripts / CONVERSATION_SUITE,
                        checkpoint=checkpoint,
                        out=report_dir / f"{label}-conversation.json",
                        cwd=source.parent,
                        log=log,
                    ),
                    "capability": self._suite(
                        script=scripts / CAPABILITY_SUITE,
                        checkpoint=checkpoint,
                        out=report_dir / f"{label}-capability.json",
                        cwd=source.parent,
                        log=log,
                    ),
                }

        passed, reason = self.decide(scores["candidate"], scores["installed"])
        result: dict[str, Any] = {
            "passed": passed,
            "reason": reason,
            "scores": scores,
            "report": str(report_dir),
            "runs": self.runs,
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "promoted": False,
        }
        if passed:
            result.update(self._promote(candidate))
            result["promoted"] = True
        _write_json(report_dir / "decision.json", result)
        return result


__all__ = ["CandidateGate", "GateError", "DEFAULT_RUNS", "MAX_RUNS"]
