#!/usr/bin/env python3
"""End-to-end smoke check that runs the same way on Linux and Windows.

    assistant/.venv/bin/python scripts/smoke.py
    assistant/.venv/bin/python scripts/smoke.py --skip-model     # no checkpoint needed

The unit tests prove behaviour with mocked edges; this proves the assembled
thing: settings, database, migrations, a real model turn (unless skipped), the
tool loop, tool routing, the approval refusal, the presence stream, the daily
summary, the read-only today page, retention and the append-only audit log. It
uses only the standard library and `pathlib` so it can be run on either OS
without a shell.

Exit codes: 0 every check passed (skips allowed), 1 a check failed, 2 bad config.
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from claireai.agent import run_task
from claireai.approval import DenyAll
from claireai.audit import AuditLog
from claireai.brain import BrainResponse, ScratchBrain, ToolCall
from claireai.cli import ask
from claireai.config import Settings
from claireai.presence import PresenceStream
from claireai.storage import Database, TaskRepository
from claireai.tools import default_registry

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class Results:
    lines: list[str] = field(default_factory=list)
    failures: int = 0
    skips: int = 0
    passed: int = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            self.lines.append(f"{PASS}  {name}" + (f" — {detail}" if detail else ""))
        else:
            self.failures += 1
            self.lines.append(f"{FAIL}  {name}" + (f" — {detail}" if detail else ""))
        return ok

    def skip(self, name: str, reason: str) -> None:
        self.skips += 1
        self.lines.append(f"{SKIP}  {name} — {reason}")

    def render(self) -> str:
        return "\n".join(self.lines)


class ScriptedBrain:
    """Deterministic brain: the smoke check tests the runtime, not the model."""

    name = "scripted"

    def __init__(self, responses) -> None:  # noqa: ANN001
        self.responses = list(responses)

    async def health(self):  # noqa: ANN201
        raise NotImplementedError

    async def generate(self, messages, tools=None, *, temperature=None) -> BrainResponse:  # noqa: ANN001
        return self.responses.pop(0)


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


async def _run(args: argparse.Namespace) -> int:
    results = Results()
    temporary = None
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="claireai-smoke-")
        data_dir = Path(temporary.name)

    # A smoke run in a fresh temporary directory has no checkpoint, which made
    # the model check fail on every machine that does have one. Copy the active
    # checkpoint in so the check exercises the real runtime; nothing is written
    # back to the real data directory.
    if temporary is not None and not args.skip_model:
        import shutil as _shutil

        candidates = [
            Settings().resolved_model_checkpoint,
            Path(__file__).resolve().parents[1] / "runtime-data" / "models" / "claire-scratch.pt",
        ]
        source = next((path for path in candidates if path.is_file()), None)
        if source is not None:
            destination = data_dir / "models" / "claire-scratch.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(source, destination)

    workspace = data_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "alpha.txt").write_text("alpha", encoding="utf-8")
    (workspace / "beta.txt").write_text("beta beta", encoding="utf-8")

    settings = Settings(_env_file=None, data_dir=data_dir, allowed_folders=[workspace])
    database = Database(settings.database_path)
    conn = database.connect()

    try:
        print(f"ClaireAI smoke check on {platform.system()} {platform.release()} "
              f"({platform.machine()}), Python {platform.python_version()}")
        print(f"data directory: {data_dir}")
        print(f"workspace: {workspace}\n")

        results.check("settings load and validate", True, f"checkpoint={settings.resolved_model_checkpoint}")

        applied = database.migrate()
        results.check(
            "database migrates",
            {"tasks", "task_log"} <= _tables(conn),
            f"applied={', '.join(applied) or 'none (already up to date)'}",
        )

        brain = ScratchBrain(checkpoint=settings.resolved_model_checkpoint)
        health = await brain.health()
        model_ok = health.available and health.model_present
        if args.skip_model:
            results.skip("local model answers", "--skip-model")
        elif not health.available:
            results.check("local model runtime reachable", False, health.detail)
        elif not health.model_present:
            results.check(
                "scratch checkpoint ready",
                False,
                health.detail,
            )
        else:
            results.check("scratch model available", True, ", ".join(health.models))
            results.check("scratch checkpoint ready", True, str(settings.resolved_model_checkpoint))

        if not args.skip_model and model_ok:
            turn = await ask(settings=settings, conn=conn, brain=brain, question="Say OK.")
            task = TaskRepository(conn).get(turn.task_id)
            rows = AuditLog(conn).for_task(turn.task_id)
            results.check(
                "chat answers and is logged",
                bool(turn.content) and task is not None and task.status == "succeeded" and len(rows) == 1,
                f"{turn.duration_ms} ms, {len(turn.content)} chars",
            )
        await brain.aclose()

        tool_brain = ScriptedBrain(
            [
                BrainResponse(
                    content="",
                    model="scripted",
                    tool_calls=(ToolCall(name="list_folder", arguments={"path": str(workspace)}),),
                ),
                BrainResponse(content="There are two files.", model="scripted"),
            ]
        )
        outcome = await run_task(settings=settings, conn=conn, brain=tool_brain, request="list it")
        tool_rows = [entry for entry in AuditLog(conn).for_task(outcome.task_id) if entry.action == "tool.call"]
        results.check(
            "read-only tool runs through the agent loop",
            outcome.steps and outcome.steps[0].status == "done" and len(tool_rows) == 1,
            f"step={outcome.steps[0].status if outcome.steps else 'none'}",
        )

        target = workspace / "smoke-should-not-exist"
        write_brain = ScriptedBrain(
            [
                # A write is only reached after a read: the agent refuses a write
                # whose target nothing has looked at (claireai.agent._execute).
                BrainResponse(
                    content="",
                    model="scripted",
                    tool_calls=(ToolCall(name="list_folder", arguments={"path": str(workspace)}),),
                ),
                BrainResponse(
                    content="",
                    model="scripted",
                    tool_calls=(ToolCall(name="create_folder", arguments={"path": str(target)}),),
                ),
                BrainResponse(content="Not allowed.", model="scripted"),
            ]
        )
        denied = await run_task(
            settings=settings,
            conn=conn,
            brain=write_brain,
            request="make a folder",
            approvals=DenyAll(),
        )
        denied_rows = AuditLog(conn).for_task(denied.task_id)
        results.check(
            "write tool is refused without an approval channel",
            denied.steps[0].tool == "list_folder"
            and denied.steps[0].status == "done"
            and denied.steps[-1].status == "skipped"
            and not target.exists()
            and any(entry.action == "approval.decision" for entry in denied_rows),
            denied.steps[-1].summary[:60],
        )

        stream = PresenceStream()
        seen: list[str] = []
        observed = PresenceStream(reporter=lambda event: seen.append(event.state))
        presence_brain = ScriptedBrain(
            [
                BrainResponse(
                    content="",
                    model="scripted",
                    tool_calls=(ToolCall(name="list_folder", arguments={"path": str(workspace)}),),
                ),
                BrainResponse(content="done", model="scripted"),
            ]
        )
        await run_task(
            settings=settings,
            conn=conn,
            brain=presence_brain,
            request="list again",
            presence=observed,
        )
        results.check(
            "presence reports the run and returns to idle",
            seen[:1] == ["thinking"] and "tool_running" in seen and seen[-1] == "idle",
            " → ".join(seen),
        )
        stream.emit("thinking")
        results.check("presence sequence advances", stream.snapshot().seq == 1)

        captured: dict[str, list] = {}

        class CapturingBrain(ScriptedBrain):
            async def generate(self, messages, tools=None, *, temperature=None):  # noqa: ANN001
                captured["tools"] = list(tools or [])
                return await super().generate(messages, tools)

        routed = await run_task(
            settings=settings,
            conn=conn,
            brain=CapturingBrain([BrainResponse(content="beta.txt is biggest.", model="scripted")]),
            request="List the biggest files in my Downloads folder",
        )
        shown = captured["tools"]
        results.check(
            "tool routing narrows the schema set",
            routed.route == "largest"
            and len(shown) == 1
            and shown[0]["function"]["name"] == "find_largest_files"
            and len(default_registry(browser_enabled=True)) > 1,
            f"{len(shown)} of {len(default_registry(browser_enabled=True))} tools, intent={routed.route}",
        )

        # Imported here so a missing scheduler dependency cannot hide the checks above.
        from claireai.scheduler import run_once

        summary, summary_path = run_once(conn=conn, settings=settings)
        report = summary_path.read_text(encoding="utf-8") if summary_path else ""
        results.check(
            "daily summary builds and saves",
            summary_path is not None
            and summary_path.exists()
            and summary.lines
            and all(f"## {heading}" in report for heading in ("Done", "Failed or skipped", "Still pending")),
            f"{len(summary.lines)} task(s) → {summary_path.name if summary_path else 'no file'}",
        )

        from claireai.retention import apply as prune_apply, plan as prune_plan
        from claireai.today import render_today

        page = render_today(conn)
        results.check(
            "the read-only today page renders",
            "Daily summary" in page
            and "list_folder" in page
            and "<script" not in page.lower()
            and "<form" not in page.lower(),
            f"{len(page)} bytes, no script or form",
        )

        stale = data_dir / "trash" / "20200101T000000Z"
        stale.mkdir(parents=True, exist_ok=True)
        (stale / "old.txt").write_text("stale", encoding="utf-8")
        fresh_batch = data_dir / "trash" / "29990101T000000Z"
        fresh_batch.mkdir(parents=True, exist_ok=True)
        (fresh_batch / "keep.txt").write_text("keep", encoding="utf-8")
        prune = prune_plan(conn=conn, data_dir=data_dir, days=30)
        report_prune = prune_apply(prune, conn=conn, data_dir=data_dir)
        results.check(
            "retention removes only what is old",
            not stale.exists() and fresh_batch.exists() and report_prune.removed_count == 1,
            f"removed {report_prune.removed_count}, kept the recent batch",
        )

        went_read_only = False
        try:
            conn.execute("UPDATE task_log SET status = 'failed' WHERE id = (SELECT MIN(id) FROM task_log)")
            conn.commit()
        except sqlite3.DatabaseError:
            went_read_only = True
        results.check("audit log is append-only at the database level", went_read_only)

    finally:
        conn.close()
        if temporary is not None and not args.keep:
            temporary.cleanup()
        elif temporary is not None:
            print(f"\nkept: {data_dir}")

    print("\n" + results.render())
    print(
        f"\n{results.passed} passed, {results.failures} failed, {results.skips} skipped"
    )
    if results.failures:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end ClaireAI smoke check.")
    parser.add_argument("--data-dir", help="use this directory instead of a temporary one")
    parser.add_argument("--skip-model", action="store_true", help="skip every step that needs a trained checkpoint")
    parser.add_argument("--keep", action="store_true", help="keep the temporary data directory")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    sys.exit(main())
