#!/usr/bin/env python3
"""End-to-end smoke check that runs the same way on Linux and Windows.

    assistant/.venv/bin/python scripts/smoke.py
    assistant/.venv/bin/python scripts/smoke.py --skip-model     # no Ollama needed

The unit tests prove behaviour with mocked edges; this proves the assembled
thing: settings, database, migrations, a real model turn (unless skipped), the
tool loop, the approval refusal, the presence stream and the append-only audit
log. It uses only the standard library and `pathlib` so it can be run on either
OS without a shell.

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
from claireai.brain import BrainResponse, OllamaBrain, ToolCall
from claireai.cli import ask
from claireai.config import Settings
from claireai.presence import PresenceStream
from claireai.storage import Database, TaskRepository

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

    async def generate(self, messages, tools=None) -> BrainResponse:  # noqa: ANN001
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

        results.check("settings load and validate", True, f"model={settings.ollama_model}")

        applied = database.migrate()
        results.check(
            "database migrates",
            {"tasks", "task_log"} <= _tables(conn),
            f"applied={', '.join(applied) or 'none (already up to date)'}",
        )

        brain = OllamaBrain(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=settings.ollama_timeout_seconds,
        )
        health = await brain.health()
        model_ok = health.available and health.model_present
        if args.skip_model:
            results.skip("local model answers", "--skip-model")
        elif not health.available:
            results.check("local model runtime reachable", False, health.detail)
        elif not health.model_present:
            results.check(
                "configured model installed",
                False,
                f"{settings.ollama_model} missing; run: ollama pull {settings.ollama_model}",
            )
        else:
            results.check("local model runtime reachable", True, f"{len(health.models)} model(s)")
            results.check("configured model installed", True, settings.ollama_model)

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
            denied.steps[0].status == "skipped"
            and not target.exists()
            and any(entry.action == "approval.decision" for entry in denied_rows),
            denied.steps[0].summary[:60],
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
    parser.add_argument("--skip-model", action="store_true", help="skip every step that needs Ollama")
    parser.add_argument("--keep", action="store_true", help="keep the temporary data directory")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    sys.exit(main())
