"""Command-line chat against the configured local model.

This is the Month 1, Week 1 acceptance surface: one typed question goes to the
Brain interface, the answer is printed, and the same turn is written to the
task log so the log and the conversation can never drift apart.

    python -m claireai.cli --once "Explain what a tokenizer does."
    python -m claireai.cli                # interactive, /log shows the audit tail
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from pydantic import ValidationError

from claireai.agent import AgentError, describe_steps, run_task
from claireai.approval import ApprovalBroker, ApprovalDecision, ApprovalRequest, DenyAll
from claireai.audit import AuditLog, redact
from claireai.brain import Brain, BrainError, Message, OllamaBrain
from claireai.config import Settings
from claireai.permissions import Sandbox
from claireai.presence import PresenceStream
from claireai.retention import (
    DEFAULT_RETENTION_DAYS,
    apply as apply_prune,
    describe as describe_prune,
    plan as plan_prune,
)
from claireai.scheduler import build_scheduler, next_run_label, run_once
from claireai.storage import Database, TaskRepository, transaction
from claireai.summary import DailySummary, SummaryError, parse_day, render_markdown
from claireai.tools import default_registry

SYSTEM_PROMPT = (
    "You are ClaireAI, a local personal assistant running on the user's own computer. "
    "Answer briefly and concretely, and say so plainly when you are unsure. "
    "You have no tools yet: do not claim to have performed actions."
)

EXIT_OK = 0
EXIT_TASK_FAILED = 1
EXIT_CONFIG = 2
EXIT_RUNTIME = 3


@dataclass
class CliApprovalBroker:
    """Ask the person at the terminal before a write happens.

    A non-interactive stdin refuses instead of blocking: piping a question into
    the CLI must never be a way to auto-approve a file write.
    """

    name: str = "cli"
    interactive: bool = field(default_factory=lambda: sys.stdin.isatty())
    read: Callable[[str], str] = field(default_factory=lambda: input)
    write: Callable[[str], None] = field(default_factory=lambda: print)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if not self.interactive:
            return ApprovalDecision(
                approved=False,
                reason=(
                    "stdin is not an interactive terminal, so approval cannot be asked; "
                    "run claireai in a terminal to approve write actions"
                ),
                decided_by="policy",
            )
        self.write("\n" + request.describe())
        try:
            answer = self.read("approve? [y/N] ").strip().lower()
        except EOFError:
            return ApprovalDecision(approved=False, reason="no answer given (end of input)")
        if answer in {"y", "yes"}:
            return ApprovalDecision(approved=True)
        return ApprovalDecision(approved=False, reason="denied at the prompt")


@dataclass(frozen=True)
class ChatTurn:
    task_id: str
    content: str
    model: str
    duration_ms: int


async def ask(
    *,
    settings: Settings,
    conn: sqlite3.Connection,
    brain: Brain,
    question: str,
    device: str = "desktop",
) -> ChatTurn:
    """Ask one question, recording the task and audit rows for the turn."""
    tasks = TaskRepository(conn)
    audit = AuditLog(conn)

    with transaction(conn):
        task = tasks.create(request=redact(question), kind="chat", device=device)
    with transaction(conn):
        tasks.update_status(task.id, "running")

    started = time.perf_counter()
    try:
        response = await brain.generate(
            [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=question)]
        )
    except BrainError as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        with transaction(conn):
            tasks.update_status(task.id, "failed", error=str(exc))
            audit.record(
                task_id=task.id,
                device=device,
                action="brain.generate",
                status="failed",
                target=settings.ollama_model,
                detail={"model": settings.ollama_model, "duration_ms": duration_ms, "error": str(exc)},
            )
        raise

    duration_ms = int((time.perf_counter() - started) * 1000)
    with transaction(conn):
        tasks.update_status(task.id, "succeeded")
        audit.record(
            task_id=task.id,
            device=device,
            action="brain.generate",
            status="done",
            tool=None,
            target=response.model,
            detail={
                "model": response.model,
                "duration_ms": duration_ms,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            },
        )
    return ChatTurn(task_id=task.id, content=response.content, model=response.model, duration_ms=duration_ms)


def _print_turn(turn: ChatTurn, database_path: Path) -> None:
    print(turn.content)
    print(f"\n[task {turn.task_id} | {turn.model} | {turn.duration_ms} ms | logged: {database_path}]\n")


def _print_summary(summary: DailySummary, path: Path | None) -> None:
    print()
    print(render_markdown(summary))
    if path:
        print(f"saved: {path}")


def _run_prune(args: argparse.Namespace, settings: Settings, conn: sqlite3.Connection) -> int:
    """`--prune` shows what is old; `--prune --apply` removes it."""
    from claireai.retention import RetentionError, human_size

    try:
        plan = plan_prune(conn=conn, data_dir=settings.resolved_data_dir, days=args.days)
    except RetentionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    print(describe_prune(plan))
    if not args.apply:
        print("dry run: nothing was removed (add --apply to remove these items)")
        return EXIT_OK

    report = apply_prune(plan, conn=conn, data_dir=settings.resolved_data_dir, device=args.device)
    print(f"removed {report.removed_count} item(s), freed {human_size(report.freed_bytes)}")
    for error in report.errors:
        print(f"  ! {error}", file=sys.stderr)
    if report.errors:
        print("the run was logged as failed; the items listed above were not removed", file=sys.stderr)
        return EXIT_TASK_FAILED
    return EXIT_OK


def _run_summary(args: argparse.Namespace, settings: Settings, conn: sqlite3.Connection) -> int:
    """`--summary [DATE]` prints one day; `--summary-at HH:MM` waits for the job."""
    try:
        if args.summary_at:
            scheduler = build_scheduler(settings=settings, when=args.summary_at, on_run=_print_summary)
            print(f"daily summary scheduled for {args.summary_at} local time; Ctrl+C stops it")
            print(f"next run: {next_run_label(scheduler)}")
            scheduler.start()
            return EXIT_OK

        day = parse_day(args.summary)
        summary, path = run_once(conn=conn, settings=settings, day=day)
        _print_summary(summary, path)
        return EXIT_OK
    except SummaryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except (KeyboardInterrupt, SystemExit):
        return EXIT_OK


def _print_log(conn: sqlite3.Connection, limit: int = 10) -> None:
    entries = AuditLog(conn).recent(limit)
    if not entries:
        print("(no audit entries yet)")
        return
    for entry in entries:
        task = (entry.task_id or "-")[:8]
        print(f"{entry.created_at}  {entry.status:<7} {entry.action:<16} task={task}  {entry.detail or ''}")


async def _handle_turn(
    *,
    args: argparse.Namespace,
    settings: Settings,
    conn: sqlite3.Connection,
    brain: Brain,
    question: str,
    database_path: Path,
) -> int:
    """Answer one question, with tools unless ``--no-tools`` was given."""
    if args.no_tools:
        try:
            turn = await ask(settings=settings, conn=conn, brain=brain, question=question, device=args.device)
        except BrainError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_RUNTIME
        _print_turn(turn, database_path)
        return EXIT_OK

    approvals: ApprovalBroker = CliApprovalBroker()
    try:
        outcome = await run_task(
            settings=settings,
            conn=conn,
            brain=brain,
            request=question,
            approvals=approvals,
            presence=getattr(args, "presence", None),
            device=args.device,
            dry_run=args.dry_run,
        )
    except AgentError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return EXIT_TASK_FAILED
    except BrainError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    if outcome.steps:
        print("steps:")
        print(describe_steps(outcome.steps))
        print()
    print(outcome.content or "(the model returned no answer)")
    route_label = f" | route: {outcome.route}" if outcome.route else ""
    print(
        f"\n[task {outcome.task_id} | {outcome.model} | {outcome.duration_ms} ms"
        f" | steps: {len(outcome.steps)}{route_label} | logged: {database_path}]\n"
    )
    return EXIT_OK


async def _run(args: argparse.Namespace, settings: Settings, database_path: Path) -> int:
    database = Database(database_path)
    applied = database.migrate()
    conn = database.connect()
    if applied:
        print(f"Applied migrations: {', '.join(applied)}")
    print(f"database: {database_path}")
    print(f"brain: {'ollama'} @ {settings.ollama_base_url} model={settings.ollama_model}")
    sandbox = Sandbox.from_settings(settings)
    tool_label = "off (--no-tools)" if args.no_tools else f"{len(default_registry())} tools"
    print(f"tools: {tool_label} | allow-list: {sandbox.describe()}")
    print(f"routing: {settings.tool_routing} (narrow the tool set per request)")
    approvals_label = "interactive prompt" if sys.stdin.isatty() else "none (write tools refused)"
    print(f"approvals: {approvals_label}")
    if args.dry_run:
        print("dry run: tools will report what they would do, nothing will run")

    # The daily summary and retention need no model, so they run before the
    # runtime is touched.
    if args.summary is not None or args.summary_at:
        try:
            return _run_summary(args, settings, conn)
        finally:
            conn.close()
    if args.prune:
        try:
            return _run_prune(args, settings, conn)
        finally:
            conn.close()

    brain = OllamaBrain(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
        # Bounded decoding: an unbounded reply took 126 s on this 4-thread CPU.
        options={"num_predict": settings.max_output_tokens},
    )
    try:
        health = await brain.health()
        if not health.available:
            print(f"\nERROR: local model runtime unavailable — {health.detail}", file=sys.stderr)
            print("Install/start Ollama, then retry. No hosted fallback is used.", file=sys.stderr)
            return EXIT_RUNTIME
        if not health.model_present:
            print(f"\nERROR: model {settings.ollama_model!r} is not installed locally.", file=sys.stderr)
            print(f"Download it once: ollama pull {settings.ollama_model}", file=sys.stderr)
            print(f"Installed models: {', '.join(health.models) or '(none)'}", file=sys.stderr)
            return EXIT_RUNTIME

        if args.once:
            return await _handle_turn(
                args=args,
                settings=settings,
                conn=conn,
                brain=brain,
                question=args.once,
                database_path=database_path,
            )

        print("ClaireAI ready. Commands: /log [n], /summary, /exit\n")
        while True:
            try:
                question = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return EXIT_OK
            if not question:
                continue
            if question in {"/exit", "/quit"}:
                return EXIT_OK
            if question.startswith("/log"):
                parts = question.split()
                limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
                _print_log(conn, limit)
                continue
            if question == "/summary":
                summary, path = run_once(conn=conn, settings=settings)
                _print_summary(summary, path)
                continue
            await _handle_turn(
                args=args,
                settings=settings,
                conn=conn,
                brain=brain,
                question=question,
                database_path=database_path,
            )
    finally:
        await brain.aclose()
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claireai", description="Chat with the local ClaireAI model.")
    parser.add_argument("--once", "-q", metavar="QUESTION", help="ask one question, print the answer, exit")
    parser.add_argument("--model", help="override CLAIREAI_OLLAMA_MODEL for this run")
    parser.add_argument("--base-url", help="override CLAIREAI_OLLAMA_BASE_URL (loopback only by default)")
    parser.add_argument("--data-dir", help="override the application data directory (development use)")
    parser.add_argument("--db", help="use a specific SQLite file instead of the default location")
    parser.add_argument("--device", default="desktop", help="device label recorded in the task log")
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="plain chat: do not offer file tools to the model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show which tools would be called without running them",
    )
    parser.add_argument(
        "--presence",
        action="store_true",
        help="print presence state changes (used by the renderer preview)",
    )
    parser.add_argument(
        "--no-routing",
        action="store_true",
        help="offer every tool on every call instead of narrowing per request",
    )
    parser.add_argument(
        "--summary",
        nargs="?",
        const="",
        metavar="DATE",
        help="print and save the daily summary for DATE (YYYY-MM-DD, default today)",
    )
    parser.add_argument(
        "--summary-at",
        metavar="HH:MM",
        help="run the summary job every day at this local time and stay in the foreground",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="show trash batches and summaries older than --days (dry run unless --apply)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        metavar="N",
        help=f"retention window in days for --prune (default {DEFAULT_RETENTION_DAYS})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="with --prune, actually remove the planned items and log the run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides: dict[str, object] = {}
    if args.model:
        overrides["ollama_model"] = args.model
    if args.base_url:
        overrides["ollama_base_url"] = args.base_url
    if args.data_dir:
        overrides["data_dir"] = Path(args.data_dir)
    if args.no_routing:
        overrides["tool_routing"] = "off"

    try:
        settings = Settings(**overrides)
        settings.ensure_directories()
    except ValidationError as exc:
        print("Invalid configuration:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            label = f"{field}: " if field else ""
            print(f"  {label}{error['msg']}", file=sys.stderr)
        return EXIT_CONFIG

    if args.presence:
        args.presence = PresenceStream(
            reporter=lambda event: print(f"[presence] {event.state} seq={event.seq}")
        )
    else:
        args.presence = None

    database_path = Path(args.db).expanduser() if args.db else settings.database_path
    try:
        return asyncio.run(_run(args, settings, database_path))
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
