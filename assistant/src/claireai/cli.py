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
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from claireai.audit import AuditLog, redact
from claireai.brain import Brain, BrainError, Message, OllamaBrain
from claireai.config import Settings
from claireai.storage import Database, TaskRepository, transaction

SYSTEM_PROMPT = (
    "You are ClaireAI, a local personal assistant running on the user's own computer. "
    "Answer briefly and concretely, and say so plainly when you are unsure. "
    "You have no tools yet: do not claim to have performed actions."
)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_RUNTIME = 3


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


def _print_log(conn: sqlite3.Connection, limit: int = 10) -> None:
    entries = AuditLog(conn).recent(limit)
    if not entries:
        print("(no audit entries yet)")
        return
    for entry in entries:
        task = (entry.task_id or "-")[:8]
        print(f"{entry.created_at}  {entry.status:<7} {entry.action:<16} task={task}  {entry.detail or ''}")


async def _run(args: argparse.Namespace, settings: Settings, database_path: Path) -> int:
    database = Database(database_path)
    applied = database.migrate()
    conn = database.connect()
    if applied:
        print(f"Applied migrations: {', '.join(applied)}")
    print(f"database: {database_path}")
    print(f"brain: {'ollama'} @ {settings.ollama_base_url} model={settings.ollama_model}")

    brain = OllamaBrain(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout_seconds,
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
            try:
                turn = await ask(
                    settings=settings, conn=conn, brain=brain, question=args.once, device=args.device
                )
            except BrainError as exc:
                print(f"\nERROR: {exc}", file=sys.stderr)
                return EXIT_RUNTIME
            _print_turn(turn, database_path)
            return EXIT_OK

        print("ClaireAI ready. Commands: /log [n], /exit\n")
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
            try:
                turn = await ask(
                    settings=settings, conn=conn, brain=brain, question=question, device=args.device
                )
            except BrainError as exc:
                print(f"ERROR: {exc}\n", file=sys.stderr)
                continue
            _print_turn(turn, database_path)
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

    database_path = Path(args.db).expanduser() if args.db else settings.database_path
    try:
        return asyncio.run(_run(args, settings, database_path))
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
