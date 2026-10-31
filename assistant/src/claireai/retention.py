"""Retention: keep the assistant's own data from growing without bound.

Three things accumulate on a machine running this every day:

* **trash batches** — ``move_to_trash`` never deletes, so every trashed file is
  still on disk under ``<data-dir>/trash/<stamp>/``;
* **summaries** — one row plus one markdown file per day;
* **the audit log** — one row per tool call, forever.

The first two are pruned here. The third is **not**, deliberately: dropping the
``task_log`` triggers to make deletion possible would give up the append-only
guarantee the threat model rests on, in exchange for saving a few kilobytes of
text. The plan reports its size so the growth stays visible, and says why it is
kept.

Safety rules, all enforced in :func:`plan` rather than trusted to the caller:

* nothing outside the data directory is ever named for removal;
* a symlinked trash batch is refused rather than followed;
* a timestamp that cannot be parsed is never treated as "old";
* a retention window below one day is rejected.
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from claireai.audit import AuditLog
from claireai.storage import transaction

DEFAULT_RETENTION_DAYS = 30
TRASH_DIRNAME = "trash"
SUMMARIES_DIRNAME = "summaries"
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
AUDIT_NOTE = (
    "task_log is append-only by database triggers; pruning it would remove that "
    "guarantee, so it is reported and left alone"
)


class RetentionError(ValueError):
    """The retention request was unsafe or malformed."""


@dataclass(frozen=True)
class Removal:
    kind: str  # "trash_batch" | "summary_file" | "summary_row"
    target: str
    age_days: int
    size_bytes: int = 0


@dataclass(frozen=True)
class PrunePlan:
    days: int
    cutoff: str
    removals: tuple[Removal, ...] = field(default_factory=tuple)
    kept_trash: int = 0
    kept_summaries: int = 0
    audit_rows: int = 0
    audit_bytes: int = 0
    refusals: tuple[str, ...] = field(default_factory=tuple)

    @property
    def freed_bytes(self) -> int:
        return sum(item.size_bytes for item in self.removals)

    @property
    def is_empty(self) -> bool:
        return not self.removals


@dataclass(frozen=True)
class PruneReport:
    removed: tuple[Removal, ...]
    freed_bytes: int
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def removed_count(self) -> int:
        return len(self.removed)


def _human(size: int) -> str:
    step = 1024.0
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < step or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} GiB"


human_size = _human


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def _dir_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _batch_stamp(name: str) -> datetime | None:
    try:
        return datetime.strptime(name, STAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def plan(
    *,
    conn: sqlite3.Connection,
    data_dir: Path,
    days: int = DEFAULT_RETENTION_DAYS,
    now: datetime | None = None,
) -> PrunePlan:
    """Work out what is older than ``days`` without touching anything."""
    if days < 1:
        raise RetentionError(f"the retention window must be at least 1 day, got {days}")
    data_dir = Path(data_dir).expanduser()
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    cutoff_date = cutoff.date()
    removals: list[Removal] = []
    refusals: list[str] = []

    trash_root = data_dir / TRASH_DIRNAME
    kept_trash = 0
    if trash_root.is_dir():
        for batch in sorted(trash_root.iterdir()):
            if not batch.is_dir() and not batch.is_symlink():
                continue
            if batch.is_symlink() or not _under(batch, trash_root):
                refusals.append(f"{batch}: not pruned (symlink or outside the trash folder)")
                continue
            stamp = _batch_stamp(batch.name)
            if stamp is None:
                refusals.append(f"{batch}: not pruned (folder name is not a timestamp)")
                continue
            if stamp >= cutoff:
                kept_trash += 1
                continue
            removals.append(
                Removal(
                    kind="trash_batch",
                    target=str(batch),
                    age_days=(now - stamp).days,
                    size_bytes=_dir_bytes(batch),
                )
            )

    summaries_root = data_dir / SUMMARIES_DIRNAME
    kept_summaries = 0
    if summaries_root.is_dir():
        for path in sorted(summaries_root.glob("*.md")):
            try:
                day = datetime.strptime(path.stem, "%Y-%m-%d").date()
            except ValueError:
                refusals.append(f"{path}: not pruned (name is not a date)")
                continue
            if day >= cutoff_date:
                kept_summaries += 1
                continue
            removals.append(
                Removal(
                    kind="summary_file",
                    target=str(path),
                    age_days=(now.date() - day).days,
                    size_bytes=path.stat().st_size if path.exists() else 0,
                )
            )

    rows = conn.execute(
        "SELECT summary_date FROM summaries WHERE summary_date < ? ORDER BY summary_date",
        (cutoff_date.isoformat(),),
    ).fetchall()
    for row in rows:
        removals.append(
            Removal(
                kind="summary_row",
                target=str(row["summary_date"]),
                age_days=(now.date() - datetime.strptime(row["summary_date"], "%Y-%m-%d").date()).days,
            )
        )

    audit_rows = conn.execute("SELECT COUNT(*) AS n FROM task_log").fetchone()["n"]
    audit_bytes = int(
        conn.execute("SELECT COALESCE(SUM(LENGTH(detail)), 0) AS n FROM task_log").fetchone()["n"]
    )

    return PrunePlan(
        days=days,
        cutoff=cutoff.isoformat(timespec="seconds"),
        removals=tuple(removals),
        kept_trash=kept_trash,
        kept_summaries=kept_summaries,
        audit_rows=audit_rows,
        audit_bytes=audit_bytes,
        refusals=tuple(refusals),
    )


def apply(
    plan_: PrunePlan,
    *,
    conn: sqlite3.Connection,
    data_dir: Path,
    device: str = "desktop",
) -> PruneReport:
    """Carry out a plan, then record what happened in the audit log."""
    data_dir = Path(data_dir).expanduser()
    removed: list[Removal] = []
    errors: list[str] = []

    for item in plan_.removals:
        try:
            if item.kind == "trash_batch":
                path = Path(item.target)
                if path.is_symlink() or not _under(path, data_dir / TRASH_DIRNAME):
                    errors.append(f"{path}: refused at apply time")
                    continue
                shutil.rmtree(path)
                removed.append(item)
            elif item.kind == "summary_file":
                path = Path(item.target)
                if not _under(path, data_dir / SUMMARIES_DIRNAME):
                    errors.append(f"{path}: refused at apply time")
                    continue
                path.unlink(missing_ok=True)
                removed.append(item)
            elif item.kind == "summary_row":
                conn.execute("DELETE FROM summaries WHERE summary_date = ?", (item.target,))
                removed.append(item)
        except OSError as exc:
            errors.append(f"{item.target}: {exc}")

    report = PruneReport(removed=tuple(removed), freed_bytes=sum(r.size_bytes for r in removed), errors=tuple(errors))
    with transaction(conn):
        AuditLog(conn).record(
            action="retention.prune",
            status="done" if not errors else "failed",
            device=device,
            detail={
                "days": plan_.days,
                "cutoff": plan_.cutoff,
                "removed": len(removed),
                "freed_bytes": report.freed_bytes,
                "errors": list(errors),
                "audit_rows_kept": plan_.audit_rows,
            },
        )
    return report


def describe(plan_: PrunePlan) -> str:
    """Plain-text plan for the CLI, before anything is deleted."""
    lines = [
        f"retention: keep {plan_.days} day(s); cutoff {plan_.cutoff}",
        f"trash batches kept: {plan_.kept_trash} | summaries kept: {plan_.kept_summaries}",
    ]
    if plan_.is_empty:
        lines.append("nothing to remove")
    else:
        lines.append(f"would remove {len(plan_.removals)} item(s), {_human(plan_.freed_bytes)}:")
        for item in plan_.removals:
            size = f" ({_human(item.size_bytes)})" if item.size_bytes else ""
            lines.append(f"  - {item.kind}: {item.target}{size}, {item.age_days} day(s) old")
    lines.append(f"audit log kept: {plan_.audit_rows} row(s), ~{_human(plan_.audit_bytes)} — {AUDIT_NOTE}")
    for refusal in plan_.refusals:
        lines.append(f"  ! {refusal}")
    return "\n".join(lines)


__all__ = [
    "AUDIT_NOTE",
    "DEFAULT_RETENTION_DAYS",
    "PrunePlan",
    "PruneReport",
    "Removal",
    "RetentionError",
    "apply",
    "describe",
    "human_size",
    "plan",
]
