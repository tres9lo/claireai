"""A small, authenticated HTTP surface for the existing runtime.

It is a backend, not a second assistant: every chat turn goes through ``ask`` or
``run_task``, so the folder allow-list, risk policy, approvals and audit log are
the same ones the window uses. Write tools use ``FileApprovalBroker`` and
therefore still need a signed token issued from a terminal; a bearer token alone
cannot approve a file write.

Bind loopback by default. The administrator token is read from
``CLAIREAI_API_TOKEN`` or a 0600 ``api.token`` file in the data directory. Only
``/health`` and one-time device claiming are unauthenticated.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from claireai.agent import AgentError, run_task
from claireai.api.devices import (
    claim_pairing,
    create_pairing,
    device_for_token,
    list_devices,
    revoke_device,
)
from claireai.approval_file import (
    FileApprovalBroker,
    pending_approvals,
    revoke_approval,
)
from claireai.audit import AuditLog, redact
from claireai.brain import BrainError, build_brain
from claireai.cli import ask
from claireai.config import Settings
from claireai.memory import EmbeddingError, embedder_for, render_hits
from claireai.runtime_control import RuntimeControl
from claireai.storage import (
    ConversationRepository,
    Database,
    PreferenceRepository,
    TaskRepository,
    transaction,
)
from claireai.storage.tasks import TERMINAL_STATUSES, utc_now
from claireai.summary import build as build_summary
from claireai.summary import parse_day, render_markdown, save as save_summary, stored
from claireai.tools import default_registry

TOKEN_FILE = "api.token"


@dataclass(frozen=True)
class Identity:
    kind: str
    id: str
    name: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32_000)
    conversation_id: str | None = None
    use_tools: bool = True


class ChatResponse(BaseModel):
    conversation_id: str
    task_id: str
    content: str
    model: str
    duration_ms: int
    route: str | None = None


class TaskCreateRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32_000)
    conversation_id: str | None = None
    use_tools: bool = True


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=5, ge=1, le=10)


class MemorySearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    text: str
    model: str


class PairRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    ttl_seconds: int = Field(default=300, ge=30, le=900)


class ClaimRequest(BaseModel):
    code: str = Field(min_length=4, max_length=40)
    name: str | None = Field(default=None, max_length=80)


class EventBus:
    """Small in-process SSE fan-out; no event is persisted here."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def publish(self, event_type: str, **data: Any) -> None:
        event = {"type": event_type, "ts": utc_now(), **data}
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def stream(self) -> AsyncIterator[str]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        try:
            yield "event: ready\ndata: {\"type\":\"ready\"}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield (
                    f"event: {event['type']}\n"
                    f"data: {json.dumps(event, sort_keys=True)}\n\n"
                )
        finally:
            self._subscribers.discard(queue)


class TaskRunner:
    """Run API-created tasks in the background without sharing SQLite handles."""

    def __init__(self, app: FastAPI, settings: Settings) -> None:
        self.app = app
        self.settings = settings
        self.jobs: dict[str, asyncio.Task[None]] = {}

    def submit(
        self,
        *,
        task_id: str,
        conversation_id: str,
        message: str,
        use_tools: bool,
        device: str,
    ) -> None:
        job = asyncio.create_task(
            self._run(
                task_id=task_id,
                conversation_id=conversation_id,
                message=message,
                use_tools=use_tools,
                device=device,
            ),
            name=f"claireai-api-{task_id[:8]}",
        )
        self.jobs[task_id] = job
        job.add_done_callback(
            lambda _job, tid=task_id: self.jobs.pop(tid, None)
        )

    def _mark(
        self, conn, task_id: str, status: str, *, error: str | None = None
    ) -> None:
        repository = TaskRepository(conn)
        task = repository.get(task_id)
        if task is None or task.status in TERMINAL_STATUSES:
            return
        try:
            repository.update_status(task_id, status, error=error)
        except (ValueError, LookupError):
            pass

    async def _run(
        self,
        *,
        task_id: str,
        conversation_id: str,
        message: str,
        use_tools: bool,
        device: str,
    ) -> None:
        conn = Database(self.settings.database_path).connect()
        brain = None
        try:
            brain = build_brain(self.settings)
            control = RuntimeControl(self.settings.control_state_path).read()
            self.app.state.events.publish(
                "task.started", task_id=task_id, conversation_id=conversation_id
            )
            if use_tools:
                outcome = await run_task(
                    settings=self.settings,
                    conn=conn,
                    brain=brain,
                    request=message,
                    registry=default_registry(
                        browser_enabled=bool(control["browser"]),
                        memory_enabled=self.settings.memory_enabled,
                        capabilities=self.settings.allowed_capabilities,
                    ),
                    approvals=FileApprovalBroker(self.settings.resolved_data_dir),
                    device=device,
                    conversation_id=conversation_id,
                    existing_task_id=task_id,
                )
                content, model = outcome.content, outcome.model
            else:
                turn = await ask(
                    settings=self.settings,
                    conn=conn,
                    brain=brain,
                    question=message,
                    device=device,
                    conversation_id=conversation_id,
                    existing_task_id=task_id,
                )
                content, model = turn.content, turn.model
            self.app.state.events.publish(
                "task.completed",
                task_id=task_id,
                conversation_id=conversation_id,
                model=model,
                content=content,
            )
        except asyncio.CancelledError:
            self._mark(conn, task_id, "cancelled", error="cancelled by API")
            self.app.state.events.publish("task.cancelled", task_id=task_id)
            raise
        except Exception as exc:  # noqa: BLE001 - the task row is the report
            self._mark(conn, task_id, "failed", error=str(exc))
            self.app.state.events.publish(
                "task.failed", task_id=task_id, error=str(exc)
            )
        finally:
            if brain is not None:
                await brain.aclose()
            conn.close()

    def cancel(self, task_id: str) -> bool:
        conn = Database(self.settings.database_path).connect()
        try:
            task = TaskRepository(conn).get(task_id)
            if task is None:
                raise LookupError("task does not exist")
            if task.status in TERMINAL_STATUSES:
                return False
            job = self.jobs.get(task_id)
            if job is not None and job is not asyncio.current_task():
                job.cancel()
            self._mark(conn, task_id, "cancelled", error="cancelled by API")
            self.app.state.events.publish("task.cancelled", task_id=task_id)
            return True
        finally:
            conn.close()


def load_or_create_token(data_dir: Path) -> str:
    """Use the local token file, creating it with owner-only permissions once."""
    path = Path(data_dir) / TOKEN_FILE
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    try:
        handle = os.fdopen(
            os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
        )
    except FileExistsError:
        return path.read_text(encoding="utf-8").strip()
    with handle:
        handle.write(token)
    return token


def _task_dict(task) -> dict[str, Any]:  # noqa: ANN001
    return {
        "id": task.id,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "finished_at": task.finished_at,
        "device": task.device,
        "actor": task.actor,
        "kind": task.kind,
        "request": task.request,
        "status": task.status,
        "current_step": task.current_step,
        "error": task.error,
    }


def _audit_dict(entry) -> dict[str, Any]:  # noqa: ANN001
    return {
        "id": entry.id,
        "created_at": entry.created_at,
        "device": entry.device,
        "task_id": entry.task_id,
        "actor": entry.actor,
        "tool": entry.tool,
        "action": entry.action,
        "target": entry.target,
        "status": entry.status,
        "approval_ref": entry.approval_ref,
        "detail": entry.detail,
    }


def _device_label(identity: Identity) -> str:
    return "api" if identity.kind == "admin" else f"api:{identity.id}"


def create_app(
    settings: Settings | None = None,
    *,
    token: str | None = None,
) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_directories()
    expected = token or settings.api_token_value or load_or_create_token(
        settings.resolved_data_dir
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        Database(settings.database_path).migrate()
        yield
        jobs = list(getattr(_app.state.runner, "jobs", {}).values())
        for job in jobs:
            job.cancel()
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)

    app = FastAPI(
        title="ClaireAI local API",
        version="0.2.0",
        description=(
            "Local assistant backend. Every route except /health and device "
            "claiming needs a bearer token."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.token = expected
    app.state.events = EventBus()
    app.state.jobs = {}
    app.state.runner = TaskRunner(app, settings)

    def connection():
        return Database(settings.database_path).connect()

    def authorize(authorization: str | None = Header(default=None)) -> Identity:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        supplied = authorization.removeprefix("Bearer ").strip()
        if hmac.compare_digest(supplied, expected):
            return Identity(kind="admin", id="admin", name="administrator")
        conn = connection()
        try:
            device = device_for_token(conn, supplied)
        finally:
            conn.close()
        if device is not None:
            return Identity(kind="device", id=device["id"], name=device["name"])
        raise HTTPException(status_code=401, detail="invalid bearer token")

    def require_admin(identity: Identity = Depends(authorize)) -> Identity:
        if identity.kind != "admin":
            raise HTTPException(status_code=403, detail="administrator token required")
        return identity

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "claireai-api"}

    @app.get("/v1/status", dependencies=[Depends(authorize)])
    def status() -> dict[str, Any]:
        conn = connection()
        try:
            memory = None
            tasks = 0
            devices = 0
            try:
                memory = int(
                    conn.execute("SELECT COUNT(*) AS n FROM memory_entries").fetchone()["n"]
                )
                tasks = int(conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"])
                devices = int(
                    conn.execute(
                        "SELECT COUNT(*) AS n FROM api_devices WHERE revoked_at IS NULL"
                    ).fetchone()["n"]
                )
            except Exception:
                memory = 0
            return {
                "provider": settings.brain_provider,
                "model": settings.brain_model,
                "coding_model": settings.brain_coding_model,
                "memory_enabled": bool(settings.memory_enabled),
                "memory_entries": memory,
                "tasks": tasks,
                "devices": devices,
                "capabilities": list(settings.allowed_capabilities),
                "browser": bool(
                    RuntimeControl(settings.control_state_path).read()["browser"]
                ),
            }
        finally:
            conn.close()

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> dict[str, Any]:
        brain = build_brain(settings)
        try:
            health_report = await brain.health()
            return {
                "provider": settings.brain_provider,
                "model": settings.brain_model,
                "coding_model": settings.brain_coding_model,
                "available": health_report.available,
                "model_present": health_report.model_present,
                "detail": health_report.detail,
            }
        finally:
            await brain.aclose()

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest, identity: Identity = Depends(authorize)
    ) -> ChatResponse:
        message = payload.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="message must not be empty")
        conn = connection()
        brain = build_brain(settings)
        device = _device_label(identity)
        try:
            control = RuntimeControl(settings.control_state_path).read()
            app.state.events.publish("chat.started", device=device)
            if payload.use_tools:
                conversation_id = payload.conversation_id
                if conversation_id is None:
                    conversation_id = ConversationRepository(conn).create()
                outcome = await run_task(
                    settings=settings,
                    conn=conn,
                    brain=brain,
                    request=message,
                    registry=default_registry(
                        browser_enabled=bool(control["browser"]),
                        memory_enabled=settings.memory_enabled,
                        capabilities=settings.allowed_capabilities,
                    ),
                    approvals=FileApprovalBroker(settings.resolved_data_dir),
                    device=device,
                    conversation_id=conversation_id,
                )
                response = ChatResponse(
                    conversation_id=conversation_id,
                    task_id=outcome.task_id,
                    content=outcome.content,
                    model=outcome.model,
                    duration_ms=outcome.duration_ms,
                    route=outcome.route,
                )
            else:
                turn = await ask(
                    settings=settings,
                    conn=conn,
                    brain=brain,
                    question=message,
                    device=device,
                    conversation_id=payload.conversation_id,
                )
                response = ChatResponse(
                    conversation_id=turn.conversation_id,
                    task_id=turn.task_id,
                    content=turn.content,
                    model=turn.model,
                    duration_ms=turn.duration_ms,
                )
            app.state.events.publish(
                "chat.completed",
                task_id=response.task_id,
                conversation_id=response.conversation_id,
                model=response.model,
            )
            return response
        except AgentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BrainError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await brain.aclose()
            conn.close()

    @app.post("/v1/tasks")
    async def create_task(
        payload: TaskCreateRequest, identity: Identity = Depends(authorize)
    ) -> dict[str, Any]:
        message = payload.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="message must not be empty")
        conn = connection()
        try:
            with transaction(conn):
                conversations = ConversationRepository(conn)
                conversation_id = payload.conversation_id
                if conversation_id is None:
                    conversation_id = conversations.create()
                elif not conversations.exists(conversation_id):
                    raise ValueError("conversation does not exist")
                task = TaskRepository(conn).create(
                    request=redact(message),
                    kind="agent" if payload.use_tools else "chat",
                    device=_device_label(identity),
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            conn.close()
        app.state.runner.submit(
            task_id=task.id,
            conversation_id=conversation_id,
            message=message,
            use_tools=payload.use_tools,
            device=_device_label(identity),
        )
        app.state.events.publish(
            "task.queued", task_id=task.id, conversation_id=conversation_id
        )
        return task.__dict__ | {"conversation_id": conversation_id}

    @app.get("/v1/tasks", dependencies=[Depends(authorize)])
    def tasks(limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= limit <= 200:
            raise HTTPException(status_code=422, detail="limit must be 1-200")
        conn = connection()
        try:
            return [_task_dict(task) for task in TaskRepository(conn).recent(limit)]
        finally:
            conn.close()

    @app.get("/v1/tasks/{task_id}", dependencies=[Depends(authorize)])
    def task(task_id: str) -> dict[str, Any]:
        conn = connection()
        try:
            repository = TaskRepository(conn)
            found = repository.get(task_id)
            if found is None:
                raise HTTPException(status_code=404, detail="task does not exist")
            return {
                **_task_dict(found),
                "audit": [
                    _audit_dict(entry) for entry in AuditLog(conn).for_task(task_id)
                ],
            }
        finally:
            conn.close()

    @app.post("/v1/tasks/{task_id}/cancel", dependencies=[Depends(authorize)])
    def cancel_task(task_id: str) -> dict[str, Any]:
        try:
            cancelled = app.state.runner.cancel(task_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not cancelled:
            raise HTTPException(status_code=409, detail="task is already finished")
        conn = connection()
        try:
            found = TaskRepository(conn).get(task_id)
            return _task_dict(found) if found else {"id": task_id, "status": "cancelled"}
        finally:
            conn.close()

    @app.get("/v1/conversations", dependencies=[Depends(authorize)])
    def conversations() -> list[dict[str, Any]]:
        conn = connection()
        try:
            return ConversationRepository(conn).recent()
        finally:
            conn.close()

    @app.get("/v1/conversations/{conversation_id}", dependencies=[Depends(authorize)])
    def conversation(conversation_id: str) -> dict[str, Any]:
        conn = connection()
        try:
            repository = ConversationRepository(conn)
            if not repository.exists(conversation_id):
                raise HTTPException(status_code=404, detail="conversation does not exist")
            messages = repository.messages(conversation_id)
            return {
                "conversation_id": conversation_id,
                "messages": [
                    {
                        "role": item.role,
                        "content": item.content,
                        "created_at": item.created_at,
                        "task_id": item.task_id,
                    }
                    for item in messages
                ],
            }
        finally:
            conn.close()

    @app.post(
        "/v1/memory/search",
        response_model=MemorySearchResponse,
        dependencies=[Depends(authorize)],
    )
    def memory_search(payload: MemorySearchRequest) -> MemorySearchResponse:
        if not settings.memory_enabled:
            raise HTTPException(status_code=503, detail="local memory is disabled")
        conn = connection()
        try:
            from claireai.memory import MemoryIndex

            index = MemoryIndex(conn, embedder_for(settings))
            hits = index.search(payload.query, limit=payload.limit)
            return MemorySearchResponse(
                query=payload.query,
                text=render_hits(payload.query, hits),
                model=index.embedder.name,
                results=[
                    {
                        "id": hit.id,
                        "text": hit.text,
                        "score": round(hit.score, 4),
                        "created_at": hit.created_at,
                    }
                    for hit in hits
                ],
            )
        except (EmbeddingError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            conn.close()

    @app.get("/v1/memory", dependencies=[Depends(authorize)])
    def memory(limit: int = 100) -> dict[str, Any]:
        if not settings.memory_enabled:
            raise HTTPException(status_code=503, detail="local memory is disabled")
        if not 1 <= limit <= 200:
            raise HTTPException(status_code=422, detail="limit must be 1-200")
        conn = connection()
        try:
            from claireai.memory import memory_index

            index = memory_index(settings, conn)
            if index is None:
                return {"entries": [], "count": 0, "embedder": embedder_for(settings).name}
            entries = index.list_entries(limit=limit)
            return {
                "entries": [
                    {
                        "id": hit.id,
                        "created_at": hit.created_at,
                        "kind": hit.kind,
                        "source_id": hit.source_id,
                        "text": hit.text,
                        "model": hit.model,
                    }
                    for hit in entries
                ],
                "count": index.count(),
                "embedder": index.embedder.name,
            }
        finally:
            conn.close()

    @app.delete("/v1/memory/{entry_id}", dependencies=[Depends(authorize)])
    def delete_memory(entry_id: str) -> dict[str, Any]:
        conn = connection()
        try:
            from claireai.memory import memory_index

            index = memory_index(settings, conn)
            if index is None or not index.delete(entry_id):
                raise HTTPException(status_code=404, detail="memory entry does not exist")
            return {"id": entry_id, "deleted": True}
        finally:
            conn.close()

    @app.get("/v1/approvals")
    def approvals(_identity: Identity = Depends(require_admin)) -> list[dict[str, Any]]:
        return pending_approvals(settings.resolved_data_dir)

    @app.post("/v1/approvals/{ref}/deny")
    def deny_approval(ref: str, _identity: Identity = Depends(require_admin)) -> dict[str, Any]:
        if not revoke_approval(settings.resolved_data_dir, ref):
            raise HTTPException(status_code=404, detail="approval is not pending")
        conn = connection()
        try:
            with transaction(conn):
                AuditLog(conn).record(
                    action="approval.denied",
                    status="skipped",
                    device="api",
                    actor="owner",
                    approval_ref=ref,
                    detail={"reason": "denied from local API"},
                )
        finally:
            conn.close()
        return {"ref": ref, "denied": True}

    @app.post("/v1/approvals/{ref}/approve")
    def approve_approval(ref: str, _identity: Identity = Depends(require_admin)) -> None:
        raise HTTPException(
            status_code=403,
            detail=(
                "an API token cannot approve an action; issue a signed one-action "
                "token from a terminal with claireai --preauthorize"
            ),
        )

    @app.post("/v1/approvals/{ref}/expire")
    def expire_approval(ref: str, _identity: Identity = Depends(require_admin)) -> dict[str, Any]:
        if not revoke_approval(settings.resolved_data_dir, ref):
            raise HTTPException(status_code=404, detail="approval is not pending")
        return {"ref": ref, "expired": True}

    @app.get("/v1/audit")
    def audit(limit: int = 50, _identity: Identity = Depends(require_admin)) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500:
            raise HTTPException(status_code=422, detail="limit must be 1-500")
        conn = connection()
        try:
            return [_audit_dict(entry) for entry in AuditLog(conn).recent(limit)]
        finally:
            conn.close()

    @app.get("/v1/summary")
    def summary(day: str | None = None, _identity: Identity = Depends(require_admin)) -> dict[str, Any]:
        try:
            selected = parse_day(day)
        except Exception as exc:  # SummaryError is a ValueError subclass
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        conn = connection()
        try:
            content = stored(conn, selected)
            is_stored = content is not None
            if content is None:
                content = render_markdown(build_summary(conn, day=selected))
            return {"day": selected.isoformat(), "stored": is_stored, "content": content}
        finally:
            conn.close()

    @app.post("/v1/summary")
    def regenerate_summary(
        day: str | None = None, _identity: Identity = Depends(require_admin)
    ) -> dict[str, Any]:
        try:
            selected = parse_day(day)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        conn = connection()
        try:
            report = build_summary(conn, day=selected)
            with transaction(conn):
                path = save_summary(conn, report, data_dir=settings.resolved_data_dir)
            return {
                "day": selected.isoformat(),
                "stored": True,
                "path": str(path),
                "content": render_markdown(report),
            }
        finally:
            conn.close()

    @app.get("/v1/settings")
    def read_settings(_identity: Identity = Depends(require_admin)) -> dict[str, Any]:
        conn = connection()
        try:
            return PreferenceRepository(conn).effective(settings)
        finally:
            conn.close()

    @app.put("/v1/settings")
    def write_settings(
        values: dict[str, Any], _identity: Identity = Depends(require_admin)
    ) -> dict[str, Any]:
        from claireai.storage.user_data import EDITABLE_PREFERENCES

        unknown = sorted(set(values) - EDITABLE_PREFERENCES)
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"settings cannot be changed through this endpoint: {', '.join(unknown)}",
            )
        conn = connection()
        try:
            with transaction(conn):
                PreferenceRepository(conn).update(values)
            return PreferenceRepository(conn).effective(settings)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            conn.close()

    @app.get("/v1/devices")
    def devices(_identity: Identity = Depends(require_admin)) -> list[dict[str, Any]]:
        conn = connection()
        try:
            return list_devices(conn)
        finally:
            conn.close()

    @app.post("/v1/devices/pair")
    def pair(
        payload: PairRequest, _identity: Identity = Depends(require_admin)
    ) -> dict[str, Any]:
        conn = connection()
        try:
            return create_pairing(conn, name=payload.name, ttl_seconds=payload.ttl_seconds)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            conn.close()

    @app.post("/v1/devices/claim")
    def claim(payload: ClaimRequest) -> dict[str, Any]:
        conn = connection()
        try:
            device, _token = claim_pairing(conn, code=payload.code, name=payload.name)
            return device
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            conn.close()

    @app.delete("/v1/devices/{device_id}")
    def remove_device(
        device_id: str, _identity: Identity = Depends(require_admin)
    ) -> dict[str, Any]:
        conn = connection()
        try:
            if not revoke_device(conn, device_id):
                raise HTTPException(status_code=404, detail="device does not exist")
            return {"id": device_id, "revoked": True}
        finally:
            conn.close()

    @app.get("/v1/events")
    async def events(identity: Identity = Depends(authorize)) -> StreamingResponse:
        return StreamingResponse(
            app.state.events.stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
