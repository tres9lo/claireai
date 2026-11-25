"""A small, authenticated HTTP surface for the existing runtime.

It is a backend, not a second assistant: every chat turn goes through ``ask`` or
``run_task``, so the folder allow-list, risk policy, approvals and audit log are
the same ones the window uses. Write tools use ``FileApprovalBroker`` and
therefore still need a signed token issued from a terminal; a bearer token alone
cannot approve a file write.

Bind loopback by default. The token is read from ``CLAIREAI_API_TOKEN`` or a
0600 ``api.token`` file in the data directory. Only ``/health`` is unauthenticated.
"""

from __future__ import annotations

import hmac
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from claireai.agent import AgentError, run_task
from claireai.approval_file import FileApprovalBroker
from claireai.brain import BrainError, build_brain
from claireai.cli import ask
from claireai.config import Settings
from claireai.memory import EmbeddingError, embedder_for, render_hits
from claireai.runtime_control import RuntimeControl
from claireai.storage import ConversationRepository, Database
from claireai.tools import default_registry

TOKEN_FILE = "api.token"


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


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=5, ge=1, le=10)


class MemorySearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    text: str
    model: str


def create_app(
    settings: Settings | None = None,
    *,
    token: str | None = None,
) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_directories()
    expected = token or settings.api_token_value or load_or_create_token(settings.resolved_data_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        Database(settings.database_path).migrate()
        yield

    app = FastAPI(
        title="ClaireAI local API",
        version="0.1.0",
        description="Local assistant backend. Every route except /health needs a bearer token.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.token = expected

    def connection():
        return Database(settings.database_path).connect()

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        supplied = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="invalid bearer token")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "claireai-api"}

    @app.get("/v1/status", dependencies=[Depends(authorize)])
    def status() -> dict[str, Any]:
        conn = connection()
        try:
            memory = None
            try:
                row = conn.execute("SELECT COUNT(*) AS n FROM memory_entries").fetchone()
                memory = int(row["n"])
            except Exception:
                memory = 0
            return {
                "provider": settings.brain_provider,
                "model": settings.brain_model,
                "coding_model": settings.brain_coding_model,
                "memory_enabled": bool(settings.memory_enabled),
                "memory_entries": memory,
                "capabilities": list(settings.allowed_capabilities),
                "browser": bool(RuntimeControl(settings.control_state_path).read()["browser"]),
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

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(authorize)])
    async def chat(payload: ChatRequest) -> ChatResponse:
        message = payload.message.strip()
        if not message:
            raise HTTPException(status_code=422, detail="message must not be empty")
        conn = connection()
        brain = build_brain(settings)
        try:
            control = RuntimeControl(settings.control_state_path).read()
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
                    device="api",
                    conversation_id=conversation_id,
                )
                return ChatResponse(
                    conversation_id=conversation_id,
                    task_id=outcome.task_id,
                    content=outcome.content,
                    model=outcome.model,
                    duration_ms=outcome.duration_ms,
                    route=outcome.route,
                )
            turn = await ask(
                settings=settings,
                conn=conn,
                brain=brain,
                question=message,
                device="api",
                conversation_id=payload.conversation_id,
            )
            return ChatResponse(
                conversation_id=turn.conversation_id,
                task_id=turn.task_id,
                content=turn.content,
                model=turn.model,
                duration_ms=turn.duration_ms,
            )
        except AgentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BrainError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await brain.aclose()
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
            if not ConversationRepository(conn).exists(conversation_id):
                raise HTTPException(status_code=404, detail="conversation does not exist")
            messages = ConversationRepository(conn).messages(conversation_id)
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

    return app
