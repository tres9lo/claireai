"""Local semantic memory: embeddings plus cosine search in SQLite.

The default embedder is Ollama's ``/api/embed`` endpoint with a small embedding
model (``nomic-embed-text`` by default). That endpoint is loopback and needs no
API key. A deterministic hashing embedder is available as an explicit offline
provider for tests and machines that cannot run the model; it is not silently
mixed with model vectors.

Memory is written when the owner invokes ``remember_note`` (which needs approval),
when code explicitly adds an entry, or when the owner opts into
``memory_auto_embed``. Ordinary chat text is otherwise kept only as conversation
history under the normal retention rules.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import struct
import uuid
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import httpx

from claireai.config import Settings
from claireai.storage.database import transaction
from claireai.storage.tasks import utc_now

MAX_NOTE_CHARS = 8_000
MAX_QUERY_CHARS = 2_000
MAX_RESULTS = 10
MIN_SCORE = 0.05


class EmbeddingError(RuntimeError):
    """The local embedding model could not produce a vector."""


class Embedder(Protocol):
    name: str
    dimensions: int | None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


def _normalise(vector: Iterable[float]) -> list[float]:
    values = [float(item) for item in vector]
    if not values or any(not math.isfinite(item) for item in values):
        raise EmbeddingError("the embedding model returned an empty or non-finite vector")
    length = math.sqrt(sum(item * item for item in values))
    if length == 0:
        raise EmbeddingError("the embedding model returned a zero vector")
    return [item / length for item in values]


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes, dimensions: int) -> list[float]:
    if dimensions <= 0 or len(blob) != dimensions * 4:
        raise EmbeddingError("a stored vector has the wrong size")
    return list(struct.unpack(f"<{dimensions}f", blob))


class OllamaEmbedder:
    """Batch embeddings from the local Ollama daemon."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.model = str(model or "").strip()
        self.timeout = float(timeout)
        self._client = client
        self.dimensions: int | None = None
        if not self.base_url or not self.model:
            raise ValueError("Ollama embeddings need a base URL and a model")

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = [str(text) for text in texts]
        if not items:
            return []
        payload = {"model": self.model, "input": items}
        try:
            if self._client is not None:
                response = self._client.post(
                    f"{self.base_url}/embed", json=payload, timeout=self.timeout
                )
            else:
                with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                    response = client.post(f"{self.base_url}/embed", json=payload)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError(
                f"the local embedding model {self.model!r} is unavailable; "
                f"run `ollama pull {self.model}` and check Ollama is running"
            ) from exc

        embeddings = body.get("embeddings") if isinstance(body, dict) else None
        if not isinstance(embeddings, list):
            # Older Ollama releases returned one ``embedding`` for one input.
            single = body.get("embedding") if isinstance(body, dict) else None
            embeddings = [single] if isinstance(single, list) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(items):
            raise EmbeddingError("the embedding endpoint returned the wrong number of vectors")
        vectors = [_normalise(vector) for vector in embeddings if isinstance(vector, list)]
        if len(vectors) != len(items):
            raise EmbeddingError("the embedding endpoint returned a malformed vector")
        self.dimensions = len(vectors[0])
        return vectors


class HashingEmbedder:
    """Deterministic local feature hashing, used explicitly as an offline mode."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 32:
            raise ValueError("hashing dimensions must be at least 32")
        self.dimensions = int(dimensions)

    @property
    def name(self) -> str:
        return f"hash:{self.dimensions}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            tokens = re.findall(r"[a-z0-9_'-]+", str(text).lower())
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                number = int.from_bytes(digest, "little")
                index = number % self.dimensions
                sign = -1.0 if number & 1 else 1.0
                vector[index] += sign
            vectors.append(_normalise(vector))
        return vectors


def embedder_for(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashingEmbedder()
    return OllamaEmbedder(
        base_url=settings.resolved_embedding_base_url,
        model=settings.embedding_model,
        timeout=settings.embedding_timeout_seconds,
    )


@dataclass(frozen=True)
class MemoryHit:
    id: str
    text: str
    kind: str
    source_id: str | None
    created_at: str
    score: float
    model: str


class MemoryIndex:
    """Small exact cosine index over the local ``memory_entries`` table."""

    def __init__(self, conn: sqlite3.Connection, embedder: Embedder) -> None:
        self.conn = conn
        self.embedder = embedder

    def add(
        self,
        text: str,
        *,
        kind: str = "note",
        source_id: str | None = None,
    ) -> str:
        clean = " ".join(str(text).split()).strip()
        if not clean:
            raise ValueError("memory text must not be empty")
        if len(clean) > MAX_NOTE_CHARS:
            raise ValueError(f"memory text exceeds {MAX_NOTE_CHARS} characters")
        if kind not in {"note", "turn"}:
            raise ValueError("memory kind must be note or turn")
        if source_id is not None and not str(source_id).strip():
            raise ValueError("source_id must not be empty when provided")
        vector = self.embedder.embed([clean])[0]
        dimensions = len(vector)
        entry_id = uuid.uuid4().hex
        with transaction(self.conn):
            if source_id is not None:
                self.conn.execute(
                    "DELETE FROM memory_entries WHERE kind = ? AND source_id = ?",
                    (kind, str(source_id)),
                )
            self.conn.execute(
                "INSERT INTO memory_entries "
                "(id, created_at, kind, source_id, text, embedding, dimensions, model) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_id,
                    utc_now(),
                    kind,
                    str(source_id) if source_id is not None else None,
                    clean,
                    _pack(vector),
                    dimensions,
                    self.embedder.name,
                ),
            )
        return entry_id

    def search(self, query: str, *, limit: int = 5) -> list[MemoryHit]:
        clean = " ".join(str(query).split()).strip()
        if not clean:
            return []
        if len(clean) > MAX_QUERY_CHARS:
            raise ValueError(f"memory query exceeds {MAX_QUERY_CHARS} characters")
        if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
            raise ValueError(f"limit must be from 1 to {MAX_RESULTS}")
        query_vector = self.embedder.embed([clean])[0]
        dimensions = len(query_vector)
        rows = self.conn.execute(
            "SELECT id, created_at, kind, source_id, text, embedding, dimensions, model "
            "FROM memory_entries WHERE dimensions = ? AND model = ?",
            (dimensions, self.embedder.name),
        ).fetchall()
        hits: list[MemoryHit] = []
        for row in rows:
            try:
                vector = _unpack(row["embedding"], int(row["dimensions"]))
            except EmbeddingError:
                continue
            score = sum(a * b for a, b in zip(query_vector, vector))
            if score < MIN_SCORE:
                continue
            hits.append(
                MemoryHit(
                    id=row["id"],
                    text=row["text"],
                    kind=row["kind"],
                    source_id=row["source_id"],
                    created_at=row["created_at"],
                    score=score,
                    model=row["model"],
                )
            )
        hits.sort(key=lambda item: (item.score, item.created_at), reverse=True)
        return hits[:limit]

    def count(self) -> int:
        return int(
            self.conn.execute("SELECT COUNT(*) AS n FROM memory_entries").fetchone()["n"]
        )

    def list_entries(
        self, *, limit: int = 50, offset: int = 0, kind: str | None = None
    ) -> list[MemoryHit]:
        """Newest local entries, for the owner-facing memory manager."""
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("limit must be from 1 to 200")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be zero or greater")
        query = (
            "SELECT id, created_at, kind, source_id, text, dimensions, model "
            "FROM memory_entries"
        )
        params: list[object] = []
        if kind is not None:
            if kind not in {"note", "turn"}:
                raise ValueError("memory kind must be note or turn")
            query += " WHERE kind = ?"
            params.append(kind)
        query += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend((limit, offset))
        return [
            MemoryHit(
                id=row["id"],
                text=row["text"],
                kind=row["kind"],
                source_id=row["source_id"],
                created_at=row["created_at"],
                score=0.0,
                model=row["model"],
            )
            for row in self.conn.execute(query, params)
        ]

    def delete(self, entry_id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
        return cursor.rowcount == 1


def remember_completed_turn(
    settings: Settings,
    conn: sqlite3.Connection,
    *,
    task_id: str,
    question: str,
    answer: str,
) -> str | None:
    """Best-effort local embedding of a completed chat turn.

    This is opt-in through ``memory_auto_embed``. A missing Ollama model must
    never turn a delivered answer into a failed turn, so callers treat errors as
    a skipped memory write.
    """
    if not settings.memory_enabled or not settings.memory_auto_embed:
        return None
    text = f"User: {question.strip()}\nAssistant: {answer.strip()}".strip()
    if not text:
        return None
    return MemoryIndex(conn, embedder_for(settings)).add(
        text[:MAX_NOTE_CHARS], kind="turn", source_id=task_id
    )


def memory_index(settings: Settings, conn: sqlite3.Connection) -> MemoryIndex | None:
    """Build the index for a run, or return None when memory is disabled."""
    if not settings.memory_enabled:
        return None
    return MemoryIndex(conn, embedder_for(settings))


def render_hits(query: str, hits: Sequence[MemoryHit]) -> str:
    if not hits:
        return f"No stored memory matched {query!r}."
    lines = [f"Stored memory matching {query!r}:"]
    for index, hit in enumerate(hits, start=1):
        age = hit.created_at[:10]
        lines.append(f"{index}. [{age}, {hit.score:.2f}] {hit.text}")
    return "\n".join(lines)
