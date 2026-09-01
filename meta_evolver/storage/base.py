"""The storage contract, and the URL that picks a backend.

A memory bank is state that outlives a run, is written concurrently, and is
queried by similarity. A JSONL file handles exactly one of those. The other
two are why this layer exists:

**Concurrency.** The evolution graph fans rollouts out with ``Send``, and
credit assignment is a read-modify-write over shared counters. Two processes
rewriting one JSONL file lose updates silently -- and the loss looks like "the
memory bank stopped improving", not like corruption. A store makes usage an
atomic increment.

**Similarity search.** Scoring every memory in Python is fine at a hundred
items and absurd at a hundred thousand. A backend that can rank vectors
server-side says so via :attr:`MemoryStore.supports_vector_search`, and the
bank hands it the candidate query instead.

Backends are chosen by URL, so the same code runs against a file in CI and
Postgres in production:

===============================  ==========================================
``memories.jsonl``               file (default; no dependencies)
``postgresql://user@host/db``    Postgres + pgvector, server-side ANN
``mongodb://host/db``            MongoDB (Atlas ``$vectorSearch`` if present)
===============================  ==========================================

Every backend degrades rather than failing: a store that cannot rank vectors
returns ``None`` from :meth:`search` and the bank scores in Python, which is
the same path the file backend has always taken.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar
from urllib.parse import urlparse

from meta_evolver.core.types import MemoryItem

#: Read when no URL is passed explicitly, so a deployment can point every
#: entry point at one database without touching a command line.
DB_URL_ENV = "META_EVOLVER_DB_URL"


class StoreError(RuntimeError):
    """Raised when a backend is unreachable or misconfigured.

    Deliberately distinct from a missing-driver ``ImportError``: one means
    "install the extra", the other means "the server is down", and the fix
    differs.
    """


class MemoryStore(ABC):
    """Persistence for :class:`MemoryItem`.

    Implementations must be safe to call from several processes at once.
    ``record_usage`` in particular has to be atomic per item, because it is on
    the concurrent path and a lost increment silently corrupts the utility
    signal that drives pruning.
    """

    #: True when :meth:`search` can rank server-side. False means the bank
    #: loads and scores in Python.
    supports_vector_search: ClassVar[bool] = False

    #: Human-readable, for logs and ``meta-evolver`` output.
    describe: str = "unknown"

    @abstractmethod
    def load(self) -> list[MemoryItem]:
        """Every item in this namespace."""

    @abstractmethod
    def upsert(self, items: Sequence[MemoryItem]) -> None:
        """Insert or replace by id."""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Remove by id. Missing ids are not an error."""

    @abstractmethod
    def record_usage(self, events: Sequence[tuple[str, bool]]) -> None:
        """Apply ``(memory_id, succeeded)`` outcomes as atomic increments.

        Not a read-modify-write: concurrent rollouts credit the same memory,
        and a lost increment biases the utility that decides what gets pruned.
        """

    def search(
        self, vector: Sequence[float], k: int, min_similarity: float = 0.0
    ) -> list[tuple[MemoryItem, float]] | None:
        """Server-side nearest neighbours, or ``None`` if unsupported.

        Returns more than ``k`` is fine and often better: the bank re-ranks
        with MMR, which needs candidates to choose diversity from.
        """
        return None

    def count(self) -> int:
        return len(self.load())

    def close(self) -> None:
        return None

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_store(
    url: str | os.PathLike[str] | None = None,
    namespace: str = "default",
    dim: int | None = None,
    **kwargs: object,
) -> MemoryStore:
    """Build the backend named by ``url``.

    ``namespace`` scopes a benchmark's memories, so one database can serve
    several without them retrieving each other's lessons.

    ``dim`` is the embedding width the vector index is built for. It matters
    only to Postgres, where the column type is fixed at DDL time.
    """
    from meta_evolver.llm.client import DEFAULT_EMBED_DIMENSIONS, load_dotenv_once

    load_dotenv_once()
    raw = str(url) if url is not None else os.environ.get(DB_URL_ENV, "")
    width = DEFAULT_EMBED_DIMENSIONS if dim is None else dim

    scheme = urlparse(raw).scheme.lower() if "://" in raw else ""

    if scheme in {"postgresql", "postgres", "psql"}:
        from meta_evolver.storage.postgres import PostgresMemoryStore

        return PostgresMemoryStore(raw, namespace=namespace, dim=width, **kwargs)

    if scheme in {"mongodb", "mongodb+srv"}:
        from meta_evolver.storage.mongo import MongoMemoryStore

        return MongoMemoryStore(raw, namespace=namespace, dim=width, **kwargs)

    from meta_evolver.storage.jsonl import JsonlMemoryStore

    path = raw[len("jsonl://") :] if scheme == "jsonl" else raw
    return JsonlMemoryStore(path or "memories.jsonl", namespace=namespace, **kwargs)


def redact(url: str) -> str:
    """A connection string safe to print: credentials replaced.

    Connection strings end up in logs, run summaries and error messages. One
    with a password in it ends up in all three.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    user = f"{parsed.username}:***@" if parsed.username else ""
    return f"{parsed.scheme}://{user}{host}{parsed.path}"
