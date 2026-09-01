"""LangGraph checkpointers -- making a run survive the process that started it.

A checkpointer persists graph state after every superstep. That is what turns
a long evolution run from something you must not interrupt into something you
can: kill it in generation four, restart, and it resumes from the last
completed step rather than from the base prompt.

It also enables what a checkpointed graph gets for free -- inspecting an
episode mid-flight, time-travelling to an earlier superstep, and
``interrupt()`` for human review before a decision is committed.

Two backends:

``InMemorySaver``
    In-process. Gives resumability *within* one run and costs nothing, which
    is why it is the default rather than "no checkpointer at all".

``AsyncPostgresSaver``
    Durable and shared. Requires ``[postgres]`` and a reachable server; it
    creates its own tables on first use, alongside the memory bank's.

The **async** saver specifically. LangGraph's sync ``PostgresSaver`` raises a
bare ``NotImplementedError`` when a graph is driven with ``ainvoke`` -- and
because a node's exception becomes an episode error rather than a crash, the
symptom is every rollout failing with an empty message. This module only ever
hands back a saver that matches how these graphs are actually run.

Threads are the unit of resumption. Ids are derived deterministically from the
run, phase, generation and task, so a resumed run addresses the same threads it
created; a random id would leave the checkpoints present but unreachable.
"""
from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from meta_evolver.storage.base import StoreError, redact

_POSTGRES_SCHEMES = {"postgresql", "postgres", "psql"}


def thread_id(*parts: Any) -> str:
    """A stable thread id for a run/phase/generation/task tuple."""
    payload = "|".join(str(p) for p in parts)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


def _memory_saver() -> Any:
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except ImportError:  # pragma: no cover - older langgraph
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


def is_postgres(url: str | None) -> bool:
    if not url or "://" not in url:
        return False
    return url.split("://", 1)[0].lower() in _POSTGRES_SCHEMES


@asynccontextmanager
async def open_checkpointer(url: str | None = None, setup: bool = True) -> AsyncIterator[Any]:
    """Yield a checkpointer for ``url``; an in-memory one when there is none.

    Asynchronous because the Postgres saver owns a connection pool with an
    async lifecycle. The in-memory saver ignores all of that, so both are used
    identically at the call site.

    A non-Postgres URL (a Mongo memory bank, say) falls back to the in-memory
    saver rather than failing: LangGraph has other savers, but they are
    separate packages this project does not depend on, and losing durable
    checkpointing is not a reason to refuse to run.
    """
    if not is_postgres(url):
        yield _memory_saver()
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:  # pragma: no cover - depends on install
        raise StoreError(
            "Postgres checkpointing needs the extra: pip install 'meta-evolver[postgres]'"
        ) from exc

    try:
        async with AsyncPostgresSaver.from_conn_string(url) as saver:
            if setup:
                # Idempotent; creates the checkpoint tables on first use.
                await saver.setup()
            yield saver
    except StoreError:
        raise
    except Exception as exc:
        raise StoreError(f"cannot checkpoint to {redact(url)}: {exc}") from exc


def describe_checkpointer(checkpointer: Any) -> str:
    """Human-readable name, for logs and run summaries."""
    return type(checkpointer).__name__ if checkpointer is not None else "none"
