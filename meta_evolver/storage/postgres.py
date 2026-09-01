"""Postgres + pgvector: the backend that makes the bank a shared, live thing.

Three properties the file store cannot give, each of which the evolution loop
actually depends on:

**Atomic credit assignment.** ``record_usage`` is a single ``UPDATE ... SET
uses = uses + 1`` per event. Rollouts run concurrently and all credit the same
memories; under a read-modify-write, increments are lost, utilities drift low,
and the pruner deletes memories that were doing fine. In SQL that race does not
exist.

**Server-side nearest neighbours.** Embeddings live in a ``vector`` column with
an HNSW cosine index, so retrieval is an indexed lookup rather than a Python
loop over the whole bank. The query fetches a *candidate pool* rather than
exactly ``k``, because the bank re-ranks with MMR and MMR needs something to
choose diversity from.

**A bank several runs can share.** ``namespace`` scopes each benchmark, so one
database serves all of them without their lessons bleeding across.

That scoping is in the *primary key*, ``(namespace, id)``, not only in the
``WHERE`` clauses -- and the distinction is not cosmetic. A memory's id is
derived from its scenario and lesson text, so two benchmarks that
independently learn the same thing derive the same id. Keyed on ``id`` alone,
the second one's ``ON CONFLICT`` updates the first one's row: the write
succeeds, lands in a namespace its author cannot read, and the memory simply
vanishes from the bank that created it.

Two details worth stating because they are easy to get wrong:

*Dimension.* A ``vector`` column's width is fixed at DDL time, but a bank can
legitimately hold mixed widths -- items embedded remotely sit beside items the
local fallback encoded during an outage. So the JSON copy is the source of
truth and the typed column is the *index*: rows whose width matches get one,
rows that do not are still stored and still returned by ``load``. They are
simply invisible to ANN, and :meth:`search` says how many that was.

The width used for that test is read back **from the live column**, not taken
from the constructor argument. ``CREATE TABLE IF NOT EXISTS`` is a no-op
against an existing table, so a store told 768 can easily be talking to a
column built at 1536 by an earlier run or by ``init.sql``. Trusting the
argument means every insert is rejected by the server with a dimension error;
trusting the column means the mismatch degrades to "not indexed" instead.

*Schema creation.* ``CREATE TABLE IF NOT EXISTS`` runs on connect. That is
convenient for a local Docker instance and wrong for a managed database where
the application role should not hold DDL rights -- pass ``create_schema=False``
and apply ``docker/init.sql`` yourself.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from meta_evolver.core.types import MemoryItem
from meta_evolver.storage.base import MemoryStore, StoreError, redact

#: Fetched per query before MMR re-ranking. Four times the requested k gives
#: the diversity pass room to work without pulling the whole table back.
CANDIDATE_MULTIPLIER = 4

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS meta_evolver_memories (
    id                  text NOT NULL,
    namespace           text NOT NULL DEFAULT 'default',
    title               text NOT NULL DEFAULT '',
    scenario            text NOT NULL DEFAULT 'general',
    lesson              text NOT NULL DEFAULT '',
    procedure           text NOT NULL DEFAULT '',
    triggers            jsonb NOT NULL DEFAULT '[]'::jsonb,
    polarity            text NOT NULL DEFAULT 'success',
    source_task_ids     jsonb NOT NULL DEFAULT '[]'::jsonb,
    benchmark           text NOT NULL DEFAULT '',
    embedding_json      jsonb,
    embedding           vector({dim}),
    uses                integer NOT NULL DEFAULT 0,
    wins                integer NOT NULL DEFAULT 0,
    created_generation  integer NOT NULL DEFAULT 0,
    last_used_generation integer NOT NULL DEFAULT 0,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),
    -- Composite, so the same derived id can exist independently in two
    -- namespaces. See the module note.
    PRIMARY KEY (namespace, id)
);

CREATE INDEX IF NOT EXISTS meta_evolver_memories_namespace_idx
    ON meta_evolver_memories (namespace);

CREATE INDEX IF NOT EXISTS meta_evolver_memories_embedding_idx
    ON meta_evolver_memories USING hnsw (embedding vector_cosine_ops);
"""

_COLUMNS = (
    "id, title, scenario, lesson, procedure, triggers, polarity, "
    "source_task_ids, benchmark, embedding_json, uses, wins, "
    "created_generation, last_used_generation"
)


class PostgresMemoryStore(MemoryStore):
    """pgvector-backed memory store."""

    supports_vector_search = True

    def __init__(
        self,
        url: str,
        namespace: str = "default",
        dim: int = 768,
        create_schema: bool = True,
        min_size: int = 1,
        max_size: int = 8,
        **_: Any,
    ) -> None:
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - depends on install
            raise StoreError(
                "Postgres support needs the extra: pip install 'meta-evolver[postgres]'"
            ) from exc

        self.url = url
        self.namespace = namespace
        self.requested_dim = int(dim)
        self.dim = int(dim)
        self.last_indexed = 0
        self.last_unindexed = 0

        try:
            self.pool = ConnectionPool(
                url, min_size=min_size, max_size=max_size, open=True, timeout=15
            )
            self.pool.wait(timeout=15)
        except Exception as exc:
            raise StoreError(f"cannot connect to {redact(url)}: {exc}") from exc

        if create_schema:
            self._create_schema()

        self._check_primary_key()

        # The column is authoritative. See the module note on dimension.
        actual = self._column_dim()
        if actual is not None:
            self.dim = actual

        self.describe = f"postgres:{redact(url)} ns={namespace} dim={self.dim}"
        if actual is not None and actual != self.requested_dim:
            self.describe += (
                f" (column is {actual}, embeddings are {self.requested_dim}: "
                "those rows are stored but not indexed)"
            )

    def _check_primary_key(self) -> None:
        """Refuse to run against a table keyed on ``id`` alone.

        Such a table predates the composite key and silently misfiles writes
        across namespaces. Failing loudly at connect is far kinder than the
        alternative, which is memories quietly disappearing at runtime.
        """
        try:
            with self.pool.connection() as conn:
                row = conn.execute(
                    "SELECT array_agg(a.attname ORDER BY a.attname) "
                    "FROM pg_index i "
                    "JOIN pg_attribute a ON a.attrelid = i.indrelid "
                    "                   AND a.attnum = ANY(i.indkey) "
                    "WHERE i.indrelid = to_regclass('meta_evolver_memories') "
                    "  AND i.indisprimary"
                ).fetchone()
        except Exception:
            return
        if not row or not row[0]:
            return
        columns = set(row[0])
        if columns == {"id"}:
            raise StoreError(
                "meta_evolver_memories is keyed on (id) alone, which misfiles "
                "writes between namespaces. Migrate with:\n"
                "  ALTER TABLE meta_evolver_memories "
                "DROP CONSTRAINT meta_evolver_memories_pkey, "
                "ADD PRIMARY KEY (namespace, id);"
            )

    def _column_dim(self) -> int | None:
        """Width of the live ``embedding`` column, or ``None`` if absent."""
        try:
            with self.pool.connection() as conn:
                row = conn.execute(
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = to_regclass('meta_evolver_memories') "
                    "  AND attname = 'embedding' AND NOT attisdropped"
                ).fetchone()
        except Exception:
            return None
        if not row or not row[0]:
            return None
        match = re.search(r"\((\d+)\)", str(row[0]))
        return int(match.group(1)) if match else None

    def _create_schema(self) -> None:
        try:
            with self.pool.connection() as conn:
                conn.execute(_SCHEMA.format(dim=self.dim))
                conn.commit()
        except Exception as exc:
            raise StoreError(
                f"could not create schema on {redact(self.url)}: {exc}. "
                "If the role lacks DDL rights, apply docker/init.sql and pass "
                "create_schema=False."
            ) from exc

    # -- reads -------------------------------------------------------------

    def load(self) -> list[MemoryItem]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM meta_evolver_memories "
                "WHERE namespace = %s ORDER BY created_generation, id",
                (self.namespace,),
            ).fetchall()
        return [_row_to_item(row) for row in rows]

    def count(self) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT count(*) FROM meta_evolver_memories WHERE namespace = %s",
                (self.namespace,),
            ).fetchone()
        return int(row[0]) if row else 0

    def search(
        self, vector: Sequence[float], k: int, min_similarity: float = 0.0
    ) -> list[tuple[MemoryItem, float]] | None:
        """Indexed cosine nearest neighbours, or ``None`` if none are indexed.

        Returns a candidate pool rather than exactly ``k``: MMR re-ranking
        upstream needs alternatives to trade relevance against diversity.
        """
        if len(vector) != self.dim:
            # A query embedded at a different width than the index cannot use
            # it. Falling back to Python scoring is correct and quiet; raising
            # would take down a run over a recoverable mismatch.
            return None

        limit = max(k * CANDIDATE_MULTIPLIER, k)
        # `<=>` is cosine distance in pgvector, so similarity is 1 - distance.
        sql = (
            f"SELECT {_COLUMNS}, 1 - (embedding <=> %s::vector) AS similarity "
            "FROM meta_evolver_memories "
            "WHERE namespace = %s AND embedding IS NOT NULL "
            "  AND 1 - (embedding <=> %s::vector) >= %s "
            "ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        literal = _vector_literal(vector)
        with self.pool.connection() as conn:
            rows = conn.execute(
                sql,
                (literal, self.namespace, literal, float(min_similarity), literal, limit),
            ).fetchall()
            counts = conn.execute(
                "SELECT count(*) FILTER (WHERE embedding IS NOT NULL), count(*) "
                "FROM meta_evolver_memories WHERE namespace = %s",
                (self.namespace,),
            ).fetchone()

        self.last_indexed = int(counts[0]) if counts else 0
        self.last_unindexed = (int(counts[1]) - self.last_indexed) if counts else 0
        if self.last_indexed == 0:
            # Nothing is indexed (an all-fallback bank, say). Let the caller
            # score in Python rather than silently returning an empty result,
            # which would look like "no memories matched".
            return None
        return [(_row_to_item(row[:-1]), float(row[-1])) for row in rows]

    # -- writes ------------------------------------------------------------

    def upsert(self, items: Sequence[MemoryItem]) -> None:
        if not items:
            return
        sql = """
            INSERT INTO meta_evolver_memories (
                id, namespace, title, scenario, lesson, procedure, triggers,
                polarity, source_task_ids, benchmark, embedding_json, embedding,
                uses, wins, created_generation, last_used_generation, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s::jsonb,
                %s::vector, %s, %s, %s, %s, now()
            )
            ON CONFLICT (namespace, id) DO UPDATE SET
                title = EXCLUDED.title,
                scenario = EXCLUDED.scenario,
                lesson = EXCLUDED.lesson,
                procedure = EXCLUDED.procedure,
                triggers = EXCLUDED.triggers,
                polarity = EXCLUDED.polarity,
                source_task_ids = EXCLUDED.source_task_ids,
                benchmark = EXCLUDED.benchmark,
                embedding_json = EXCLUDED.embedding_json,
                embedding = EXCLUDED.embedding,
                -- Counters are NOT overwritten from the incoming row: an
                -- upsert of edited text must not erase a memory's earned
                -- track record. record_usage owns uses/wins.
                created_generation = meta_evolver_memories.created_generation,
                last_used_generation = GREATEST(
                    meta_evolver_memories.last_used_generation,
                    EXCLUDED.last_used_generation
                ),
                updated_at = now()
        """
        params = [
            (
                item.id,
                self.namespace,
                item.title,
                item.scenario,
                item.lesson,
                item.procedure,
                json.dumps(item.triggers),
                item.polarity,
                json.dumps(item.source_task_ids),
                item.benchmark,
                json.dumps(item.embedding) if item.embedding else None,
                _vector_literal(item.embedding) if _indexable(item.embedding, self.dim) else None,
                item.uses,
                item.wins,
                item.created_generation,
                item.last_used_generation,
            )
            for item in items
        ]
        with self.pool.connection() as conn:
            conn.cursor().executemany(sql, params)
            conn.commit()

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        with self.pool.connection() as conn:
            conn.execute(
                "DELETE FROM meta_evolver_memories WHERE namespace = %s AND id = ANY(%s)",
                (self.namespace, list(ids)),
            )
            conn.commit()

    def record_usage(self, events: Sequence[tuple[str, bool]]) -> None:
        """Atomic per-item increments -- the whole point of this backend."""
        if not events:
            return
        sql = (
            "UPDATE meta_evolver_memories "
            "SET uses = uses + 1, wins = wins + %s, updated_at = now() "
            "WHERE namespace = %s AND id = %s"
        )
        params = [(1 if won else 0, self.namespace, mid) for mid, won in events]
        with self.pool.connection() as conn:
            conn.cursor().executemany(sql, params)
            conn.commit()

    def close(self) -> None:
        try:
            self.pool.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _indexable(vector: Sequence[float] | None, dim: int) -> bool:
    return bool(vector) and len(vector) == dim


def _vector_literal(vector: Sequence[float] | None) -> str | None:
    """pgvector's text input form: ``[1,2,3]``."""
    if not vector:
        return None
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def _row_to_item(row: Sequence[Any]) -> MemoryItem:
    (
        item_id,
        title,
        scenario,
        lesson,
        procedure,
        triggers,
        polarity,
        source_task_ids,
        benchmark,
        embedding_json,
        uses,
        wins,
        created_generation,
        last_used_generation,
    ) = row
    return MemoryItem(
        id=item_id,
        title=title or "",
        scenario=scenario or "general",
        lesson=lesson or "",
        procedure=procedure or "",
        triggers=_as_list(triggers),
        polarity="failure" if polarity == "failure" else "success",
        source_task_ids=_as_list(source_task_ids),
        benchmark=benchmark or "",
        embedding=_as_list(embedding_json) or None,
        uses=int(uses or 0),
        wins=int(wins or 0),
        created_generation=int(created_generation or 0),
        last_used_generation=int(last_used_generation or 0),
    )


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []
