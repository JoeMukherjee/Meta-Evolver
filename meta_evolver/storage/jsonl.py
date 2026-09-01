"""File-backed store: the zero-dependency default.

Keeps a run working with nothing installed and nothing running, which is what
makes ``pip install`` to first evolution one step. Everything a database gives
you it does not: no concurrent writers, no server-side search, and a rewrite
of the whole file on every save.

Those limits are real but bounded. A bank of a few hundred memories scored in
Python is imperceptible next to a single model call, and a single-process run
never contends. Reach for Postgres when either assumption breaks -- concurrent
processes sharing one bank, or a bank large enough that O(n) scoring shows up.

One thing it does take seriously: a write that fails partway must not destroy
the bank it was replacing. Saves go to a temporary file and are renamed over
the target, so an interrupted write leaves the previous version intact.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from meta_evolver.core.types import MemoryItem
from meta_evolver.storage.base import MemoryStore


class JsonlMemoryStore(MemoryStore):
    """One JSON object per line, keyed by ``id``."""

    supports_vector_search = False

    def __init__(self, path: str | Path = "memories.jsonl", namespace: str = "default") -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.describe = f"jsonl:{self.path}"

    # -- reads -------------------------------------------------------------

    def load(self) -> list[MemoryItem]:
        if not self.path.exists():
            return []
        items: list[MemoryItem] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(MemoryItem(**json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                # A single malformed line costs one memory, not the run. A
                # partial final line is normal after an interrupted write.
                continue
        return items

    # -- writes ------------------------------------------------------------

    def upsert(self, items: Sequence[MemoryItem]) -> None:
        """Insert or replace by id, preserving each item's earned counters.

        ``uses`` and ``wins`` belong to :meth:`record_usage`, not to whoever
        happens to be writing revised text. Overwriting them would let an
        upsert of reworded lesson silently return a proven memory to
        "unproven", where the pruner can drop it on its next bad episode --
        and would make this backend disagree with Postgres and Mongo, which
        both protect those columns.
        """
        if not items:
            return
        existing = {item.id: item for item in self.load()}
        for item in items:
            incumbent = existing.get(item.id)
            if incumbent is not None:
                item = item.model_copy(
                    update={
                        "uses": incumbent.uses,
                        "wins": incumbent.wins,
                        "created_generation": incumbent.created_generation,
                        "last_used_generation": max(
                            incumbent.last_used_generation, item.last_used_generation
                        ),
                    }
                )
            existing[item.id] = item
        self._write(list(existing.values()))

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        drop = set(ids)
        self._write([item for item in self.load() if item.id not in drop])

    def record_usage(self, events: Sequence[tuple[str, bool]]) -> None:
        if not events:
            return
        items = {item.id: item for item in self.load()}
        for memory_id, won in events:
            item = items.get(memory_id)
            if item is None:
                continue
            item.uses += 1
            if won:
                item.wins += 1
        self._write(list(items.values()))

    def _write(self, items: Sequence[MemoryItem]) -> None:
        """Atomic replace: write a sibling temp file, then rename over.

        A crash mid-write would otherwise leave a truncated bank, and the
        symptom -- memories silently missing -- is far harder to diagnose than
        a failed write.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def count(self) -> int:
        return len(self.load())
