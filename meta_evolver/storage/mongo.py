"""MongoDB store: durable, shared, schema-free.

Worth reaching for when the surrounding system is already Mongo, or when
memories carry per-benchmark fields that do not fit a fixed schema. What it
gives you over the file store is durability and safe concurrent writes --
``record_usage`` is a ``$inc``, which is atomic per document, so parallel
rollouts crediting the same memory cannot lose an increment.

What it does not give you, on a plain deployment, is vector search. That is an
Atlas feature (``$vectorSearch`` over a configured index), not a MongoDB one.
Rather than pretend, this store probes for the index once and reports what it
found: with Atlas and an index it ranks server-side; without, :meth:`search`
returns ``None`` and the bank scores in Python exactly as it does for a file.

So the honest summary: **Mongo for durability and concurrency, Postgres for
similarity search.** If the bank is large enough that O(n) scoring matters,
Postgres is the right backend.

One scoping detail: ``_id`` is unique per *collection*, while a memory's id is
derived from its text and is therefore only unique per *namespace*. Documents
are keyed ``"<namespace>/<id>"`` so two benchmarks that independently learn the
same lesson do not overwrite each other -- which, keyed on the bare id, would
land one benchmark's memory in another's namespace where its author cannot
read it.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from meta_evolver.core.types import MemoryItem
from meta_evolver.storage.base import MemoryStore, StoreError, redact

#: Fetched per query before MMR re-ranking, when Atlas search is available.
CANDIDATE_MULTIPLIER = 4

DEFAULT_DB = "meta_evolver"
COLLECTION = "memories"

#: The Atlas Search index this store looks for. Create it with
#: `db.memories.createSearchIndex(...)` of type "vectorSearch" on `embedding`.
VECTOR_INDEX = "meta_evolver_embedding"


class MongoMemoryStore(MemoryStore):
    """Memory persistence in MongoDB."""

    def __init__(
        self,
        url: str,
        namespace: str = "default",
        dim: int = 768,
        database: str | None = None,
        vector_index: str = VECTOR_INDEX,
        **_: Any,
    ) -> None:
        try:
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover - depends on install
            raise StoreError(
                "MongoDB support needs the extra: pip install 'meta-evolver[mongo]'"
            ) from exc

        self.url = url
        self.namespace = namespace
        self.dim = int(dim)
        self.vector_index = vector_index

        parsed = urlparse(url)
        db_name = database or (parsed.path.lstrip("/") or DEFAULT_DB)
        try:
            self.client = MongoClient(url, serverSelectionTimeoutMS=8000)
            self.client.admin.command("ping")
        except Exception as exc:
            raise StoreError(
                f"cannot connect to {redact(url)}: {exc}{_auth_hint(url, parsed, exc)}"
            ) from exc

        self.collection = self.client[db_name][COLLECTION]
        self.collection.create_index("namespace")
        self.supports_vector_search = self._probe_vector_index()
        self.describe = (
            f"mongodb:{redact(url)}/{db_name} ns={namespace} "
            f"vector_search={'yes' if self.supports_vector_search else 'no (python scoring)'}"
        )

    def _probe_vector_index(self) -> bool:
        """Is an Atlas vector index configured on this collection?

        Probed once at construction rather than per query: on a plain MongoDB
        the command is unsupported and the exception is the answer, which is
        not something to pay for on every retrieval.
        """
        try:
            for index in self.collection.list_search_indexes():
                if index.get("name") == self.vector_index:
                    return True
        except Exception:
            return False
        return False

    # -- reads -------------------------------------------------------------

    def _key(self, memory_id: str) -> str:
        """Collection-unique document id for a namespace-unique memory id."""
        return f"{self.namespace}/{memory_id}"

    def load(self) -> list[MemoryItem]:
        docs = self.collection.find({"namespace": self.namespace}).sort(
            [("created_generation", 1), ("_id", 1)]
        )
        return [_doc_to_item(doc) for doc in docs]

    def count(self) -> int:
        return int(self.collection.count_documents({"namespace": self.namespace}))

    def search(
        self, vector: Sequence[float], k: int, min_similarity: float = 0.0
    ) -> list[tuple[MemoryItem, float]] | None:
        if not self.supports_vector_search or len(vector) != self.dim:
            return None
        limit = max(k * CANDIDATE_MULTIPLIER, k)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index,
                    "path": "embedding",
                    "queryVector": [float(v) for v in vector],
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": {"namespace": self.namespace},
                }
            },
            {"$addFields": {"similarity": {"$meta": "vectorSearchScore"}}},
            {"$match": {"similarity": {"$gte": float(min_similarity)}}},
        ]
        try:
            docs = list(self.collection.aggregate(pipeline))
        except Exception:
            # An index that disappeared, or a cluster that does not support
            # the stage. Degrade to Python scoring rather than failing a run.
            self.supports_vector_search = False
            return None
        return [(_doc_to_item(doc), float(doc.get("similarity", 0.0))) for doc in docs]

    # -- writes ------------------------------------------------------------

    def upsert(self, items: Sequence[MemoryItem]) -> None:
        if not items:
            return
        from pymongo import UpdateOne

        operations = []
        for item in items:
            payload = item.model_dump()
            payload.pop("id", None)
            # Counters are excluded from $set so an upsert of edited text
            # cannot erase a memory's earned track record; record_usage owns
            # them. $setOnInsert seeds them for a genuinely new document.
            uses = payload.pop("uses", 0)
            wins = payload.pop("wins", 0)
            payload["namespace"] = self.namespace
            payload["memory_id"] = item.id
            operations.append(
                UpdateOne(
                    {"_id": self._key(item.id)},
                    {"$set": payload, "$setOnInsert": {"uses": uses, "wins": wins}},
                    upsert=True,
                )
            )
        self.collection.bulk_write(operations, ordered=False)

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self.collection.delete_many(
            {"_id": {"$in": [self._key(i) for i in ids]}, "namespace": self.namespace}
        )

    def record_usage(self, events: Sequence[tuple[str, bool]]) -> None:
        """``$inc`` is atomic per document, so concurrent credit is safe."""
        if not events:
            return
        from pymongo import UpdateOne

        operations = [
            UpdateOne(
                {"_id": self._key(memory_id), "namespace": self.namespace},
                {"$inc": {"uses": 1, "wins": 1 if won else 0}},
            )
            for memory_id, won in events
        ]
        self.collection.bulk_write(operations, ordered=False)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def _auth_hint(url: str, parsed: Any, exc: Exception) -> str:
    """Name the usual cause of an authentication failure.

    A Mongo root user created by ``MONGO_INITDB_ROOT_USERNAME`` lives in the
    ``admin`` database, so a URL that names an application database without
    ``authSource=admin`` authenticates against the wrong one and fails. The
    server's message is just "Authentication failed", which sends people
    looking for a wrong password instead.
    """
    if "auth" not in str(exc).lower():
        return ""
    if not parsed.username or "authsource" in (parsed.query or "").lower():
        return ""
    separator = "&" if parsed.query else "?"
    return (
        "\n  Hint: credentials created by MONGO_INITDB_ROOT_* live in the "
        f"'admin' database. Try {redact(url)}{separator}authSource=admin"
    )


def _doc_to_item(doc: dict[str, Any]) -> MemoryItem:
    payload = {
        k: v
        for k, v in doc.items()
        if k not in {"_id", "namespace", "similarity", "memory_id"}
    }
    # The stored `memory_id` is the namespace-scoped id callers use; `_id`
    # carries the namespace prefix that keeps it unique in the collection.
    payload["id"] = str(doc.get("memory_id") or str(doc.get("_id", "")).split("/", 1)[-1])
    # Drop anything the schema does not know about rather than raising: a
    # collection shared with another tool should not break a load.
    allowed = set(MemoryItem.model_fields)
    return MemoryItem(**{k: v for k, v in payload.items() if k in allowed})
