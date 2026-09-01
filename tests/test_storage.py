"""Storage backends: the contract, and the two that need a live server.

The contract tests run against every available backend through the same
parametrized suite, which is the point -- a backend that passes them is
substitutable, and the bank does not care which one it got.

Postgres and MongoDB tests skip when nothing is listening rather than failing,
so `pytest` stays green on a laptop with no Docker. To run them:

    docker compose -f docker/docker-compose.yml up -d postgres
    docker compose -f docker/docker-compose.yml --profile mongo up -d mongo
"""
from __future__ import annotations

import os

import pytest

from meta_evolver.core.types import MemoryItem
from meta_evolver.llm.embeddings import Embedder, hashed_embedding
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.storage import JsonlMemoryStore, StoreError, open_store, redact
from meta_evolver.storage.base import DB_URL_ENV

POSTGRES_URL = os.environ.get(
    "META_EVOLVER_TEST_POSTGRES", "postgresql://meta:meta@localhost:5433/meta_evolver"
)
MONGO_URL = os.environ.get(
    "META_EVOLVER_TEST_MONGO",
    "mongodb://meta:meta@localhost:27018/meta_evolver?authSource=admin",
)


#: Reachability is probed once per session, not per test. Each failed Mongo
#: connect costs its full server-selection timeout, and paying that seven
#: times turns "no database installed" into a 90-second test run.
_REACHABLE: dict[str, str | None] = {}


def _unreachable(label: str, url: str) -> str | None:
    """Reason this backend cannot be used, or ``None`` if it can."""
    if label not in _REACHABLE:
        try:
            open_store(url, namespace="pytest-probe", dim=768).close()
            _REACHABLE[label] = None
        except Exception as exc:
            _REACHABLE[label] = f"no {label} at {redact(url)}: {type(exc).__name__}"
    return _REACHABLE[label]


def item(**kwargs) -> MemoryItem:
    base = {"scenario": "general", "lesson": "a lesson", "title": "t"}
    merged = {**base, **kwargs}
    merged.setdefault("embedding", hashed_embedding(merged["lesson"], 768))
    memory = MemoryItem(**merged)
    memory.id = memory.id or f"mem-{memory.key()}"
    return memory


# --- URL routing (no server needed) -----------------------------------------


def test_url_picks_the_backend(tmp_path):
    assert isinstance(open_store(tmp_path / "m.jsonl"), JsonlMemoryStore)
    assert isinstance(open_store(f"jsonl://{tmp_path / 'm.jsonl'}"), JsonlMemoryStore)


def test_no_url_falls_back_to_a_file(monkeypatch, tmp_path):
    monkeypatch.delenv(DB_URL_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert isinstance(open_store(None), JsonlMemoryStore)


def test_env_var_is_honoured_when_no_url_is_passed(monkeypatch, tmp_path):
    monkeypatch.setenv(DB_URL_ENV, str(tmp_path / "from-env.jsonl"))
    store = open_store(None)
    assert isinstance(store, JsonlMemoryStore)
    assert store.path.name == "from-env.jsonl"


def test_credentials_are_redacted():
    """Connection strings reach logs, summaries and error messages."""
    assert redact("postgresql://user:hunter2@db.internal:5432/x") == (
        "postgresql://user:***@db.internal:5432/x"
    )
    assert "hunter2" not in redact("mongodb://user:hunter2@host/db")
    assert redact("memories.jsonl") == "memories.jsonl"


def test_unreachable_server_raises_store_error():
    with pytest.raises(StoreError):
        open_store("postgresql://nobody:nobody@127.0.0.1:1/none")


# --- the contract, across every available backend ---------------------------


@pytest.fixture(params=["jsonl", "postgres", "mongo"])
def store(request, tmp_path):
    """One fresh store per test, for every backend that is reachable.

    Built per test rather than once at collection: a shared connection pool
    closed by the first test's teardown would break every later one, and the
    failure would look like a backend bug rather than a fixture bug.
    """
    label = request.param
    if label == "jsonl":
        backend = JsonlMemoryStore(tmp_path / "contract.jsonl")
    else:
        url = POSTGRES_URL if label == "postgres" else MONGO_URL
        reason = _unreachable(label, url)
        if reason:
            pytest.skip(reason)
        backend = open_store(url, namespace="pytest-contract", dim=768)
        backend.delete([i.id for i in backend.load()])

    yield backend

    try:
        backend.delete([i.id for i in backend.load()])
        backend.close()
    except Exception:
        pass


def test_roundtrip(store):
    store.upsert([item(id="a", lesson="first lesson"), item(id="b", lesson="second lesson")])
    loaded = {i.id: i for i in store.load()}
    assert set(loaded) == {"a", "b"}
    assert loaded["a"].lesson == "first lesson"
    assert loaded["a"].embedding, "embeddings must survive the round trip"


def test_upsert_replaces_by_id(store):
    store.upsert([item(id="a", lesson="before")])
    store.upsert([item(id="a", lesson="after")])
    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].lesson == "after"


def test_upsert_does_not_clobber_earned_counters(store):
    """Editing a memory's text must not erase its track record.

    The pruner reads utility. If an upsert of revised wording reset uses and
    wins, a well-performing memory would silently return to unproven and could
    be pruned on its next bad episode.
    """
    store.upsert([item(id="a", lesson="original")])
    store.record_usage([("a", True)] * 5)

    store.upsert([item(id="a", lesson="revised wording", uses=0, wins=0)])
    reloaded = store.load()[0]
    assert reloaded.lesson == "revised wording"
    assert (reloaded.uses, reloaded.wins) == (5, 5)


def test_record_usage_increments(store):
    store.upsert([item(id="a"), item(id="b")])
    store.record_usage([("a", True), ("a", False), ("b", True)])

    loaded = {i.id: i for i in store.load()}
    assert (loaded["a"].uses, loaded["a"].wins) == (2, 1)
    assert (loaded["b"].uses, loaded["b"].wins) == (1, 1)


def test_record_usage_ignores_unknown_ids(store):
    store.upsert([item(id="a")])
    store.record_usage([("a", True), ("does-not-exist", True)])
    assert store.load()[0].uses == 1


def test_delete(store):
    store.upsert([item(id="a"), item(id="b")])
    store.delete(["a", "missing"])  # a missing id is not an error
    assert [i.id for i in store.load()] == ["b"]


def test_empty_writes_are_no_ops(store):
    store.upsert([])
    store.delete([])
    store.record_usage([])
    assert store.count() == 0


# --- file-store specifics ---------------------------------------------------


def test_jsonl_write_is_atomic(tmp_path):
    """An interrupted save must not destroy the bank it was replacing."""
    path = tmp_path / "bank.jsonl"
    store = JsonlMemoryStore(path)
    store.upsert([item(id="a", lesson="survives")])

    original = path.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        import json as _json
        from unittest.mock import patch

        with patch.object(_json, "dumps", side_effect=RuntimeError("disk full")):
            store.upsert([item(id="b", lesson="never lands")])

    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".bank.jsonl.*")), "temp file must be cleaned up"


def test_jsonl_survives_a_corrupt_line(tmp_path):
    path = tmp_path / "bank.jsonl"
    path.write_text('{"id": "ok", "lesson": "fine"}\nnot json\n', encoding="utf-8")
    assert len(JsonlMemoryStore(path).load()) == 1


# --- bank integration -------------------------------------------------------


def test_bank_writes_through_to_its_store(tmp_path):
    store = JsonlMemoryStore(tmp_path / "bank.jsonl")
    bank = ReasoningMemoryBank(store=store, embedder=Embedder())

    bank.add(item(lesson="restart the owner after patching its config"))
    assert store.count() == 1

    bank.credit_assign([([bank.items[0].id], False)] * 5)
    assert store.load()[0].uses == 5

    bank.prune(min_uses=4, min_utility=0.34)
    assert store.count() == 0, "a pruned memory must leave the store too"


def test_bank_reload_sees_another_writer(tmp_path):
    path = tmp_path / "shared.jsonl"
    writer = ReasoningMemoryBank(store=JsonlMemoryStore(path), embedder=Embedder())
    reader = ReasoningMemoryBank(store=JsonlMemoryStore(path), embedder=Embedder())

    writer.add(item(lesson="written by the other process"))
    assert len(reader) == 0

    reader.reload()
    assert len(reader) == 1


def test_postgres_ranks_server_side():
    """pgvector does the nearest-neighbour search, not Python."""
    reason = _unreachable("postgres", POSTGRES_URL)
    if reason:
        pytest.skip(reason)
    store = open_store(POSTGRES_URL, namespace="pytest-ann", dim=768)

    store.delete([i.id for i in store.load()])
    try:
        assert store.supports_vector_search
        # dim=768 so the local encoder's vectors fit the index too; this test
        # must not need an embedding API to exercise server-side ranking.
        bank = ReasoningMemoryBank(store=store, embedder=Embedder(dim=768))

        for scenario, lesson in [
            ("resource_limit", "connection pool exhausted; raise the limit and restart"),
            ("search_order", "check drawers and dishwashers before countertops"),
            ("retention", "a full disk is fixed at the writer that owns the mount"),
        ]:
            bank.add(item(scenario=scenario, title=scenario, lesson=lesson))

        hits = bank.retrieve("the connection pool keeps timing out", k=1, mode="cosine")
        assert hits and hits[0].scenario == "resource_limit"
        assert store.last_indexed == 3, "every row should be in the vector index"
    finally:
        store.delete([i.id for i in store.load()])
        store.close()


def test_postgres_column_width_wins_over_the_argument():
    """The live column is authoritative; the constructor argument is a wish.

    `CREATE TABLE IF NOT EXISTS` is a no-op against an existing table, so a
    store told 256 against a 768 column must degrade to "stored but not
    indexed" rather than have every insert rejected by the server.
    """
    reason = _unreachable("postgres", POSTGRES_URL)
    if reason:
        pytest.skip(reason)
    store = open_store(POSTGRES_URL, namespace="pytest-dim", dim=256)

    try:
        assert store.dim == 768
        assert store.requested_dim == 256
        assert "not indexed" in store.describe

        narrow = item(id="narrow", embedding=hashed_embedding("narrow", 256))
        store.upsert([narrow])  # must not raise
        assert store.load()[0].embedding is not None
        assert len(store.load()[0].embedding) == 256
    finally:
        store.delete(["narrow"])
        store.close()


def test_mongo_reports_whether_it_can_rank_vectors():
    """Vector search is an Atlas feature, not a MongoDB one.

    A local Mongo has no `$vectorSearch`, so the store must report that
    honestly and let the bank score in Python -- rather than returning an
    empty result set, which would look like "no memories matched".
    """
    reason = _unreachable("mongo", MONGO_URL)
    if reason:
        pytest.skip(reason)

    store = open_store(MONGO_URL, namespace="pytest-mongo-probe", dim=768)
    try:
        assert store.search([0.0] * 768, k=3) is None
        assert "vector_search=" in store.describe

        bank = ReasoningMemoryBank(store=store, embedder=Embedder(dim=768))
        bank.add(item(scenario="resource_limit", lesson="the pool limit is the constraint"))
        bank.add(item(scenario="search_order", lesson="check drawers before countertops"))

        hits = bank.retrieve("the connection pool is the constraint", k=1, mode="cosine")
        assert hits and hits[0].scenario == "resource_limit"
    finally:
        store.delete([i.id for i in store.load()])
        store.close()


def test_namespaces_do_not_collide_on_a_shared_id(store):
    """Two benchmarks that learn the same lesson must not overwrite each other.

    A memory's id is derived from its scenario and lesson text, so this is not
    a contrived case -- it is what happens the first time two benchmarks reach
    the same conclusion. Keyed on the bare id, the second write updates the
    first namespace's row: it succeeds, lands somewhere its author cannot
    read, and the memory silently vanishes from the bank that created it.
    """
    if isinstance(store, JsonlMemoryStore):
        pytest.skip("a file store holds one namespace by construction")

    shared = item(id="mem-shared", lesson="a lesson two benchmarks both learn")
    store.upsert([shared])
    store.record_usage([("mem-shared", True)] * 3)

    other = open_store(
        POSTGRES_URL if "postgres" in store.describe else MONGO_URL,
        namespace="pytest-other",
        dim=768,
    )
    try:
        other.delete(["mem-shared"])
        other.upsert([item(id="mem-shared", lesson="a lesson two benchmarks both learn")])

        # Each namespace sees exactly its own copy...
        assert [i.id for i in other.load()] == ["mem-shared"]
        assert [i.id for i in store.load()] == ["mem-shared"]
        # ...and the second write did not touch the first's track record.
        assert store.load()[0].uses == 3
        assert other.load()[0].uses == 0
    finally:
        other.delete(["mem-shared"])
        other.close()
