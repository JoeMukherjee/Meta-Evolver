"""memory.consolidate -- the offline "sleep cycle" merge pass."""
from __future__ import annotations

from meta_evolver.core.types import MemoryItem
from meta_evolver.llm.embeddings import Embedder
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.memory.consolidate import consolidate


def item(**kwargs) -> MemoryItem:
    base = {"scenario": "general", "lesson": "a lesson", "title": "t"}
    return MemoryItem(**{**base, **kwargs})


class _FixedEmbedder(Embedder):
    """Assigns embeddings by a name->vector lookup table, so clustering is
    fully deterministic regardless of the fallback hashed encoder's real
    lexical-overlap behaviour."""

    def __init__(self, table: dict[str, list[float]]):
        super().__init__(embeddings=None)
        self.table = table

    def embed(self, texts):
        return [self.table.get(t, [0.0, 0.0, 1.0]) for t in texts]


def _bank_with(items: list[MemoryItem], table: dict[str, list[float]]) -> ReasoningMemoryBank:
    bank = ReasoningMemoryBank()
    bank.embedder = _FixedEmbedder(table)
    for it in items:
        it.embedding = table.get(it.embed_text(), [0.0, 0.0, 1.0])
    bank.items = items
    return bank


# -- clustering / merging behaviour -----------------------------------------


def test_two_close_items_merge_into_one():
    # utility = (wins+1)/(uses+2): a = 4/6 = 0.667, b = 3/4 = 0.75 -- b wins.
    a = item(id="a", lesson="restart the pool", uses=4, wins=3)
    b = item(id="b", lesson="restart pool now", uses=2, wins=2)
    table = {a.embed_text(): [1.0, 0.0, 0.0], b.embed_text(): [0.99, 0.01, 0.0]}
    bank = _bank_with([a, b], table)

    removed = consolidate(bank, similarity_threshold=0.9)

    assert len(bank.items) == 1
    assert [it.id for it in removed] == ["a"]
    survivor = bank.items[0]
    assert survivor.id == "b"  # higher utility is the anchor


def test_merged_item_pools_uses_and_wins():
    a = item(id="a", lesson="restart the pool", uses=10, wins=9)  # utility ~0.83
    b = item(id="b", lesson="restart pool now", uses=2, wins=2)  # utility ~0.75
    table = {a.embed_text(): [1.0, 0.0, 0.0], b.embed_text(): [0.99, 0.01, 0.0]}
    bank = _bank_with([a, b], table)

    consolidate(bank, similarity_threshold=0.9)

    survivor = bank.items[0]
    assert survivor.id == "a"  # a has both higher utility and more uses
    assert survivor.uses == 12
    assert survivor.wins == 11


def test_anchor_keeps_the_text_of_the_strongest_member_not_a_rewrite():
    """No model call: the merged item's lesson text is exactly the anchor's
    own, never a new synthesis of the two."""
    a = item(id="a", lesson="restart the pool", uses=10, wins=9)
    b = item(id="b", lesson="a completely different phrasing of the same idea", uses=1, wins=1)
    table = {a.embed_text(): [1.0, 0.0, 0.0], b.embed_text(): [0.99, 0.01, 0.0]}
    bank = _bank_with([a, b], table)

    consolidate(bank, similarity_threshold=0.9)

    assert bank.items[0].lesson == "restart the pool"


def test_dissimilar_items_are_left_alone():
    a = item(id="a", lesson="restart the pool")
    b = item(id="b", lesson="the weather is nice")
    table = {a.embed_text(): [1.0, 0.0, 0.0], b.embed_text(): [0.0, 1.0, 0.0]}
    bank = _bank_with([a, b], table)

    removed = consolidate(bank, similarity_threshold=0.9)

    assert removed == []
    assert len(bank.items) == 2


def test_a_single_item_is_never_merged_with_itself():
    a = item(id="a", lesson="solo item")
    bank = _bank_with([a], {a.embed_text(): [1.0, 0.0, 0.0]})
    assert consolidate(bank, similarity_threshold=0.9) == []
    assert len(bank.items) == 1


def test_success_and_failure_items_never_merge_even_if_textually_identical():
    """A success strategy and a failure anti-pattern are never the same
    lesson, however close their text -- polarity is never crossed."""
    a = item(id="a", lesson="restart the pool", polarity="success")
    b = item(id="b", lesson="restart the pool", polarity="failure")
    table = {a.embed_text(): [1.0, 0.0, 0.0], b.embed_text(): [1.0, 0.0, 0.0]}
    bank = _bank_with([a, b], table)

    removed = consolidate(bank, similarity_threshold=0.9)

    assert removed == []
    assert len(bank.items) == 2


def test_single_linkage_transitively_merges_a_chain():
    """A-B close, B-C close, A-C not close enough alone -- still one cluster,
    because B is genuinely the same lesson as each of the others."""
    a = item(id="a", lesson="a", uses=1, wins=1)
    b = item(id="b", lesson="b", uses=1, wins=1)
    c = item(id="c", lesson="c", uses=1, wins=1)
    table = {
        a.embed_text(): [1.0, 0.0, 0.0],
        b.embed_text(): [0.9, 0.44, 0.0],  # close to both a and c
        c.embed_text(): [0.6, 0.8, 0.0],
    }
    bank = _bank_with([a, b, c], table)

    removed = consolidate(bank, similarity_threshold=0.85)

    assert len(bank.items) == 1
    assert len(removed) == 2


def test_domain_stats_are_summed_on_merge():
    a = item(id="a", lesson="restart the pool")
    a.record_domain_usage("travel", True)
    b = item(id="b", lesson="restart pool now")
    b.record_domain_usage("travel", True)
    table = {a.embed_text(): [1.0, 0.0, 0.0], b.embed_text(): [0.99, 0.01, 0.0]}
    bank = _bank_with([a, b], table)

    consolidate(bank, similarity_threshold=0.9)

    assert bank.items[0].domain_stats["travel"] == [2.0, 2.0]


def test_triggers_and_source_task_ids_are_unioned_without_duplicates():
    a = item(id="a", lesson="restart the pool", triggers=["pool exhausted"], source_task_ids=["t1"])
    b = item(
        id="b",
        lesson="restart pool now",
        triggers=["pool exhausted", "connection refused"],
        source_task_ids=["t1", "t2"],
    )
    table = {a.embed_text(): [1.0, 0.0, 0.0], b.embed_text(): [0.99, 0.01, 0.0]}
    bank = _bank_with([a, b], table)

    consolidate(bank, similarity_threshold=0.9)

    survivor = bank.items[0]
    assert sorted(survivor.triggers) == ["connection refused", "pool exhausted"]
    assert sorted(survivor.source_task_ids) == ["t1", "t2"]


def test_min_cluster_size_above_cluster_size_merges_nothing():
    a = item(id="a", lesson="restart the pool")
    b = item(id="b", lesson="restart pool now")
    table = {a.embed_text(): [1.0, 0.0, 0.0], b.embed_text(): [0.99, 0.01, 0.0]}
    bank = _bank_with([a, b], table)

    removed = consolidate(bank, similarity_threshold=0.9, min_cluster_size=3)

    assert removed == []
    assert len(bank.items) == 2


def test_store_backed_bank_keeps_pooled_counters_in_sync():
    """upsert() deliberately never overwrites uses/wins (see its own
    docstring -- protects against an unrelated reword silently resetting a
    proven memory to "unproven"), so a merge's pooled counters can only
    reach the store through record_usage(). If consolidate() relied on
    upsert() alone for this, the store would silently keep the pre-merge
    counts forever -- this is the regression that guards against exactly
    that."""
    import tempfile
    from pathlib import Path

    from meta_evolver.storage import JsonlMemoryStore

    with tempfile.TemporaryDirectory() as tmp:
        store = JsonlMemoryStore(Path(tmp) / "store.jsonl")
        a = item(id="a", lesson="restart the pool", uses=4, wins=3)
        b = item(id="b", lesson="restart pool now", uses=2, wins=2)
        store.upsert([a, b])

        bank = ReasoningMemoryBank(store=store)
        table = {a.embed_text(): [1.0, 0.0, 0.0], b.embed_text(): [0.99, 0.01, 0.0]}
        bank.embedder = _FixedEmbedder(table)
        for it in (a, b):
            it.embedding = table.get(it.embed_text())
        bank.items = [a, b]

        consolidate(bank, similarity_threshold=0.9)

        survivor_id = bank.items[0].id
        reloaded = {it.id: it for it in store.load()}
        assert len(reloaded) == 1
        assert reloaded[survivor_id].uses == bank.items[0].uses == 6
        assert reloaded[survivor_id].wins == bank.items[0].wins == 5
