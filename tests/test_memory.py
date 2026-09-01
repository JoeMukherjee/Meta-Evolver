"""ReasoningMemoryBank: retrieval, dedup, credit assignment, pruning."""
from __future__ import annotations

from meta_evolver.core.types import MemoryItem
from meta_evolver.llm.embeddings import Embedder, cosine, hashed_embedding
from meta_evolver.memory.bank import ReasoningMemoryBank


def item(**kwargs) -> MemoryItem:
    base = {"scenario": "general", "lesson": "a lesson", "title": "t"}
    return MemoryItem(**{**base, **kwargs})


def test_fallback_embedding_is_process_stable():
    """The whole bank depends on this.

    Python randomizes ``hash()`` per process, so a bank persisted by one
    process and queried by the next would compare vectors from two different
    projections and retrieve near-noise -- while looking like it worked.
    """
    a = hashed_embedding("connection pool timeout on the payment service")
    b = hashed_embedding("connection pool timeout on the payment service")
    assert a == b
    assert abs(sum(v * v for v in a) - 1.0) < 1e-6  # L2-normalized

    unrelated = hashed_embedding("the knife is inside the garbage can")
    assert cosine(a, unrelated) < cosine(a, b)


def test_cosine_tolerates_dimension_mismatch():
    # Items embedded remotely before an outage sit next to locally-embedded
    # ones after it; retrieval must not raise.
    assert cosine([1.0, 0.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([], [1.0]) == 0.0


def test_retrieval_prefers_semantically_close_items():
    bank = ReasoningMemoryBank(
        [
            item(id="pool", scenario="resource_limit", lesson="connection pool exhausted, raise the limit"),
            item(id="search", scenario="search_order", lesson="check drawers before countertops"),
        ]
    )
    hits = bank.retrieve("the connection pool is exhausted", k=1, mode="cosine")
    assert hits[0].id == "pool"


def test_mmr_spends_slots_on_diversity_not_paraphrase():
    bank = ReasoningMemoryBank(
        [
            item(id="a", embedding=[1.0, 0.0, 0.0]),
            item(id="b", embedding=[0.97, 0.05, 0.0]),
            item(id="c", embedding=[0.5, 0.86, 0.0]),
        ]
    )
    bank.embedder = _FixedEmbedder([1.0, 0.0, 0.0])

    cosine_hits = [m.id for m in bank.retrieve("q", k=2, mode="cosine", utility_weight=0.0)]
    assert cosine_hits == ["a", "b"]

    mmr_hits = [
        m.id
        for m in bank.retrieve("q", k=2, mode="mmr", mmr_lambda=0.4, utility_weight=0.0)
    ]
    assert mmr_hits == ["a", "c"]


def test_add_merges_near_duplicates_and_keeps_the_track_record():
    bank = ReasoningMemoryBank()
    first, is_new = bank.add(item(lesson="Always restart after patching config."))
    assert is_new
    first.uses, first.wins = 6, 5

    _, is_new = bank.add(
        item(lesson="always restart after patching  config", procedure="1. patch 2. restart")
    )
    assert not is_new
    assert len(bank) == 1
    # The merge inherits the incumbent's record rather than resetting it.
    assert bank.items[0].uses == 6
    assert bank.items[0].procedure == "1. patch 2. restart"


def test_credit_assignment_and_pruning():
    bank = ReasoningMemoryBank(
        [item(id="good", lesson="works"), item(id="bad", lesson="misleads")]
    )
    for _ in range(5):
        bank.credit_assign([(["good"], True), (["bad"], False)])

    assert bank.items[0].utility > 0.8
    assert bank.items[1].utility < 0.2

    dropped = bank.prune(min_uses=4, min_utility=0.34)
    assert [d.id for d in dropped] == ["bad"]
    assert [m.id for m in bank.items] == ["good"]


def test_pruning_gives_new_memories_a_fair_trial():
    """An item retrieved twice into two hard tasks has not been disproven."""
    bank = ReasoningMemoryBank([item(id="new", lesson="unproven")])
    bank.credit_assign([(["new"], False), (["new"], False)])
    assert bank.prune(min_uses=4, min_utility=0.34) == []
    assert len(bank) == 1


def test_utility_weighting_demotes_a_losing_memory():
    bank = ReasoningMemoryBank(
        [
            item(id="relevant_but_bad", lesson="connection pool exhausted, raise the limit"),
            item(id="less_relevant_good", lesson="connection pool queue depth is the signal"),
        ]
    )
    query = "connection pool exhausted, raise the limit"
    assert bank.retrieve(query, k=1, mode="cosine")[0].id == "relevant_but_bad"

    for _ in range(8):
        bank.credit_assign([(["relevant_but_bad"], False), (["less_relevant_good"], True)])
    assert bank.retrieve(query, k=1, mode="cosine")[0].id == "less_relevant_good"


def test_round_trip_persistence(tmp_path):
    path = tmp_path / "memories.jsonl"
    bank = ReasoningMemoryBank([item(id="x", lesson="persist me")], path=path)
    bank.save()

    reloaded = ReasoningMemoryBank.load(path)
    assert len(reloaded) == 1
    assert reloaded.items[0].lesson == "persist me"
    # Embeddings survive, so retrieval quality is identical across processes.
    assert reloaded.items[0].embedding == bank.items[0].embedding


def test_corrupt_line_costs_one_memory_not_the_run(tmp_path):
    path = tmp_path / "memories.jsonl"
    path.write_text(
        '{"id": "ok", "lesson": "fine"}\nnot json at all\n', encoding="utf-8"
    )
    assert len(ReasoningMemoryBank.load(path)) == 1


def test_capacity_evicts_the_least_useful():
    lessons = ["retry timed out reads", "verify before submitting", "prefer breadth first search"]
    bank = ReasoningMemoryBank(max_items=2)
    for i, lesson in enumerate(lessons):
        stored, was_new = bank.add(item(scenario=f"topic_{i}", lesson=lesson))
        assert was_new, "distinct lessons must not be merged as duplicates"
        stored.uses, stored.wins = 10, 10 - i * 5
    assert len(bank) == 2
    # The worst performer (wins=0 of 10) is the one dropped.
    assert lessons[2] not in {m.lesson for m in bank.items}


def test_prompt_block_frames_memories_as_fallible():
    bank = ReasoningMemoryBank()
    block = bank.render_prompt_block([item(lesson="l", title="T")], soft_prior=True)
    assert "priors, not instructions" in block
    assert "abandon it" in block


def test_failure_memories_render_as_anti_patterns():
    bank = ReasoningMemoryBank()
    block = bank.render_prompt_block([item(polarity="failure", lesson="do not do this")])
    assert "ANTI-PATTERN" in block


class _FixedEmbedder(Embedder):
    """Embedder returning one constant query vector."""

    def __init__(self, vector):
        super().__init__(embeddings=None)
        self.vector = vector

    def embed(self, texts):
        return [list(self.vector) for _ in texts]
