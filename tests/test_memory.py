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


# -- Phase 1: domain conditioning, decay, and archive/revive ----------------


def test_utility_for_with_no_domain_matches_the_plain_property():
    m = item(wins=3, uses=5)
    assert m.utility_for() == m.utility
    assert m.utility_for(None) == m.utility


def test_utility_for_an_untagged_domain_falls_back_to_global():
    """An item domain conditioning has never been told anything about must
    behave exactly as if conditioning were off."""
    m = item(wins=8, uses=10)
    assert m.utility_for("some-domain-never-recorded") == m.utility


def test_record_domain_usage_is_a_second_beta_counter_by_default():
    m = item()
    for _ in range(4):
        m.record_domain_usage("travel", True)
    m.record_domain_usage("support", False)
    assert m.utility_for("travel") == (4 + 1) / (4 + 2)
    assert m.utility_for("support") == (0 + 1) / (1 + 2)
    # The plain global counters are untouched by domain-only credit.
    assert m.wins == 0 and m.uses == 0


def test_record_domain_usage_gamma_discounts_old_evidence():
    """gamma<1 means old counts shrink every time a new one is recorded --
    stale success stops propping up utility forever."""
    m = item()
    for _ in range(50):
        m.record_domain_usage("d", True, gamma=0.9)
    high = m.utility_for("d")
    for _ in range(20):
        m.record_domain_usage("d", False, gamma=0.9)
    low = m.utility_for("d")
    assert high > 0.9
    assert low < high
    # Discounted evidence recovers from a bad streak far faster than the
    # ~160 straight failures the undiscounted global counter would need.
    assert low < 0.5


def test_credit_assign_plain_pairs_are_unaffected_by_domain_machinery():
    """Every existing caller passes 2-tuples; behaviour must be bit-identical
    to before domain conditioning existed."""
    bank = ReasoningMemoryBank([item(id="good", lesson="works")])
    bank.credit_assign([(["good"], True), (["good"], True)])
    assert bank.items[0].domain_stats == {}
    assert bank.items[0].wins == 2 and bank.items[0].uses == 2


def test_credit_assign_triples_populate_domain_stats_without_a_gamma():
    bank = ReasoningMemoryBank([item(id="m", lesson="x")])
    bank.credit_assign([(["m"], True, "travel"), (["m"], False, "support")])
    assert bank.items[0].domain_stats["travel"] == [1.0, 1.0]
    assert bank.items[0].domain_stats["support"] == [0.0, 1.0]
    # Plain counters still accumulate across every episode regardless of domain.
    assert bank.items[0].wins == 1 and bank.items[0].uses == 2


def _seed_specialist() -> ReasoningMemoryBank:
    bank = ReasoningMemoryBank([item(id="specialist", lesson="travel rule")])
    for _ in range(6):
        bank.credit_assign([(["specialist"], True, "travel")])
    for _ in range(20):
        bank.credit_assign([(["specialist"], False, "support")])
    return bank


def test_domain_conditioned_prune_spares_an_item_strong_in_one_domain():
    """The out-of-domain eviction trap: a rule that is genuinely good in
    'travel' must not be deleted for failing in unrelated 'support' just
    because the pooled figure blends the two."""
    bank = _seed_specialist()
    # Pooled utility is dragged well under the cutoff by the unrelated domain.
    assert bank.items[0].utility < 0.34
    assert bank.items[0].utility_for("travel") >= 0.34

    pooled = bank.prune(min_uses=4, min_utility=0.34, domain_conditioned=False)
    assert [d.id for d in pooled] == ["specialist"]

    conditioned = _seed_specialist().prune(min_uses=4, min_utility=0.34, domain_conditioned=True)
    assert conditioned == []


def test_domain_conditioned_prune_still_drops_an_item_weak_everywhere():
    bank = ReasoningMemoryBank([item(id="bad", lesson="misleads")])
    for _ in range(8):
        bank.credit_assign([(["bad"], False, "any-domain")])
    dropped = bank.prune(min_uses=4, min_utility=0.34, domain_conditioned=True)
    assert [d.id for d in dropped] == ["bad"]


def test_archive_mode_prune_does_not_delete():
    bank = ReasoningMemoryBank(
        [item(id="stale", lesson="connection pool exhausted, raise the limit")]
    )
    for _ in range(5):
        bank.credit_assign([(["stale"], False)])

    dropped = bank.prune(min_uses=4, min_utility=0.34, archive=True)
    assert [d.id for d in dropped] == ["stale"]
    assert len(bank) == 0
    assert len(bank.archived) == 1
    # archive=True must not touch the store's delete path.
    assert bank.store is None


def test_revive_picks_the_best_lexical_match_and_moves_it_back():
    # Real fallback embeddings (process-stable, see the first test in this
    # file), same as every other retrieval-quality test here -- ranking by
    # relative similarity, not an absolute threshold, matches how
    # test_utility_weighting_demotes_a_losing_memory already validates this
    # embedder's behaviour.
    bank = ReasoningMemoryBank()
    bank.archived = [
        item(id="stale", lesson="connection pool exhausted, raise the limit"),
        item(id="unrelated", lesson="the weather forecast for tomorrow"),
    ]
    revived = bank.revive("connection pool exhausted", k=1, min_similarity=0.0)
    assert [r.id for r in revived] == ["stale"]
    assert [m.id for m in bank.items] == ["stale"]
    assert [a.id for a in bank.archived] == ["unrelated"]


def test_delete_mode_prune_never_populates_archived():
    """Default behaviour: archive=False still deletes, exactly as before
    archive mode existed."""
    bank = ReasoningMemoryBank([item(id="bad", lesson="misleads")])
    for _ in range(5):
        bank.credit_assign([(["bad"], False)])
    bank.prune(min_uses=4, min_utility=0.34)
    assert bank.archived == []


def test_revive_respects_k():
    """Even with two equally-plausible matches, no more than k come back."""
    bank = ReasoningMemoryBank()
    bank.archived = [
        item(id="a", lesson="connection pool exhausted"),
        item(id="b", lesson="connection pool queue depth"),
    ]
    revived = bank.revive("connection pool", k=1, min_similarity=0.0)
    assert len(revived) == 1
    assert len(bank.archived) == 1  # one left behind by k=1


def test_revive_respects_min_similarity():
    """A precomputed, orthogonal embedding never clears a real threshold."""
    bank = ReasoningMemoryBank()
    bank.embedder = _FixedEmbedder([1.0, 0.0, 0.0])
    far = item(id="far", lesson="unrelated", embedding=[0.0, 1.0, 0.0])
    bank.archived = [far]
    assert bank.revive("query", k=5, min_similarity=0.9) == []
    assert len(bank.archived) == 1


def test_archive_and_reload_round_trip_through_a_sidecar_file(tmp_path):
    path = tmp_path / "memory.jsonl"
    bank = ReasoningMemoryBank(
        [item(id="stale", lesson="connection pool exhausted, raise the limit")], path=path
    )
    for _ in range(5):
        bank.credit_assign([(["stale"], False)])
    bank.prune(min_uses=4, min_utility=0.34, archive=True)
    bank.save()

    archive_file = tmp_path / "memory.archive.jsonl"
    assert archive_file.exists()

    reloaded = ReasoningMemoryBank.load(path)
    assert len(reloaded.items) == 0
    assert len(reloaded.archived) == 1
    assert reloaded.archived[0].id == "stale"


def test_reload_with_no_archive_file_leaves_archived_empty(tmp_path):
    path = tmp_path / "memory.jsonl"
    bank = ReasoningMemoryBank([item(id="a", lesson="x")], path=path)
    bank.save()
    reloaded = ReasoningMemoryBank.load(path)
    assert reloaded.archived == []


class _FixedEmbedder(Embedder):
    """Embedder returning one constant query vector."""

    def __init__(self, vector):
        super().__init__(embeddings=None)
        self.vector = vector

    def embed(self, texts):
        return [list(self.vector) for _ in texts]
