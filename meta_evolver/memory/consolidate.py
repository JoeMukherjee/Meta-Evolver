"""Offline memory consolidation -- SimpleMem's "sleep cycle" pass.

SimpleMem's own mechanism is asynchronous recursive consolidation: an
offline phase, separate from live retrieval, that merges related memory
units into fewer, more general ones. ``ReasoningMemoryBank`` already has a
write-time dedup (near-*exact* duplicates, at a tight cosine threshold,
merged the instant an item is added), which is a different thing -- this
module catches items that are related but not close enough to trigger that,
things two separate episodes independently distilled into two separate
memories about, in substance, the same lesson.

Deliberately no model call: merging picks whichever cluster member already
has the strongest track record and keeps its text as-is, pooling the
others' usage statistics into it, rather than asking a model to write a new
summary. That is what keeps this "zero cost" in the sense SimpleMem's own
efficiency claim means -- an embedding pass over items the bank already
holds and has already paid to embed, not a new API call per cluster.
"""
from __future__ import annotations

from meta_evolver.core.types import MemoryItem
from meta_evolver.llm.embeddings import cosine
from meta_evolver.memory.bank import ReasoningMemoryBank


def _clusters(items: list[MemoryItem], threshold: float) -> list[list[int]]:
    """Single-linkage clustering over cosine similarity, via union-find.

    Single-linkage on purpose: three memories where A-B and B-C are both
    similar enough to merge, even if A-C alone would not be, should still
    end up as one cluster -- B is genuinely the same lesson as each, so
    transitively they are all versions of the same thing.
    """
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        vi = items[i].embedding
        if not vi:
            continue
        for j in range(i + 1, n):
            vj = items[j].embedding
            if not vj:
                continue
            if cosine(vi, vj) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _merge(anchor: MemoryItem, members: list[MemoryItem]) -> None:
    """Fold ``members`` (excluding ``anchor``) into ``anchor`` in place."""
    for m in members:
        anchor.uses += m.uses
        anchor.wins += m.wins
        for domain, (w, u) in m.domain_stats.items():
            aw, au = anchor.domain_stats.get(domain, [0.0, 0.0])
            anchor.domain_stats[domain] = [aw + w, au + u]
        for trigger in m.triggers:
            if trigger not in anchor.triggers:
                anchor.triggers.append(trigger)
        for task_id in m.source_task_ids:
            if task_id not in anchor.source_task_ids:
                anchor.source_task_ids.append(task_id)
        anchor.last_used_generation = max(anchor.last_used_generation, m.last_used_generation)


def consolidate(
    bank: ReasoningMemoryBank, similarity_threshold: float = 0.80, min_cluster_size: int = 2
) -> list[MemoryItem]:
    """Merge related-but-not-identical items in ``bank`` in place.

    ``similarity_threshold`` (default 0.80) is deliberately looser than the
    bank's own write-time ``dedup_similarity`` (default 0.93) -- the two
    thresholds catch different things: write-time dedup is near-exact
    duplicates at add time, this is thematically related items an entire
    run has accumulated, on purpose set loose enough to actually find
    clusters a tighter threshold never would.

    Only merges items sharing a ``polarity`` (a success strategy and a
    failure anti-pattern are never the same lesson, however similar their
    text) and only clusters of at least ``min_cluster_size`` -- a singleton
    "cluster" is just one item on its own, nothing to merge.

    Returns the items removed (merged away), so a caller can log what
    happened; ``bank.items`` is updated in place, matching
    :meth:`~meta_evolver.memory.bank.ReasoningMemoryBank.prune`'s own
    return-the-dropped-items convention.
    """
    removed: list[MemoryItem] = []
    merges: list[tuple[MemoryItem, list[MemoryItem]]] = []
    for polarity in ("success", "failure"):
        group = [it for it in bank.items if it.polarity == polarity]
        if len(group) < min_cluster_size:
            continue
        for cluster_idxs in _clusters(group, similarity_threshold):
            if len(cluster_idxs) < min_cluster_size:
                continue
            cluster = [group[i] for i in cluster_idxs]
            # The strongest track record keeps its text and id; everything
            # else's evidence pools into it rather than being discarded.
            anchor = max(cluster, key=lambda it: (it.utility, it.uses))
            members = [it for it in cluster if it is not anchor]
            _merge(anchor, members)
            removed.extend(members)
            merges.append((anchor, members))

    if removed:
        removed_ids = {it.id for it in removed}
        bank.items = [it for it in bank.items if it.id not in removed_ids]
        if bank.store is not None:
            bank.store.delete(list(removed_ids))
            # upsert() deliberately never overwrites uses/wins (protecting
            # against an unrelated reword silently resetting a proven
            # memory to "unproven" -- see its own docstring), so it cannot
            # carry a merge's pooled counters into the store. Every other
            # field DOES go through it; the counters go through
            # record_usage() instead, replaying each absorbed member's
            # historical wins and losses onto the anchor -- the store's own
            # sanctioned way to change them, and numerically identical to
            # what _merge() just did in memory.
            bank.store.upsert([anchor for anchor, _ in merges])
            events: list[tuple[str, bool]] = []
            for anchor, members in merges:
                for member in members:
                    events.extend([(anchor.id, True)] * member.wins)
                    events.extend([(anchor.id, False)] * (member.uses - member.wins))
            if events:
                bank.store.record_usage(events)
    return removed


__all__ = ["consolidate"]
