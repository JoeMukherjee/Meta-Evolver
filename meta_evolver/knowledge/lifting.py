"""Concept lifting: induce links *between* concepts from graph evidence.

The problem
-----------
Concepts are extracted lexically, so ``heat`` and ``warm`` are two unrelated
nodes and a strategy that only ever won on ``heat`` tasks is invisible to a
``warm`` task. An embedding would blur them together; but the graph already
holds a better, auditable signal: *which strategies actually won on each*.

The rule
--------
Let ``W(c)`` be the set of Memory/Skill nodes with *winning* ``applies_to``
evidence on concept ``c`` (at least one win, Beta mean >= ``min_win_confidence``).

* ``similar_to`` -- ``|W(a) & W(b)| >= min_support`` and Jaccard
  ``>= min_jaccard``. Symmetric, so stored once with ``src < dst``.
* ``is_a`` -- ``W(a)`` is a *strict* subset of ``W(b)``, ``|W(a)| >=
  min_support`` and the pair is not already similar: everything that works for
  the narrow concept also works for the broad one, but not the reverse.

Why this is safe to leave on
----------------------------
* Induced edges are *inferred*: props carry ``inferred=True``, the ``rule`` and
  the support counts, and evidence is recorded as wins=|intersection|,
  uses=|union|, so the edge's confidence is a Jaccard-flavoured Beta mean that
  weakens transfer for loosely-overlapping concepts.
* They are ordinary edges, so they can be invalidated (``retract_lifted``) and
  recomputed; a pair whose evidence no longer holds is retired on the next run.
* An asserted edge on the same pair is never touched or duplicated.
* ``is_a`` is acyclic: a strict subset order cannot cycle by itself, but the
  graph may hold asserted ``is_a`` edges too, so each new edge is checked
  against reachability first.

Known limit: ingest links every concept of a task to the same winners, so two
words that always co-occur in instructions look identical here. That is a
property of the evidence, not a bug; ``min_support`` is the guard.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from meta_evolver.knowledge.graph import KnowledgeGraph

SIMILAR = "similar_to"
IS_A = "is_a"
LIFT_RULE = "winning-set-overlap"


@dataclass
class LiftReport:
    added: list[int] = field(default_factory=list)  # new edge ids
    refreshed: list[int] = field(default_factory=list)  # replaced because support changed
    retired: list[int] = field(default_factory=list)  # evidence no longer holds

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.refreshed) + len(self.retired)


def winning_sets(graph: KnowledgeGraph, min_win_confidence: float = 0.5) -> dict[str, set[str]]:
    """``concept id -> {Memory/Skill ids with winning applies_to evidence}``."""
    labels = {n.id: n.label for n in graph.nodes()}
    out: dict[str, set[str]] = defaultdict(set)
    for e in graph.edges(rel="applies_to"):
        if labels.get(e.dst) != "Concept" or labels.get(e.src) not in ("Memory", "Skill"):
            continue
        if e.wins >= 1 and e.confidence >= min_win_confidence:
            out[e.dst].add(e.src)
    return out


def induce_concept_links(
    graph: KnowledgeGraph,
    generation: int = 0,
    min_support: int = 2,
    min_jaccard: float = 0.5,
    min_win_confidence: float = 0.5,
) -> LiftReport:
    """Write ``similar_to`` / ``is_a`` edges between concepts. Idempotent."""
    report = LiftReport()
    wins = winning_sets(graph, min_win_confidence)

    # Intersections via an inverted index (strategy -> its concepts), so the
    # cost is per co-winning pair, not per concept pair.
    inter: dict[tuple[str, str], int] = defaultdict(int)
    by_strategy: dict[str, list[str]] = defaultdict(list)
    for c, members in wins.items():
        for m in members:
            by_strategy[m].append(c)
    for concepts in by_strategy.values():
        concepts.sort()
        for i, a in enumerate(concepts):
            for b in concepts[i + 1 :]:
                inter[(a, b)] += 1

    desired: dict[tuple[str, str, str], dict] = {}
    for (a, b), n in sorted(inter.items()):
        if n < min_support:
            continue
        wa, wb = len(wins[a]), len(wins[b])
        union = wa + wb - n
        if n / union >= min_jaccard:
            desired[(a, SIMILAR, b)] = {"intersection": n, "union": union, "jaccard": round(n / union, 4)}
        elif n == wa and wa < wb:  # W(a) strict subset of W(b)
            desired[(a, IS_A, b)] = {"intersection": n, "union": union, "narrow": wa, "broad": wb}
        elif n == wb and wb < wa:
            desired[(b, IS_A, a)] = {"intersection": n, "union": union, "narrow": wb, "broad": wa}

    # Existing induced edges are the baseline we reconcile against.
    induced = {e.id: e for rel in (SIMILAR, IS_A) for e in graph.edges(rel=rel) if e.props.get("inferred")}
    asserted_pairs = {
        frozenset((e.src, e.dst)) for rel in (SIMILAR, IS_A) for e in graph.edges(rel=rel) if not e.props.get("inferred")
        and not e.props.get("derived")
    }
    keep: set[int] = set()

    for (src, rel, dst), support in desired.items():
        if frozenset((src, dst)) in asserted_pairs:
            continue  # someone already said something about this pair; do not argue
        current = next((e for e in induced.values() if (e.src, e.rel, e.dst) == (src, rel, dst)), None)
        if current is not None:
            if current.props.get("support") == support:
                keep.add(current.id)
                continue
            graph.invalidate(current.id, generation)  # support changed: replace, keep history
            report.refreshed.append(current.id)
        if rel == IS_A and src in graph.traverse([dst], hops=50, rels=[IS_A], direction="out"):
            continue  # would close a cycle through asserted is_a edges
        props = {"inferred": True, "rule": LIFT_RULE, "support": support}
        eid = graph.add_edge(src, rel, dst, generation, props=props)
        for _ in range(support["intersection"]):
            graph.add_edge(src, rel, dst, generation, evidence=True)
        for _ in range(support["union"] - support["intersection"]):
            graph.add_edge(src, rel, dst, generation, evidence=False)
        keep.add(eid)
        if current is None:
            report.added.append(eid)

    for eid in induced:
        if eid not in keep and not any(eid == r for r in report.refreshed):
            graph.invalidate(eid, generation)
            report.retired.append(eid)
    return report


def retract_lifted(graph: KnowledgeGraph, generation: int = 0) -> int:
    """Invalidate every active induced concept link (history is kept)."""
    n = 0
    for rel in (SIMILAR, IS_A):
        for e in graph.edges(rel=rel):
            if e.props.get("inferred"):
                graph.invalidate(e.id, generation)
                n += 1
    return n
