"""Structure-based recall: what the graph recommends for a task it may never
have seen worded this way.

Two signals, both read straight off the graph:

1. **Concept transfer.** For each concept in the new task, find the strategies
   and skills that have *evidence* of working on tasks involving it
   (``applies_to``). Rarer concepts count for more (inverse-degree weight, an
   IDF), so the word ``heat`` outranks the word ``object``. A strategy with
   more wins than losses on a concept scores up; one that mostly lost there is
   excluded -- the graph can say "this is a known bad fit" where a similarity
   score cannot.
2. **Failure recall.** If this exact task has failed before, and a later
   success cited something, the ``FailureMode -mitigated_by-> X`` edge
   surfaces X first: "this task broke this way before; this fixed it".

The result is a ranked list of ids with a human-readable reason, to be merged
with -- never to replace -- the bank's own similarity retrieval. Callers
resolve ids to ``MemoryItem``/``Skill`` objects.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from meta_evolver.knowledge.graph import KnowledgeGraph, node_id
from meta_evolver.knowledge.ingest import extract_concepts


@dataclass
class Recommendation:
    kind: str  # "Memory" | "Skill"
    ref: str  # the id the bank / skill library knows it by
    score: float
    why: str


class GraphMemory:
    def __init__(self, graph: KnowledgeGraph, min_confidence: float = 0.4) -> None:
        self.graph = graph
        self.min_confidence = min_confidence

    def recommend(
        self,
        instruction: str,
        task_id: str = "",
        domain: str = "",
        exclude: Iterable[str] = (),
        k: int = 4,
    ) -> list[Recommendation]:
        g = self.graph
        skip = {node_id("Memory", x) for x in exclude} | {node_id("Skill", x) for x in exclude}
        scores: dict[str, float] = {}
        why: dict[str, list[str]] = {}

        def bump(nid: str, amount: float, reason: str) -> None:
            if nid in skip:
                return
            scores[nid] = scores.get(nid, 0.0) + amount
            why.setdefault(nid, []).append(reason)

        for concept in extract_concepts(instruction):
            cid = node_id("Concept", concept)
            if g.get_node(cid) is None:
                continue
            # IDF-like weight: a concept touched by many nodes is less specific.
            weight = 1.0 / math.log(2.0 + g.degree(cid))
            for edge in g.edges(dst=cid, rel="applies_to"):
                if edge.confidence >= self.min_confidence:
                    bump(edge.src, weight * edge.confidence, f"worked on tasks involving '{concept}'")

        if domain:
            did = node_id("Domain", domain)
            for edge in g.edges(dst=did, rel="applies_to"):
                if edge.confidence >= self.min_confidence:
                    bump(edge.src, 0.1 * edge.confidence, f"worked in domain '{domain}'")

        if task_id:
            tid = node_id("Task", task_id)
            for on_task in g.edges(dst=tid, rel="on_task"):
                for failed in g.edges(src=on_task.src, rel="failed_with"):
                    for fix in g.edges(src=failed.dst, rel="mitigated_by"):
                        if fix.confidence >= self.min_confidence:
                            bump(fix.dst, 1.0 + fix.confidence, "fixed an earlier failure on this task")

        out: list[Recommendation] = []
        for nid, score in sorted(scores.items(), key=lambda kv: -kv[1]):
            node = g.get_node(nid)
            if node is None or node.label not in ("Memory", "Skill"):
                continue
            out.append(Recommendation(node.label, node.name, score, "; ".join(dict.fromkeys(why[nid]))[:160]))
            if len(out) >= k:
                break
        return out
