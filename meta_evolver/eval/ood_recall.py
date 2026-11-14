"""Does graph recall help *as the task drifts away from what was learned*?

The claim behind the knowledge graph is narrow and testable: similarity
retrieval degrades as a task is worded further from its training tasks, and
structural recall (shared concepts, an earlier failure on the same task) does
not degrade as fast. This module measures exactly that, offline, from
trajectories a run already produced -- no model call, no API key.

Method
------
* **Learn** from ``train`` trajectories: the knowledge graph is built from
  them alone.
* **Test** on held-out *successful* episodes. The gold answer for one is the
  set of memory ids that were in its prompt when it succeeded. Gold that no
  training episode ever cited is dropped (unreachable by *any* retriever, so
  counting it would only add noise to every column equally).
* **Rank** with each retriever and score recall@k: did any gold id make the
  top k?
* **Bucket by distribution shift.** A test task's shift is
  ``1 - (max concept-overlap with any training instruction)``. Reporting recall per
  bucket is the point: a single average hides that the interesting cases are
  the shifted ones.

Retrievers compared: ``similarity`` (TF-IDF cosine over memory text -- a
dependency-free stand-in for the embedding retriever; pass your own
``text_retriever`` to use the real one), ``graph``, and ``fused``
(reciprocal-rank fusion of the two, which is how they are combined live).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence

from meta_evolver.core.types import Trajectory
from meta_evolver.knowledge import GraphIngestor, GraphMemory, KnowledgeGraph, extract_concepts

_TOK = re.compile(r"[a-z][a-z0-9_]+")


def shift_score(instruction: str, train_instructions: Iterable[str]) -> float:
    """0 = reworded copy of something seen; 1 = shares no concept with anything."""
    mine = set(extract_concepts(instruction, limit=30))
    if not mine:
        return 1.0
    best = 0.0
    for other in train_instructions:
        theirs = set(extract_concepts(other, limit=30))
        if theirs:
            best = max(best, len(mine & theirs) / len(mine | theirs))
    return 1.0 - best


class TfidfRetriever:
    """Minimal TF-IDF cosine over ``{id: text}``. Deliberately plain."""

    def __init__(self, docs: dict[str, str]) -> None:
        self.docs = {k: _TOK.findall(v.lower()) for k, v in docs.items()}
        df: Counter[str] = Counter(w for toks in self.docs.values() for w in set(toks))
        n = max(1, len(self.docs))
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}
        self.vecs = {k: self._vec(t) for k, t in self.docs.items()}

    def _vec(self, toks: Sequence[str]) -> dict[str, float]:
        tf = Counter(toks)
        v = {w: c * self.idf.get(w, 0.0) for w, c in tf.items() if w in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def top(self, query: str, k: int) -> list[str]:
        q = self._vec(_TOK.findall(query.lower()))
        scored = [(sum(q.get(w, 0.0) * x for w, x in v.items()), mid) for mid, v in self.vecs.items()]
        return [mid for s, mid in sorted(scored, key=lambda p: (-p[0], p[1])) if s > 0][:k]


def _rrf(rankings: Sequence[Sequence[str]], k: int, c: int = 60) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, mid in enumerate(ranking):
            score[mid] += 1.0 / (c + rank + 1)
    return [m for m, _ in sorted(score.items(), key=lambda p: -p[1])][:k]


def evaluate_ood_recall(
    train: Sequence[Trajectory],
    test: Sequence[Trajectory],
    memory_texts: dict[str, str],
    k: int = 3,
    shift_edges: tuple[float, float] = (0.4, 0.75),
    text_retriever: Callable[[str, int], list[str]] | None = None,
) -> dict:
    """Recall@k per retriever, overall and per distribution-shift bucket."""
    graph = KnowledgeGraph()
    GraphIngestor(graph).ingest([t for t in train if t.usable], generation=0)
    gmem = GraphMemory(graph)
    sim = text_retriever or TfidfRetriever(memory_texts).top
    train_instr = [t.instruction for t in train if t.usable]
    reachable = {m for t in train if t.usable for m in t.retrieved_memory_ids}

    def bucket(s: float) -> str:
        return "in-distribution" if s < shift_edges[0] else "shifted" if s < shift_edges[1] else "far-OOD"

    hits: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for t in test:
        gold = set(t.retrieved_memory_ids) & reachable
        if not (t.success and t.usable and gold):
            continue
        b = bucket(shift_score(t.instruction, train_instr))
        rank_sim = sim(t.instruction, k)
        rank_graph = [r.ref for r in gmem.recommend(t.instruction, task_id=t.task_id, domain=t.benchmark, k=k)]
        results = {
            "similarity": rank_sim,
            "graph": rank_graph,
            "fused": _rrf([rank_sim, rank_graph], k),
        }
        for name, ranking in results.items():
            h = int(bool(gold & set(ranking)))
            hits[name][b].append(h)
            hits[name]["all"].append(h)

    def rate(xs: list[int]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    names = ("similarity", "graph", "fused")
    buckets = ("in-distribution", "shifted", "far-OOD")
    return {
        "k": k,
        "n": len(hits["similarity"]["all"]),
        "recall": {n: rate(hits[n]["all"]) for n in names},
        "by_shift": {
            b: {"n": len(hits["similarity"][b]), **{n: rate(hits[n][b]) for n in names}} for b in buckets
        },
    }
