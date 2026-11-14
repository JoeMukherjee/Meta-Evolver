"""Episodes -> graph facts. Deterministic: no model call, so it costs nothing
per generation and cannot hallucinate an edge.

What gets written, per usable episode (infrastructure errors are skipped, for
the same reason the bank's credit pass skips them)::

    Episode -on_task-> Task -in_domain-> Domain
    Task    -involves-> Concept | Tool          (the structural features)
    Episode -used_tool-> Tool
    Episode -cited-> Memory | Skill             (what was in the prompt)
    Episode -failed_with-> FailureMode          (when it failed)
    Memory | Skill -applies_to-> Concept | Domain   (+ a win/loss: transfer evidence)
    FailureMode -mitigated_by-> Memory | Skill  (a later success on the same task)

The ``applies_to`` edges are the point. A strategy that won on tasks involving
the concept ``heat`` accrues evidence on ``applies_to heat`` -- and that is
what lets it be found for a *differently worded* task that also involves
heating, which text similarity alone cannot do.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from meta_evolver.core.types import Trajectory
from meta_evolver.knowledge.graph import KnowledgeGraph
from meta_evolver.knowledge.ontology import Ontology, agent_ontology

_STOP = frozenset(
    """a an the and or but if then else for to of in on at by with from into onto over under
    is are was were be been being do does did done have has had having this that these those
    it its you your we our they their them what which who whom how why when where while
    can could should would will shall may might must not no yes all any each every some such
    than too very just only also about after before again once here there out up down off
    use using used make made get got set put take find tell give need want please task
    step steps one two first second next last new old""".split()
)
_WORD = re.compile(r"[A-Za-z][A-Za-z_\-]{3,}")


def extract_concepts(text: str, limit: int = 8) -> list[str]:
    """Content words of ``text``, order-preserving and de-duplicated.

    Deliberately lexical rather than embedding-based: it is the *shared
    abstraction* between differently-worded tasks (``heat``, ``invoice``,
    ``rollback``) that out-of-distribution retrieval needs, and a bare word is
    an abstraction an embedding blurs away.
    """
    seen: dict[str, None] = {}
    for w in _WORD.findall(text or ""):
        w = w.lower().strip("-_")
        if len(w) >= 4 and w not in _STOP:
            seen.setdefault(w, None)
        if len(seen) >= limit:
            break
    return list(seen)


def action_verb(name: str, kwargs: dict[str, Any]) -> str:
    """The abstract action: a tool name, or the first word of a text action."""
    if name == "do" and isinstance(kwargs.get("text"), str) and kwargs["text"].split():
        return kwargs["text"].split()[0].lower()
    return name


def failure_signature(t: Trajectory) -> str:
    """A normalised fingerprint of *how* an episode failed."""
    last = t.steps[-1] if t.steps else None
    if last is None:
        return "no-steps"
    obs = re.sub(r"\d+", "#", (last.observation or "").lower())
    obs = re.sub(r"['\"`].*?['\"`]", "''", obs)
    words = re.findall(r"[a-z#']+", obs)[:8]
    return f"{action_verb(last.action.name, last.action.kwargs)}: {' '.join(words)}".strip()


class GraphIngestor:
    def __init__(self, graph: KnowledgeGraph, ontology: Ontology | None = None) -> None:
        self.graph = graph
        self.ontology = ontology or agent_ontology()

    def ingest(self, trajectories: Iterable[Trajectory], generation: int = 0) -> dict[str, int]:
        g = self.graph
        counts = {"episodes": 0, "edges": 0}
        for t in trajectories:
            if not t.usable:
                continue
            counts["episodes"] += 1
            domain = g.upsert_node("Domain", t.benchmark or "general", generation=generation)
            task = g.upsert_node("Task", t.task_id, {"instruction": (t.instruction or "")[:300]}, generation)
            ep = g.upsert_node(
                "Episode",
                f"{t.task_id}@g{t.generation}r{t.rollout_index}",
                {"success": t.success, "score": t.score, "steps": t.n_steps},
                generation,
            )
            g.add_edge(ep, "on_task", task, generation, functional=True)
            g.add_edge(task, "in_domain", domain, generation, functional=True)

            concepts = [g.upsert_node("Concept", c, generation=generation) for c in extract_concepts(t.instruction)]
            for c in concepts:
                g.add_edge(task, "involves", c, generation)

            verbs = list(dict.fromkeys(action_verb(s.action.name, s.action.kwargs) for s in t.steps))
            tools = [g.upsert_node("Tool", v, generation=generation) for v in verbs]
            for tool in tools:
                g.add_edge(ep, "used_tool", tool, generation)
                g.add_edge(task, "involves", tool, generation)

            cited: list[tuple[str, str]] = [("Memory", m) for m in t.retrieved_memory_ids]
            cited += [("Skill", s) for s in getattr(t, "retrieved_skill_ids", []) or []]
            for label, ident in cited:
                m = g.upsert_node(label, ident, generation=generation)
                g.add_edge(ep, "cited", m, generation)
                g.add_edge(m, "applies_to", domain, generation, evidence=t.success)
                for target in (*concepts, *tools):
                    g.add_edge(m, "applies_to", target, generation, evidence=t.success)

            if t.success:
                self._record_mitigations(task, [g.upsert_node(lbl, i) for lbl, i in cited], generation)
            else:
                fm = g.upsert_node("FailureMode", failure_signature(t), generation=generation)
                g.add_edge(ep, "failed_with", fm, generation)
        counts["edges"] = g.stats()["edges_active"]
        return counts

    def _record_mitigations(self, task: str, cited: list[str], generation: int) -> None:
        """A success citing M, on a task that earlier failed with F, is evidence
        that M mitigates F."""
        if not cited:
            return
        g = self.graph
        failed_modes: set[str] = set()
        for e in g.edges(dst=task, rel="on_task"):
            for f in g.edges(src=e.src, rel="failed_with"):
                failed_modes.add(f.dst)
        for fm in failed_modes:
            for m in cited:
                g.add_edge(fm, "mitigated_by", m, generation, evidence=True)
