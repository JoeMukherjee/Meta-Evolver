"""The ontology, its validator, and the auto-repair pass.

A knowledge graph that a model writes into rots: the same entity under three
spellings, a relation invented on the spot, an edge pointing the wrong way, a
``is_a`` cycle that makes inference loop. Left alone, retrieval quality decays
with every generation -- the graph analogue of a memory bank that "only grows
and never unlearns". So the schema is data, validation is a first-class pass,
and repair is deterministic where it can be.

The shape is SHACL's, reduced to what this graph needs: node-label and
relation vocabularies, per-relation domain/range, *functional* relations
(at most one active target) and *acyclic* relations. :func:`validate` returns
:class:`Violation`\\ s; :func:`repair` fixes the ones with an unambiguous
answer and returns an auditable :class:`RepairReport` -- every action it took,
and every violation it declined to guess about. Nothing is silently dropped:
an unfixable edge is *invalidated* (kept, time-stamped, inspectable), never
deleted, so a wrong repair is recoverable.

An optional ``fallback`` callable receives the residue, which is where an LLM
repairer (the SHACL-violation-to-SPARQL-update style of the recent KG-repair
literature) plugs in. Off by default: this module makes no model call.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from meta_evolver.knowledge.graph import Edge, KnowledgeGraph, slug

#: Catch-all label for something extracted but not yet classified.
GENERIC = "Entity"


@dataclass(frozen=True)
class RelationSpec:
    domain: frozenset[str] = frozenset()  # empty = any label
    range: frozenset[str] = frozenset()
    functional: bool = False  # at most one active target per source
    acyclic: bool = False  # no directed cycles (is_a, supersedes)
    symmetric: bool = False
    # Name of the converse relation (``mitigated_by`` <-> ``mitigates``). The
    # reasoner uses it to derive the reverse edge; validation ignores it.
    inverse: str | None = None


def _rel(domain=(), range=(), **kw) -> RelationSpec:
    return RelationSpec(frozenset(domain), frozenset(range), **kw)


@dataclass
class Ontology:
    labels: set[str]
    relations: dict[str, RelationSpec]
    label_aliases: dict[str, str] = field(default_factory=dict)
    relation_aliases: dict[str, str] = field(default_factory=dict)

    def canonical_label(self, label: str) -> str | None:
        if label in self.labels:
            return label
        for cand in (label.strip().title().replace(" ", ""), self.label_aliases.get(label.lower(), "")):
            if cand in self.labels:
                return cand
        return None

    def canonical_relation(self, rel: str) -> str | None:
        if rel in self.relations:
            return rel
        norm = rel.strip().lower().replace(" ", "_").replace("-", "_")
        if norm in self.relations:
            return norm
        mapped = self.relation_aliases.get(norm)
        return mapped if mapped in self.relations else None


def agent_ontology() -> Ontology:
    """The vocabulary Meta-Evolver's own learning loop writes."""
    learnable = ("Memory", "Skill")
    return Ontology(
        labels={
            "Task", "Episode", "Memory", "Skill", "Tool", "FailureMode",
            "Concept", "Domain", GENERIC,
        },
        relations={
            "on_task": _rel(["Episode"], ["Task"], functional=True),
            "in_domain": _rel(["Task", "Episode", *learnable], ["Domain"]),
            "involves": _rel(["Task"], ["Concept", "Tool"]),
            "used_tool": _rel(["Episode"], ["Tool"]),
            "cited": _rel(["Episode"], list(learnable)),
            "failed_with": _rel(["Episode"], ["FailureMode"]),
            # The transfer edge: evidence (wins/uses) that a strategy works for
            # tasks touching this concept/domain. Out-of-distribution retrieval
            # walks it.
            "applies_to": _rel(learnable, ["Concept", "Domain", "Tool"]),
            "mitigated_by": _rel(["FailureMode"], list(learnable), inverse="mitigates"),
            "mitigates": _rel(learnable, ["FailureMode"], inverse="mitigated_by"),
            # Induced by ``lifting`` (from evidence) / derived by ``reasoner``
            # (from rules); both are marked in edge props and invalidatable.
            "similar_to": _rel(["Concept"], ["Concept"], symmetric=True),
            "related_to": _rel(["FailureMode"], ["Concept"]),
            "uses_tool": _rel(["Skill"], ["Tool"]),
            "derived_from": _rel(["Skill", "Memory"], ["Episode"]),
            "supersedes": _rel(["Skill"], ["Skill"], acyclic=True),
            "is_a": _rel(["Concept"], ["Concept"], acyclic=True),
            "contradicts": _rel(learnable, learnable, symmetric=True),
        },
        label_aliases={"person": "Entity", "object": "Entity", "failure": "FailureMode", "strategy": "Memory"},
        relation_aliases={
            "caused_by": "failed_with", "fixed_by": "mitigated_by", "solved_by": "mitigated_by",
            "uses": "used_tool", "requires": "involves", "relates_to": "involves",
            "helps_with": "applies_to", "kind_of": "is_a", "subclass_of": "is_a",
            "replaces": "supersedes", "conflicts_with": "contradicts",
        },
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

UNKNOWN_LABEL = "unknown_label"
UNKNOWN_REL = "unknown_relation"
DANGLING = "dangling_edge"
DOMAIN = "domain_violation"
RANGE = "range_violation"
FUNCTIONAL = "functional_violation"
CYCLE = "cycle"
DUPLICATE = "duplicate_node"


@dataclass
class Violation:
    code: str
    message: str
    edge_ids: list[int] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)


def validate(graph: KnowledgeGraph, ontology: Ontology) -> list[Violation]:
    out: list[Violation] = []
    nodes = {n.id: n for n in graph.nodes()}

    for n in nodes.values():
        if n.label not in ontology.labels:
            out.append(Violation(UNKNOWN_LABEL, f"{n.id}: label {n.label!r} is not in the ontology", node_ids=[n.id]))

    edges = graph.edges()
    by_src_rel: dict[tuple[str, str], list[Edge]] = defaultdict(list)
    for e in edges:
        spec = ontology.relations.get(e.rel)
        if e.src not in nodes or e.dst not in nodes:
            out.append(Violation(DANGLING, f"edge {e.id} {e.src}-{e.rel}->{e.dst} has a missing endpoint", edge_ids=[e.id]))
            continue
        if spec is None:
            out.append(Violation(UNKNOWN_REL, f"edge {e.id}: relation {e.rel!r} is not in the ontology", edge_ids=[e.id]))
            continue
        if spec.domain and nodes[e.src].label not in spec.domain:
            out.append(Violation(DOMAIN, f"edge {e.id}: {e.rel} cannot start at a {nodes[e.src].label}", edge_ids=[e.id]))
        elif spec.range and nodes[e.dst].label not in spec.range:
            out.append(Violation(RANGE, f"edge {e.id}: {e.rel} cannot end at a {nodes[e.dst].label}", edge_ids=[e.id]))
        if spec.functional:
            by_src_rel[(e.src, e.rel)].append(e)

    for (src, rel), group in by_src_rel.items():
        if len(group) > 1:
            out.append(Violation(FUNCTIONAL, f"{src} has {len(group)} active {rel!r} targets", edge_ids=[e.id for e in group]))

    for rel, spec in ontology.relations.items():
        if spec.acyclic:
            for cycle_edges in _find_cycles([e for e in edges if e.rel == rel]):
                out.append(Violation(CYCLE, f"cycle through {rel!r}", edge_ids=[e.id for e in cycle_edges]))

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for n in nodes.values():
        groups[(n.label, slug(n.name).replace("-", ""))].append(n.id)
    for ids in groups.values():
        if len(ids) > 1:
            out.append(Violation(DUPLICATE, f"{len(ids)} nodes are the same entity: {ids}", node_ids=ids))
    return out


def _find_cycles(edges: list[Edge]) -> list[list[Edge]]:
    """One edge set per cycle found (DFS back-edges), enough to break each."""
    adj: dict[str, list[Edge]] = defaultdict(list)
    for e in edges:
        adj[e.src].append(e)
    color: dict[str, int] = {}
    cycles: list[list[Edge]] = []

    def dfs(u: str, stack: list[Edge]) -> None:
        color[u] = 1
        for e in adj.get(u, []):
            if color.get(e.dst, 0) == 0:
                dfs(e.dst, [*stack, e])
            elif color.get(e.dst) == 1:
                path = [*stack, e]
                start = next((i for i, x in enumerate(path) if x.src == e.dst), 0)
                cycles.append(path[start:])
        color[u] = 2

    for node in list(adj):
        if color.get(node, 0) == 0:
            dfs(node, [])
    return cycles


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


@dataclass
class RepairReport:
    actions: list[str] = field(default_factory=list)
    unresolved: list[Violation] = field(default_factory=list)

    @property
    def fixed(self) -> int:
        return len(self.actions)


def repair(
    graph: KnowledgeGraph,
    ontology: Ontology,
    violations: list[Violation] | None = None,
    generation: int = 0,
    fallback: Callable[[KnowledgeGraph, list[Violation]], list[str]] | None = None,
) -> RepairReport:
    """Fix every violation that has an unambiguous repair.

    Order matters and is fixed: vocabulary first (so aliases collapse before
    anything is judged against domain/range), then duplicates (merging changes
    which edges exist), then structure. Each is validated again afterwards by
    the caller; this function does one pass.
    """
    report = RepairReport()
    violations = violations if violations is not None else validate(graph, ontology)
    by_code: dict[str, list[Violation]] = defaultdict(list)
    for v in violations:
        by_code[v.code].append(v)

    for v in by_code[UNKNOWN_LABEL]:
        nid = v.node_ids[0]
        node = graph.get_node(nid)
        if node is None:
            continue
        target = ontology.canonical_label(node.label) or GENERIC
        new = graph.relabel(nid, target)
        report.actions.append(f"relabel {nid} -> {new}")

    current = {x.id: x for x in graph.edges()}
    for v in by_code[UNKNOWN_REL]:
        for eid in v.edge_ids:
            e = current.get(eid)
            if e is None:
                continue
            canon = ontology.canonical_relation(e.rel)
            if canon:
                graph.set_edge(eid, rel=canon)
                report.actions.append(f"rename relation {e.rel} -> {canon} (edge {eid})")
            else:
                graph.invalidate(eid, generation)
                report.actions.append(f"quarantine edge {eid}: unknown relation {e.rel!r}")

    for v in by_code[DUPLICATE]:
        ids = [i for i in v.node_ids if graph.get_node(i)]
        if len(ids) < 2:
            continue
        keep = max(ids, key=lambda i: (graph.degree(i), -len(i)))
        for drop in ids:
            if drop != keep:
                graph.merge_nodes(keep, drop)
                report.actions.append(f"merge {drop} into {keep}")

    live = {e.id: e for e in graph.edges()}

    for v in by_code[DANGLING]:
        for eid in v.edge_ids:
            graph.delete_edge(eid)
            report.actions.append(f"delete dangling edge {eid}")

    for v in by_code[DOMAIN] + by_code[RANGE]:
        for eid in v.edge_ids:
            e = live.get(eid)
            if e is None:
                continue
            spec = ontology.relations.get(e.rel)
            if spec is None:
                continue
            fixed = _fix_label_or_direction(graph, ontology, e, spec)
            if fixed:
                report.actions.append(fixed)
            else:
                graph.invalidate(eid, generation)
                report.actions.append(f"quarantine edge {eid}: {e.src}-{e.rel}->{e.dst} violates domain/range")

    for v in by_code[FUNCTIONAL]:
        group = [e for e in (live.get(i) for i in v.edge_ids) if e is not None and e.valid_to is None]
        if len(group) < 2:
            continue
        group.sort(key=lambda e: (e.confidence, e.valid_from, e.id))
        for loser in group[:-1]:
            graph.invalidate(loser.id, generation)
            report.actions.append(f"invalidate edge {loser.id}: functional {loser.rel} keeps the best-evidenced target")

    for v in by_code[CYCLE]:
        cyc = [e for e in (live.get(i) for i in v.edge_ids) if e is not None]
        cyc = [e for e in cyc if any(x.id == e.id for x in graph.edges())]
        if not cyc:
            continue  # an earlier fix already broke it
        weakest = min(cyc, key=lambda e: (e.confidence, -e.valid_from, -e.id))
        graph.invalidate(weakest.id, generation)
        report.actions.append(f"break cycle: invalidate edge {weakest.id} ({weakest.src}-{weakest.rel}->{weakest.dst})")

    residue = validate(graph, ontology) if report.actions else violations
    if fallback is not None and residue:
        report.actions += list(fallback(graph, residue))
        residue = validate(graph, ontology)
    report.unresolved = residue
    return report


def _fix_label_or_direction(graph: KnowledgeGraph, onto: Ontology, e: Edge, spec: RelationSpec) -> str | None:
    """Two repairs with exactly one defensible answer.

    1. An endpoint labelled with the catch-all ``Entity`` where the relation
       allows a single specific label: it is that label.
    2. The edge is simply backwards: swapping src/dst satisfies the spec.
    """
    src, dst = graph.get_node(e.src), graph.get_node(e.dst)
    if src is None or dst is None:
        return None
    if src.label == GENERIC and len(spec.domain) == 1 and dst.label in (spec.range or {dst.label}):
        new = graph.relabel(src.id, next(iter(spec.domain)))
        return f"relabel {src.id} -> {new} (domain of {e.rel})"
    if dst.label == GENERIC and len(spec.range) == 1 and src.label in (spec.domain or {src.label}):
        new = graph.relabel(dst.id, next(iter(spec.range)))
        return f"relabel {dst.id} -> {new} (range of {e.rel})"
    if (not spec.domain or dst.label in spec.domain) and (not spec.range or src.label in spec.range):
        graph.set_edge(e.id, src=e.dst, dst=e.src)
        return f"reverse edge {e.id}: {e.rel} points from {dst.label} to {src.label}"
    return None


def health(graph: KnowledgeGraph, ontology: Ontology) -> dict[str, Any]:
    """Counts a run report can show: is the graph getting cleaner or dirtier?"""
    v = validate(graph, ontology)
    by_code: dict[str, int] = defaultdict(int)
    for x in v:
        by_code[x.code] += 1
    return {"violations": len(v), "by_code": dict(by_code), **graph.stats()}
