"""A small forward-chaining reasoner over the graph.

Rules (all data-driven off the ontology, none hard-coded to a relation name
except the default property chain):

* **transitive**  -- ``a is_a b`` and ``b is_a c`` => ``a is_a c`` for every
  relation the ontology marks ``acyclic`` and that is ``is_a``-like
  (``TRANSITIVE_RELS``). ``supersedes`` is acyclic but its closure is not
  something to materialise, so it is opt-in.
* **symmetric**   -- ``a r b`` => ``b r a`` for ``symmetric`` relations.
* **inverse**     -- ``a r b`` => ``b inv(r) a``.
* **chain**       -- ``a r1 b`` and ``b r2 c`` => ``a r3 c`` (default:
  ``FailureMode -mitigated_by-> M`` and ``M -applies_to-> Concept`` =>
  ``FailureMode -related_to-> Concept``).

Guarantees
----------
* **Provenance.** A derived edge's props hold ``derived=True``, ``rule``, the
  ``premises`` (edge ids) and ``confidence`` = product of the premises'
  confidences (``Edge.confidence`` honours it), so a long chain is weaker than
  any link in it. :func:`explain` unrolls it.
* **Never overwrites.** If *any* active edge already has the triple
  (asserted or derived) nothing is written. (Limit: re-asserting a triple that
  is currently only derived strengthens the derived edge instead of creating
  an asserted one -- :func:`retract_derived` would then drop it. Retract first
  if you are about to assert.)
* **Valid by construction.** A derivation that would break the result
  relation's domain/range, create a self-loop, or reverse an acyclic edge is
  skipped, so ``validate`` stays clean.
* **Idempotent and terminating.** No node is ever created, so there are
  finitely many triples; each round adds only absent ones and stops when a
  round adds nothing (``max_rounds`` is a belt-and-braces bound).
* **Truth maintenance.** Before deriving, derived edges whose premises are no
  longer active are invalidated (cascading), so retracting a fact retracts
  its consequences.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from meta_evolver.knowledge.graph import Edge, KnowledgeGraph
from meta_evolver.knowledge.ontology import Ontology

TRANSITIVE_RELS = frozenset({"is_a"})


@dataclass(frozen=True)
class PropertyChain:
    name: str
    first: str
    second: str
    result: str


DEFAULT_CHAINS: tuple[PropertyChain, ...] = (
    PropertyChain("failure-relates-to-mitigator-concepts", "mitigated_by", "applies_to", "related_to"),
)


@dataclass
class ReasonReport:
    added: list[int] = field(default_factory=list)
    retired: list[int] = field(default_factory=list)  # derived edges whose premises vanished
    by_rule: dict[str, int] = field(default_factory=dict)
    rounds: int = 0

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.retired)


@dataclass
class Explanation:
    edge_id: int
    fact: str
    asserted: bool
    rule: str | None
    confidence: float
    premises: list["Explanation"] = field(default_factory=list)

    def render(self, indent: int = 0) -> str:
        how = "asserted" if self.asserted else f"derived by {self.rule}"
        lines = [f"{'  ' * indent}{self.fact}  [{how}, conf={self.confidence:.3f}, edge {self.edge_id}]"]
        lines += [p.render(indent + 1) for p in self.premises]
        return "\n".join(lines)


def is_derived(e: Edge) -> bool:
    return bool(e.props.get("derived"))


def reason(
    graph: KnowledgeGraph,
    ontology: Ontology,
    generation: int = 0,
    chains: tuple[PropertyChain, ...] = DEFAULT_CHAINS,
    min_confidence: float = 0.5,
    max_rounds: int = 20,
) -> ReasonReport:
    """Materialise the consequences of the active edges. Idempotent.

    ``min_confidence`` gates *premises*: a strategy that mostly lost on a
    concept must not make a failure mode "related" to it.
    """
    report = ReasonReport()
    report.retired = _retire_orphans(graph, generation)
    labels = {n.id: n.label for n in graph.nodes()}
    inverse = _inverse_map(ontology)

    def fits(rel: str, src: str, dst: str) -> bool:
        spec = ontology.relations.get(rel)
        if spec is None or src == dst:
            return False
        if spec.domain and labels.get(src) not in spec.domain:
            return False
        return not spec.range or labels.get(dst) in spec.range

    for _ in range(max_rounds):
        report.rounds += 1
        active = sorted(graph.edges(), key=lambda e: e.id)
        present = {(e.src, e.rel, e.dst) for e in active}
        by_src: dict[tuple[str, str], list[Edge]] = {}
        for e in active:
            by_src.setdefault((e.src, e.rel), []).append(e)
        # The gate applies to *asserted* premises only: a derived edge's
        # confidence is already a product and shrinks with depth, so gating it
        # again would stop closure after two hops.
        def ok(e: Edge) -> bool:
            return is_derived(e) or e.confidence >= min_confidence

        usable = [e for e in active if ok(e)]
        new: list[tuple[str, str, str, str, list[Edge]]] = []  # src, rel, dst, rule, premises

        for e in usable:
            spec = ontology.relations.get(e.rel)
            if spec is None:
                continue
            if spec.symmetric:
                new.append((e.dst, e.rel, e.src, "symmetric", [e]))
            if e.rel in inverse:
                new.append((e.dst, inverse[e.rel], e.src, "inverse", [e]))
            if e.rel in TRANSITIVE_RELS and spec.acyclic:
                for f in by_src.get((e.dst, e.rel), []):
                    if ok(f):
                        new.append((e.src, e.rel, f.dst, "transitive", [e, f]))
        for ch in chains:
            for e in usable:
                if e.rel != ch.first:
                    continue
                for f in by_src.get((e.dst, ch.second), []):
                    if ok(f):
                        new.append((e.src, ch.result, f.dst, ch.name, [e, f]))

        added_this_round = 0
        for src, rel, dst, rule, premises in new:
            key = (src, rel, dst)
            if key in present or not fits(rel, src, dst):
                continue
            spec = ontology.relations[rel]
            if spec.acyclic and (dst, rel, src) in present:
                continue  # closing a cycle: leave the asserted structure alone
            conf = 1.0
            for p in premises:
                conf *= p.confidence
            eid = graph.add_edge(
                src, rel, dst, generation,
                props={"derived": True, "rule": rule, "premises": [p.id for p in premises], "confidence": round(conf, 6)},
            )
            present.add(key)
            report.added.append(eid)
            report.by_rule[rule] = report.by_rule.get(rule, 0) + 1
            added_this_round += 1
        if added_this_round == 0:
            break
    return report


def explain(graph: KnowledgeGraph, edge_id: int, _seen: frozenset[int] = frozenset()) -> Explanation | None:
    """The premises chain behind an edge (``None`` if the id is unknown).

    Premises are read even if since invalidated: an explanation is a record of
    why the edge was believed.
    """
    e = graph.get_edge(edge_id)
    if e is None:
        return None
    derived = is_derived(e)
    premises: list[Explanation] = []
    if derived:
        for pid in e.props.get("premises", []):
            if pid in _seen or pid == edge_id:  # defensive: provenance must be a DAG
                continue
            sub = explain(graph, pid, _seen | {edge_id})
            if sub is not None:
                premises.append(sub)
    return Explanation(
        edge_id, f"{e.src} -{e.rel}-> {e.dst}", not derived,
        e.props.get("rule") if derived else None, e.confidence, premises,
    )


def retract_derived(graph: KnowledgeGraph) -> int:
    """Delete every derived edge (active or not) so a fresh :func:`reason`
    recomputes from the asserted facts alone. Asserted edges are never touched."""
    ids = [e.id for e in graph.edges(include_inactive=True) if is_derived(e)]
    for eid in ids:
        graph.delete_edge(eid)
    return len(ids)


def _retire_orphans(graph: KnowledgeGraph, generation: int) -> list[int]:
    """Invalidate derived edges with an inactive/missing premise, to a fixpoint."""
    retired: list[int] = []
    while True:
        active_ids = {e.id for e in graph.edges()}
        stale = [
            e.id for e in graph.edges()
            if is_derived(e) and any(p not in active_ids for p in e.props.get("premises", []))
        ]
        if not stale:
            return retired
        for eid in stale:
            graph.invalidate(eid, generation)
        retired += stale


def _inverse_map(ontology: Ontology) -> dict[str, str]:
    """Inverse declared on either side counts for both."""
    inv: dict[str, str] = {}
    for rel, spec in ontology.relations.items():
        if spec.inverse and spec.inverse in ontology.relations:
            inv.setdefault(rel, spec.inverse)
            inv.setdefault(spec.inverse, rel)
    return inv
