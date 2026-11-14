"""Forward-chaining reasoner: closure, symmetry, inverse, property chain, provenance."""
import pytest

from meta_evolver.knowledge import (
    KnowledgeGraph,
    agent_ontology,
    explain,
    reason,
    retract_derived,
    validate,
)


def concept(g, n):
    return g.upsert_node("Concept", n)


def chain_graph():
    g = KnowledgeGraph()
    a, b, c, d = (concept(g, n) for n in ("poodle", "dog", "animal", "thing"))
    ids = [g.add_edge(a, "is_a", b, evidence=True), g.add_edge(b, "is_a", c, evidence=True),
           g.add_edge(c, "is_a", d, evidence=True)]
    return g, (a, b, c, d), ids


def derived(g, **kw):
    return [e for e in g.edges(**kw) if e.props.get("derived")]


def test_transitive_closure_of_is_a():
    g, (a, b, c, d), _ = chain_graph()
    reason(g, agent_ontology())
    pairs = {(e.src, e.dst) for e in g.edges(rel="is_a")}
    assert {(a, c), (a, d), (b, d)} <= pairs and len(pairs) == 6


def test_confidence_is_product_of_premises_and_attenuates():
    g, (a, b, c, d), _ = chain_graph()
    reason(g, agent_ontology())
    base = (1 + 1) / (1 + 2)  # asserted edges: 1 win / 1 use
    (ac,) = g.edges(src=a, rel="is_a", dst=c)
    (ad,) = g.edges(src=a, rel="is_a", dst=d)
    assert ac.confidence == pytest.approx(base**2)
    assert ad.confidence < ac.confidence  # longer chain, weaker belief


def test_symmetric_relation_gets_its_reverse():
    g = KnowledgeGraph()
    a, b = concept(g, "heat"), concept(g, "warm")
    eid = g.add_edge(a, "similar_to", b, evidence=True)
    reason(g, agent_ontology())
    (rev,) = g.edges(src=b, rel="similar_to", dst=a)
    assert rev.props["rule"] == "symmetric" and rev.props["premises"] == [eid]


def test_inverse_relation_is_derived():
    g = KnowledgeGraph()
    f, m = g.upsert_node("FailureMode", "locked"), g.upsert_node("Memory", "m-key")
    g.add_edge(f, "mitigated_by", m, evidence=True)
    reason(g, agent_ontology())
    (inv,) = g.edges(src=m, rel="mitigates", dst=f)
    assert inv.props["rule"] == "inverse"


def _failure_graph(wins=3, losses=0):
    g = KnowledgeGraph()
    f, m, c = g.upsert_node("FailureMode", "locked"), g.upsert_node("Memory", "m-key"), concept(g, "safe")
    e1 = g.add_edge(f, "mitigated_by", m, evidence=True)
    e2 = None
    for _ in range(wins):
        e2 = g.add_edge(m, "applies_to", c, evidence=True)
    for _ in range(losses):
        e2 = g.add_edge(m, "applies_to", c, evidence=False)
    return g, (f, m, c), (e1, e2)


def test_property_chain_failure_relates_to_concept():
    g, (f, m, c), (e1, e2) = _failure_graph()
    reason(g, agent_ontology())
    (e,) = g.edges(src=f, rel="related_to", dst=c)
    assert e.props["rule"] and e.props["premises"] == [e1, e2]
    assert e.confidence == pytest.approx(g.get_edge(e1).confidence * g.get_edge(e2).confidence)


def test_chain_ignores_premises_that_mostly_lost():
    g, (f, _, c), _ = _failure_graph(wins=1, losses=4)
    reason(g, agent_ontology())
    assert g.edges(src=f, rel="related_to", dst=c) == []


def test_reason_is_idempotent_and_terminates():
    g, _, _ = chain_graph()
    first = reason(g, agent_ontology())
    n = g.stats()["edges_total"]
    second = reason(g, agent_ontology())
    assert first.added and second.added == [] and second.changed == 0
    assert g.stats()["edges_total"] == n and second.rounds == 1


def test_cyclic_asserted_data_still_terminates_without_self_loops():
    g = KnowledgeGraph()
    a, b = concept(g, "a"), concept(g, "b")
    g.add_edge(a, "is_a", b, evidence=True)
    g.add_edge(b, "is_a", a, evidence=True)  # invalid input, repair's job
    reason(g, agent_ontology())
    assert all(e.src != e.dst for e in g.edges())
    assert g.stats()["edges_total"] == 2


def test_derived_never_overwrites_asserted():
    g = KnowledgeGraph()
    a, b = concept(g, "heat"), concept(g, "warm")
    g.add_edge(a, "similar_to", b, evidence=True)
    own = g.add_edge(b, "similar_to", a, evidence=False)  # asserted reverse, weak
    reason(g, agent_ontology())
    (e,) = g.edges(src=b, rel="similar_to", dst=a)
    assert e.id == own and "derived" not in e.props and (e.wins, e.uses) == (0, 1)


def test_explain_unrolls_the_premises_chain():
    g, (a, b, c, d), ids = chain_graph()
    reason(g, agent_ontology())
    (ad,) = g.edges(src=a, rel="is_a", dst=d)
    ex = explain(g, ad.id)
    assert not ex.asserted and ex.rule == "transitive" and len(ex.premises) == 2

    def leaves(x):
        return [x] if x.asserted else [leaf for p in x.premises for leaf in leaves(p)]

    assert {leaf.edge_id for leaf in leaves(ex)} == set(ids)  # bottoms out in asserted facts
    assert "derived by transitive" in ex.render() and "asserted" in ex.render()
    assert explain(g, ids[0]).asserted and explain(g, 9999) is None


def test_derived_edges_are_removable_and_recomputable():
    g, _, ids = chain_graph()
    reason(g, agent_ontology())
    assert derived(g)
    removed = retract_derived(g)
    assert removed == 3 and derived(g, include_inactive=True) == []
    assert {e.id for e in g.edges()} == set(ids)  # asserted untouched
    again = reason(g, agent_ontology())
    assert len(again.added) == 3


def test_retracting_a_premise_retires_its_consequences():
    g, (a, b, c, d), ids = chain_graph()
    reason(g, agent_ontology())
    g.invalidate(ids[1], 4)  # dog is_a animal no longer holds
    report = reason(g, agent_ontology(), generation=4)
    assert report.retired
    pairs = {(e.src, e.dst) for e in g.edges(rel="is_a")}
    assert (a, c) not in pairs and (a, d) not in pairs and (a, b) in pairs
    assert any(e.props.get("derived") for e in g.edges(rel="is_a", include_inactive=True))  # history kept


def test_validate_is_clean_after_reasoning():
    g, onto = KnowledgeGraph(), agent_ontology()
    chain = [concept(g, n) for n in ("a", "b", "c")]
    g.add_edge(chain[0], "is_a", chain[1], evidence=True)
    g.add_edge(chain[1], "is_a", chain[2], evidence=True)
    g.add_edge(chain[0], "similar_to", concept(g, "z"), evidence=True)
    f, m = g.upsert_node("FailureMode", "locked"), g.upsert_node("Memory", "m-key")
    g.add_edge(f, "mitigated_by", m, evidence=True)
    g.add_edge(m, "applies_to", chain[0], evidence=True)
    assert validate(g, onto) == []
    report = reason(g, onto)
    assert report.changed and validate(g, onto) == []
    assert {"transitive", "symmetric", "inverse"} <= set(report.by_rule)
