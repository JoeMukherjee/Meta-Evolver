"""Concept lifting: induced similar_to / is_a links and opt-in evidence transfer."""
from meta_evolver.knowledge import (
    GraphMemory,
    KnowledgeGraph,
    agent_ontology,
    induce_concept_links,
    node_id,
    retract_lifted,
    validate,
)
from meta_evolver.knowledge.ontology import CYCLE


def win(g, mem, concept, n=2, lose=0):
    m, c = g.upsert_node("Memory", mem), g.upsert_node("Concept", concept)
    for _ in range(n):
        g.add_edge(m, "applies_to", c, evidence=True)
    for _ in range(lose):
        g.add_edge(m, "applies_to", c, evidence=False)


def heat_warm_graph():
    """m1,m2 won on both words; m3 only ever on 'heat'; m9 is unrelated."""
    g = KnowledgeGraph()
    for m in ("m1", "m2"):
        win(g, m, "heat")
        win(g, m, "warm")
    win(g, "m3", "heat", n=3)
    win(g, "m9", "clean")
    return g


def cid(name):
    return node_id("Concept", name)


def test_synonym_transfer_after_lifting():
    g = heat_warm_graph()
    mem = GraphMemory(g)
    # Before lifting a 'warm' task cannot see m3, which never touched 'warm'.
    assert "m3" not in [r.ref for r in mem.recommend("warm sandwich", k=5, transfer=True)]
    induce_concept_links(g)
    recs = mem.recommend("warm sandwich", k=5, transfer=True)
    refs = [r.ref for r in recs]
    assert "m3" in refs and "m9" not in refs
    m3 = next(r for r in recs if r.ref == "m3")
    assert "heat" in m3.why and "linked" in m3.why  # says it was borrowed
    # Borrowed evidence ranks below the same strategy's direct evidence.
    assert refs.index("m3") > refs.index("m1")


def test_transfer_is_off_by_default_and_changes_nothing():
    g = heat_warm_graph()
    mem = GraphMemory(g)
    before = [(r.ref, r.score, r.why) for r in mem.recommend("warm sandwich", k=5)]
    induce_concept_links(g)
    assert [(r.ref, r.score, r.why) for r in mem.recommend("warm sandwich", k=5)] == before
    assert [(r.ref, r.score, r.why) for r in mem.recommend("warm sandwich", k=5, transfer=False)] == before


def test_transfer_without_links_equals_plain_recommend():
    g = heat_warm_graph()  # never lifted
    mem = GraphMemory(g)
    a = [(r.ref, r.score) for r in mem.recommend("warm sandwich", k=5)]
    b = [(r.ref, r.score) for r in mem.recommend("warm sandwich", k=5, transfer=True)]
    assert a == b


def test_induced_edges_are_marked_inferred_with_support():
    g = heat_warm_graph()
    report = induce_concept_links(g, generation=3)
    assert report.added
    (e,) = g.edges(rel="similar_to")
    assert {e.src, e.dst} == {cid("heat"), cid("warm")} and e.src < e.dst  # stored once
    assert e.props["inferred"] is True and e.props["rule"]
    assert e.props["support"]["intersection"] == 2 and e.props["support"]["union"] == 3
    assert e.valid_from == 3
    assert (e.wins, e.uses) == (2, 3)  # Jaccard-flavoured Beta evidence


def test_strict_subset_becomes_is_a_not_similar():
    g = KnowledgeGraph()
    for m in ("m1", "m2"):
        win(g, m, "microwave")
    for m in ("m1", "m2", "m3", "m4", "m5"):
        win(g, m, "heat")
    induce_concept_links(g)
    assert g.edges(rel="similar_to") == []
    (e,) = g.edges(rel="is_a")
    assert (e.src, e.dst) == (cid("microwave"), cid("heat"))
    assert e.props["support"]["narrow"] == 2 and e.props["support"]["broad"] == 5


def test_min_support_and_losing_evidence_are_respected():
    g = KnowledgeGraph()
    win(g, "m1", "heat")
    win(g, "m1", "warm")  # a single shared winner is not enough
    induce_concept_links(g)
    assert g.edges(rel="similar_to") == []
    g2 = KnowledgeGraph()
    for m in ("m1", "m2"):
        win(g2, m, "heat")
        win(g2, m, "warm", n=1, lose=3)  # mostly lost on 'warm': not winning evidence
    induce_concept_links(g2)
    assert g2.edges(rel="similar_to") == []


def test_no_is_a_cycles_even_against_asserted_edges():
    g, onto = KnowledgeGraph(), agent_ontology()
    for m in ("m1", "m2"):
        win(g, m, "microwave")
    for m in ("m1", "m2", "m3", "m4", "m5"):
        win(g, m, "heat")
    mid = g.upsert_node("Concept", "appliance")
    # asserted: heat is_a appliance is_a microwave -- lifting's microwave is_a heat would close a loop
    g.add_edge(cid("heat"), "is_a", mid, evidence=True)
    g.add_edge(mid, "is_a", cid("microwave"), evidence=True)
    induce_concept_links(g)
    assert not [e for e in g.edges(rel="is_a") if e.props.get("inferred")]
    assert CYCLE not in {v.code for v in validate(g, onto)}


def test_asserted_edge_on_the_same_pair_is_never_overwritten():
    g = heat_warm_graph()
    g.add_edge(cid("heat"), "is_a", cid("warm"), evidence=True)
    induce_concept_links(g)
    assert g.edges(rel="similar_to") == []
    (e,) = g.edges(rel="is_a")
    assert "inferred" not in e.props


def test_induction_is_idempotent():
    g = heat_warm_graph()
    induce_concept_links(g)
    n = g.stats()["edges_total"]
    again = induce_concept_links(g)
    assert again.changed == 0 and g.stats()["edges_total"] == n


def test_stale_links_are_retired_and_retractable():
    g = heat_warm_graph()
    induce_concept_links(g)
    # m1 and m2 start losing on 'warm': the overlap evidence disappears.
    for m in ("m1", "m2"):
        for _ in range(6):
            g.add_edge(node_id("Memory", m), "applies_to", cid("warm"), evidence=False)
    report = induce_concept_links(g, generation=5)
    assert report.retired and g.edges(rel="similar_to") == []
    assert len(g.edges(rel="similar_to", include_inactive=True)) == 1  # auditable
    g2 = heat_warm_graph()
    induce_concept_links(g2)
    assert retract_lifted(g2) == 1 and g2.edges(rel="similar_to") == []


def test_validate_is_clean_after_lifting():
    g, onto = heat_warm_graph(), agent_ontology()
    induce_concept_links(g)
    assert validate(g, onto) == []
