"""Knowledge graph: storage, time-awareness, ontology repair, OOD recall."""
import pytest

from meta_evolver.core.types import Action, StepRecord, Trajectory
from meta_evolver.knowledge import (
    GraphIngestor,
    GraphMemory,
    KnowledgeGraph,
    agent_ontology,
    extract_concepts,
    health,
    node_id,
    repair,
    validate,
)
from meta_evolver.knowledge.ontology import (
    CYCLE,
    DANGLING,
    DOMAIN,
    DUPLICATE,
    FUNCTIONAL,
    UNKNOWN_LABEL,
    UNKNOWN_REL,
)


def traj(task, instruction, success, steps=("go", "take"), mem=(), bench="textgame", gen=0, obs="ok"):
    return Trajectory(
        task_id=task,
        benchmark=bench,
        instruction=instruction,
        steps=[
            StepRecord(step_idx=i, action=Action(name="do", kwargs={"text": f"{v} thing 1"}), observation=obs)
            for i, v in enumerate(steps)
        ],
        success=success,
        score=1.0 if success else 0.0,
        retrieved_memory_ids=list(mem),
        generation=gen,
    )


# -- storage -------------------------------------------------------------------


def test_upsert_is_idempotent_and_merges_props():
    g = KnowledgeGraph()
    a = g.upsert_node("Concept", "Heat Source", {"x": 1})
    b = g.upsert_node("Concept", "heat source", {"y": 2})
    assert a == b == node_id("Concept", "heat source")
    assert g.get_node(a).props == {"x": 1, "y": 2}
    assert g.stats()["nodes"] == 1


def test_reasserting_an_edge_strengthens_not_duplicates():
    g = KnowledgeGraph()
    a, b = g.upsert_node("Memory", "m1"), g.upsert_node("Concept", "heat")
    for ok in (True, True, False):
        g.add_edge(a, "applies_to", b, evidence=ok)
    (e,) = g.edges(src=a)
    assert (e.wins, e.uses) == (2, 3)
    assert e.confidence == pytest.approx(3 / 5)  # Beta(1,1) posterior mean


def test_functional_edge_invalidates_and_history_is_queryable():
    g = KnowledgeGraph()
    t = g.upsert_node("Task", "t1")
    d1, d2 = g.upsert_node("Domain", "a"), g.upsert_node("Domain", "b")
    g.add_edge(t, "in_domain", d1, generation=1, functional=True)
    g.add_edge(t, "in_domain", d2, generation=5, functional=True)
    assert [e.dst for e in g.edges(src=t)] == [d2]  # now
    assert [e.dst for e in g.edges(src=t, at=3)] == [d1]  # as of generation 3
    assert len(g.edges(src=t, include_inactive=True)) == 2  # nothing was deleted


def test_traverse_scores_by_hops_and_edge_confidence():
    g = KnowledgeGraph()
    a, b, c = (g.upsert_node("Concept", n) for n in "abc")
    g.add_edge(a, "is_a", b, evidence=True)
    g.add_edge(b, "is_a", c, evidence=True)
    reach = g.traverse([a], hops=2, rels=["is_a"])
    assert reach[b].distance == 1 and reach[c].distance == 2
    assert reach[b].score > reach[c].score > 0
    assert c not in g.traverse([a], hops=1, rels=["is_a"])


def test_ntriples_export_has_types_and_relations():
    g = KnowledgeGraph()
    g.add_edge(g.upsert_node("Concept", "x"), "is_a", g.upsert_node("Concept", "y"))
    out = g.to_ntriples()
    assert "class/Concept" in out and "rel/is_a" in out


def test_persists_to_disk(tmp_path):
    p = tmp_path / "kg.sqlite"
    g = KnowledgeGraph(p)
    g.upsert_node("Concept", "kept")
    g.close()
    assert KnowledgeGraph(p).get_node(node_id("Concept", "kept")) is not None


# -- ontology validation + repair ------------------------------------------------


def _codes(g, onto):
    return {v.code for v in validate(g, onto)}


def test_clean_graph_has_no_violations():
    g, onto = KnowledgeGraph(), agent_ontology()
    GraphIngestor(g, onto).ingest([traj("t1", "heat the potato", True, mem=["m1"])])
    assert validate(g, onto) == []


def test_unknown_label_and_relation_are_normalised_via_aliases():
    g, onto = KnowledgeGraph(), agent_ontology()
    a = g.upsert_node("strategy", "m1")  # alias of Memory
    c = g.upsert_node("Concept", "heat")
    g.add_edge(a, "helps_with", c)  # alias of applies_to
    assert {UNKNOWN_LABEL, UNKNOWN_REL} <= _codes(g, onto)
    report = repair(g, onto)
    assert report.unresolved == []
    assert g.get_node(node_id("Memory", "m1")) is not None
    assert [e.rel for e in g.edges()] == ["applies_to"]


def test_unknown_relation_without_alias_is_quarantined_not_deleted():
    g, onto = KnowledgeGraph(), agent_ontology()
    g.add_edge(g.upsert_node("Concept", "a"), "vibes_with", g.upsert_node("Concept", "b"))
    report = repair(g, onto, generation=3)
    assert report.unresolved == []
    assert g.edges() == []  # no longer active ...
    assert len(g.edges(include_inactive=True)) == 1  # ... but still auditable


def test_duplicate_nodes_merge_and_pool_evidence():
    g, onto = KnowledgeGraph(), agent_ontology()
    m = g.upsert_node("Memory", "m1")
    c1, c2 = g.upsert_node("Concept", "heat-source"), g.upsert_node("Concept", "heatsource")
    g.add_edge(m, "applies_to", c1, evidence=True)
    g.add_edge(m, "applies_to", c2, evidence=True)
    assert DUPLICATE in _codes(g, onto)
    repair(g, onto)
    (e,) = g.edges(src=m)
    assert e.uses == 2 and e.wins == 2  # the two edges' evidence was pooled
    assert validate(g, onto) == []


def test_wrong_direction_edge_is_reversed():
    g, onto = KnowledgeGraph(), agent_ontology()
    g.add_edge(g.upsert_node("Task", "t1"), "on_task", g.upsert_node("Episode", "e1"))  # backwards
    assert DOMAIN in _codes(g, onto)
    repair(g, onto)
    (e,) = g.edges()
    assert (g.get_node(e.src).label, g.get_node(e.dst).label) == ("Episode", "Task")


def test_generic_entity_is_relabelled_to_the_only_legal_label():
    g, onto = KnowledgeGraph(), agent_ontology()
    ep = g.upsert_node("Episode", "e1")
    g.add_edge(ep, "on_task", g.upsert_node("Entity", "t9"))  # range: only Task
    repair(g, onto)
    assert g.get_node(node_id("Task", "t9")) is not None
    assert validate(g, onto) == []


def test_functional_violation_keeps_the_best_evidenced_target():
    g, onto = KnowledgeGraph(), agent_ontology()
    ep = g.upsert_node("Episode", "e1")
    good, bad = g.upsert_node("Task", "good"), g.upsert_node("Task", "bad")
    g.add_edge(ep, "on_task", bad, evidence=False)
    g.add_edge(ep, "on_task", good, evidence=True)
    assert FUNCTIONAL in _codes(g, onto)
    repair(g, onto)
    assert [e.dst for e in g.edges(src=ep)] == [good]


def test_cycle_in_is_a_is_broken_at_the_weakest_edge():
    g, onto = KnowledgeGraph(), agent_ontology()
    a, b, c = (g.upsert_node("Concept", n) for n in ("a", "b", "c"))
    g.add_edge(a, "is_a", b, evidence=True)
    g.add_edge(b, "is_a", c, evidence=True)
    g.add_edge(c, "is_a", a, evidence=False)  # the weak link
    assert CYCLE in _codes(g, onto)
    repair(g, onto)
    assert validate(g, onto) == []
    assert {(e.src, e.dst) for e in g.edges()} == {(a, b), (b, c)}


def test_dangling_edge_is_deleted_and_fallback_sees_only_residue():
    g, onto = KnowledgeGraph(), agent_ontology()
    g.upsert_node("Concept", "a")
    g.add_edge(node_id("Concept", "a"), "is_a", "Concept:ghost")
    assert DANGLING in _codes(g, onto)
    seen = []
    report = repair(g, onto, fallback=lambda graph, residue: seen.append(residue) or [])
    assert report.unresolved == [] and seen == []  # nothing left to guess about


def test_health_reports_a_drop_in_violations_after_repair():
    g, onto = KnowledgeGraph(), agent_ontology()
    g.add_edge(g.upsert_node("Concept", "a"), "vibes_with", g.upsert_node("Concept", "b"))
    before = health(g, onto)["violations"]
    repair(g, onto)
    assert before > 0 and health(g, onto)["violations"] == 0


# -- ingest + out-of-distribution recall -----------------------------------------


def test_extract_concepts_drops_stopwords_and_keeps_order():
    assert extract_concepts("Please heat the potato and put it in the fridge") == ["heat", "potato", "fridge"]


def test_ingest_skips_infrastructure_errors():
    g = KnowledgeGraph()
    t = traj("t1", "heat potato", True)
    t.error = "rate limited"
    assert GraphIngestor(g).ingest([t])["episodes"] == 0


def test_ood_task_with_new_wording_recalls_strategy_via_shared_concept():
    """The headline behaviour. No word of the new instruction overlaps the old
    task except the concept 'heat' -- text similarity has nothing to go on."""
    g = KnowledgeGraph()
    GraphIngestor(g).ingest(
        [
            traj("old1", "heat the potato then serve", True, mem=["m-heat"]),
            traj("old2", "heat soup in the microwave", True, mem=["m-heat"]),
            traj("old3", "clean the mug", True, mem=["m-clean"]),
        ]
    )
    recs = GraphMemory(g).recommend("warm up a sandwich: heat it somewhere hot", k=3)
    assert recs and recs[0].ref == "m-heat"
    assert "m-clean" not in [r.ref for r in recs]
    assert "heat" in recs[0].why


def test_a_strategy_that_mostly_lost_on_a_concept_is_not_recommended():
    g = KnowledgeGraph()
    GraphIngestor(g).ingest([traj(f"t{i}", "heat the potato", False, mem=["m-bad"]) for i in range(4)])
    assert GraphMemory(g).recommend("heat some soup") == []


def test_rare_concepts_outweigh_common_ones():
    g = KnowledgeGraph()
    ing = GraphIngestor(g)
    ing.ingest([traj(f"c{i}", "object thing item", True, mem=["m-common"]) for i in range(6)])
    ing.ingest([traj("r1", "defrost salmon", True, mem=["m-rare"])])
    recs = GraphMemory(g).recommend("defrost salmon object", k=2)
    assert recs[0].ref == "m-rare"


def test_failure_recall_surfaces_what_fixed_this_task_before():
    g = KnowledgeGraph()
    ing = GraphIngestor(g)
    ing.ingest([traj("t1", "open the safe", False, gen=0, obs="the safe is locked")])
    ing.ingest([traj("t1", "open the safe", True, gen=1, mem=["m-key"])])
    recs = GraphMemory(g).recommend("open the safe", task_id="t1", k=2)
    assert recs[0].ref == "m-key" and "earlier failure" in recs[0].why


def test_exclude_removes_already_retrieved_ids():
    g = KnowledgeGraph()
    GraphIngestor(g).ingest([traj("t1", "heat the potato", True, mem=["m1", "m2"])])
    refs = [r.ref for r in GraphMemory(g).recommend("heat soup", exclude=["m1"])]
    assert refs == ["m2"]
