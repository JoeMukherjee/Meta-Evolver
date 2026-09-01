"""The causal graph recorder.

Skips when no Neo4j is listening rather than failing:

    docker compose -f docker/docker-compose.yml --profile graph up -d neo4j
"""
from __future__ import annotations

import os

import pytest

from meta_evolver.core.types import GenerationReport, MemoryItem, Trajectory
from meta_evolver.graph_view import SAVED_QUERIES, CausalGraphRecorder, episode_uid

NEO4J_URL = os.environ.get(
    "META_EVOLVER_TEST_NEO4J", "bolt://neo4j:evolution@localhost:7688"
)

_REACHABLE: dict[str, str | None] = {}


def _unreachable() -> str | None:
    """Probed once per session: a failed connect costs a full timeout."""
    if "neo4j" not in _REACHABLE:
        probe = CausalGraphRecorder(NEO4J_URL, run_id="pytest-probe")
        _REACHABLE["neo4j"] = None if probe.enabled else f"no neo4j: {probe.describe}"
        probe.close()
    return _REACHABLE["neo4j"]


@pytest.fixture
def recorder(request):
    reason = _unreachable()
    if reason:
        pytest.skip(reason)
    rec = CausalGraphRecorder(
        NEO4J_URL, run_id=f"pytest-{request.node.name[:40]}", benchmark="devops"
    )
    rec.query("MATCH (n) WHERE n.run_id = $run_id DETACH DELETE n")
    yield rec
    rec.query("MATCH (n) WHERE n.run_id = $run_id DETACH DELETE n")
    rec.close()


def trajectory(task: str, generation: int = 0, success: bool = True, **kwargs) -> Trajectory:
    return Trajectory(
        task_id=task,
        benchmark="devops",
        generation=generation,
        success=success,
        score=1.0 if success else 0.0,
        instruction=f"do {task}",
        **kwargs,
    )


def report(generation: int = 0, **kwargs) -> GenerationReport:
    base = {
        "benchmark": "devops",
        "n_tasks": 2,
        "pass_rate": 0.5,
        "prompt_version": "base",
        "curriculum_level": 0.2,
    }
    return GenerationReport(generation=generation, **{**base, **kwargs})


# --- disabled-by-default behaviour (no server needed) ------------------------


def test_recorder_is_disabled_without_a_url(monkeypatch):
    """Observability is opt-in and must never be load-bearing."""
    monkeypatch.delenv("META_EVOLVER_GRAPH_URL", raising=False)
    rec = CausalGraphRecorder(url=None)
    assert not rec.enabled
    assert rec.describe == "disabled"

    # Every method is a safe no-op, so a caller never needs to check.
    rec.record_generation(report(), [trajectory("t1")], prompt_text="p")
    rec.record_memories([MemoryItem(id="m1", lesson="x")], generation=0)
    rec.record_pruned([MemoryItem(id="m1", lesson="x")], generation=0)
    assert rec.query("MATCH (n) RETURN n") == []
    assert rec.n_writes == 0


def test_unreachable_server_degrades_rather_than_raising():
    """A graph that is down costs the picture, not the run."""
    rec = CausalGraphRecorder("bolt://neo4j:nope@127.0.0.1:1", run_id="x")
    assert not rec.enabled
    assert rec.errors, "the failure is recorded, not swallowed silently"
    rec.record_generation(report(), [trajectory("t1")], prompt_text="p")


def test_episode_uid_is_stable_and_distinguishes_rollouts():
    """Two attempts at one task must not collapse onto one node."""
    a = episode_uid("run", trajectory("t1"))
    b = episode_uid("run", trajectory("t1", rollout_index=1))
    assert a == episode_uid("run", trajectory("t1"))
    assert a != b


def test_credentials_are_not_in_the_description():
    rec = CausalGraphRecorder("bolt://neo4j:hunter2@127.0.0.1:1", run_id="x")
    assert "hunter2" not in rec.describe


# --- against a live server ---------------------------------------------------


def test_generation_writes_the_expected_shape(recorder):
    trajectories = [
        trajectory("db_pool", success=True, retrieved_memory_ids=["m1"]),
        trajectory("jwt_auth", success=False, retrieved_memory_ids=["m1"]),
    ]
    recorder.record_generation(report(), trajectories, prompt_text="be rigorous")
    assert not recorder.errors

    labels = {
        row["label"]: row["n"]
        for row in recorder.query(
            "MATCH (n) WHERE n.run_id = $run_id "
            "RETURN labels(n)[0] AS label, count(*) AS n"
        )
    }
    assert labels["Generation"] == 1
    assert labels["Episode"] == 2
    assert labels["Memory"] == 1  # both episodes retrieved the same one

    edges = {
        row["rel"]: row["n"]
        for row in recorder.query(
            "MATCH (a)-[r]->() WHERE a.run_id = $run_id "
            "RETURN type(r) AS rel, count(*) AS n"
        )
    }
    assert edges["RAN"] == 2
    assert edges["RETRIEVED"] == 2
    assert edges["USED_PROMPT"] == 1


def test_recording_twice_updates_rather_than_duplicating(recorder):
    """A resumed run replays a generation; it must not double the graph."""
    trajectories = [trajectory("db_pool", retrieved_memory_ids=["m1"])]
    recorder.record_generation(report(), trajectories, prompt_text="p")
    recorder.record_generation(report(pass_rate=1.0), trajectories, prompt_text="p")

    rows = recorder.query(
        "MATCH (g:Generation {run_id: $run_id}) RETURN count(g) AS n, collect(g.pass_rate) AS rates"
    )
    assert rows[0]["n"] == 1
    assert rows[0]["rates"] == [1.0], "the later write wins"


def test_memory_provenance_links_both_directions(recorder):
    """A memory records the episode that made it and the ones that used it."""
    recorder.record_generation(
        report(),
        [trajectory("db_pool", retrieved_memory_ids=["mem-x"])],
        prompt_text="p",
    )
    recorder.record_memories(
        [
            MemoryItem(
                id="mem-x",
                title="Restart after patching",
                scenario="stale_config",
                lesson="config needs a restart",
                source_task_ids=["db_pool"],
            )
        ],
        generation=0,
    )

    rows = recorder.query(
        "MATCH (m:Memory {run_id: $run_id, id: 'mem-x'}) "
        "OPTIONAL MATCH (born:Episode)-[:INDUCED]->(m) "
        "OPTIONAL MATCH (used:Episode)-[:RETRIEVED]->(m) "
        "RETURN m.title AS title, count(DISTINCT born) AS born, count(DISTINCT used) AS used"
    )
    assert rows[0]["title"] == "Restart after patching"
    assert rows[0]["born"] == 1
    assert rows[0]["used"] == 1


def test_pruning_is_recorded_where_it_happened(recorder):
    recorder.record_generation(report(generation=1), [trajectory("t", generation=1)], prompt_text="p")
    recorder.record_pruned([MemoryItem(id="mem-bad", lesson="misleads")], generation=1)

    rows = recorder.query(
        "MATCH (:Generation {run_id: $run_id, index: 1})-[:PRUNED]->(m:Memory) "
        "RETURN m.id AS id, m.pruned_at AS at"
    )
    assert rows == [{"id": "mem-bad", "at": 1}]


def test_rejected_prompt_candidates_are_kept(recorder):
    """The rejections are the evidence that alternatives were measured."""
    from meta_evolver.prompts.optimizer import PromptCandidate

    recorder.record_generation(report(generation=2), [trajectory("t", generation=2)], prompt_text="p")
    recorder.record_prompt_candidates(
        [
            PromptCandidate(text="better", version="g2c0", parent="base", score=0.9),
            PromptCandidate(text="worse", version="g2c1", parent="base", score=0.1),
        ],
        generation=2,
        adopted_version="g2c0",
    )

    rows = {
        row["version"]: row
        for row in recorder.query(SAVED_QUERIES["prompt_lineage"][1])
    }
    assert rows["g2c0"]["adopted"] is True
    assert rows["g2c1"]["adopted"] is False, "a rejected candidate is still recorded"
    assert rows["g2c1"]["parent"] == "base"


def test_saved_queries_all_execute(recorder):
    """Every shipped query must at least be valid Cypher against the schema."""
    recorder.record_generation(
        report(), [trajectory("db_pool", retrieved_memory_ids=["m1"])], prompt_text="p"
    )
    for name, (_description, statement) in SAVED_QUERIES.items():
        recorder.query(statement, generation=0)
        assert not recorder.errors, f"{name} failed: {recorder.errors}"
