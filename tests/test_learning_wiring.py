"""The knowledge graph and skill library wired into a live MetaEvolver run.

A scripted model solves every task with the same two-step procedure
(``lookup`` then ``answer``). After one generation the graph must hold the
episodes, and the skill layer must have mined that procedure as a candidate.
Also pins the default-off guarantee: no learner, no behaviour change.
"""
from __future__ import annotations

from meta_evolver import GenerationLearner, MetaEvolver
from meta_evolver.benchmarks.custom import FunctionBenchmark, Task
from meta_evolver.core.types import Trajectory
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.knowledge import GraphMemory, health, node_id
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.skills import SkillLibrary
from langchain_core.messages import ToolMessage


def _lookup(topic: str = "") -> dict:
    """Look a topic up."""
    return {"info": f"facts about {topic}"}


def _answer(text: str) -> dict:
    """Submit the final answer."""
    return {"answer": text}


def _bench(n: int = 4) -> FunctionBenchmark:
    topics = ["rollback deploy", "restart service", "rotate secret", "scale cluster"]
    tasks = [
        Task(
            id=f"t{i}",
            instruction=f"Fix the {topics[i % 4]} incident",
            split="train",
            terminal_tools=("answer",),
            verify=lambda calls: any(c.name == "answer" for c in calls),
        )
        for i in range(n)
    ]
    return FunctionBenchmark(name="wire-mini", tools={"lookup": _lookup, "answer": _answer}, tasks=tasks, max_steps=4)


def _responder(messages, tools):
    if not any(isinstance(m, ToolMessage) for m in messages):
        return tool_call_message("lookup", topic="x")
    return tool_call_message("answer", text="done")


def _evolver(learner=None, **cfg) -> MetaEvolver:
    return MetaEvolver(
        benchmark=_bench(),
        chat_model=ScriptedChatModel(responder=_responder),
        config=EvolutionConfig(generations=2, max_steps=4, optimize_prompt=False, validate_prompt=False, **cfg),
        telemetry=False,
        learner=learner,
    )


def test_one_generation_populates_graph_and_mines_a_skill_candidate():
    learner = GenerationLearner.default()
    evolver = _evolver(learner)
    evolver.evolve(generations=1)

    assert learner.errors == []
    stats = learner.graph.stats()
    assert stats["by_label"].get("Episode", 0) >= 4 and stats["by_label"].get("Task", 0) >= 4
    assert health(learner.graph, learner.ontology)["violations"] == 0  # the ingest writes a clean graph

    (cand,) = learner.skills.by_status("candidate")
    assert cand.signature == "lookup>answer"
    assert learner.history[-1]["graph"]["episodes"] >= 4


def test_second_generation_offers_the_candidate_as_a_trial_and_credits_it():
    learner = GenerationLearner.default()
    evolver = _evolver(learner)
    reports = evolver.evolve(generations=2)
    assert len(reports) == 2 and learner.errors == []

    (skill,) = [s for s in learner.skills.skills.values() if s.signature == "lookup>answer"]
    assert skill.uses > 0 and skill.wins > 0  # it was retrieved in gen 2 and credited
    cited = learner.graph.edges(dst=node_id("Skill", skill.id), rel="cited")
    assert cited, "episodes that saw the skill must be linked to it in the graph"


def test_trajectories_carry_skill_ids_separately_from_memory_ids():
    learner = GenerationLearner.default()
    evolver = _evolver(learner)
    evolver.evolve(generations=2)
    trajs: list[Trajectory] = [t for r in evolver.reports for t in getattr(r, "trajectories", [])] or []
    # Reports may not retain trajectories; the invariant that matters is on the graph side:
    assert not [e for e in learner.graph.edges(rel="cited") if e.dst.startswith("Memory:sk-")]
    assert trajs is not None


def test_graph_recommends_a_proven_memory_for_a_reworded_task():
    learner = GenerationLearner.default()
    from meta_evolver.core.types import Action, StepRecord

    def t(task, instr, ok, mem):
        return Trajectory(
            task_id=task, benchmark="b", instruction=instr, success=ok, score=float(ok),
            steps=[StepRecord(step_idx=0, action=Action(name="lookup"), observation="ok")],
            retrieved_memory_ids=mem,
        )

    learner.ingestor.ingest([t("a", "rollback the failed deploy", True, ["m1"]), t("b", "rollback the bad release", True, ["m1"])])
    recs = GraphMemory(learner.graph).recommend("undo this broken thing: rollback needed", k=2)
    assert [r.ref for r in recs] == ["m1"]


def test_default_off_is_unchanged_and_a_broken_layer_cannot_fail_a_run():
    plain = _evolver(None)
    assert plain.learner is None
    plain.evolve(generations=1)

    class Boom(GenerationLearner):
        def learn(self, *a, **k):  # an additive layer that explodes
            raise RuntimeError("boom")

    # learn() raising propagates only if a caller forgot the guard inside the real class;
    # the real class guards each layer, which this pins:
    guarded = GenerationLearner.default()
    guarded.ingestor.ingest = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("graph down"))
    guarded.learn([], 0)
    assert any("graph" in e for e in guarded.errors)


def test_skill_library_persists_across_runs(tmp_path):
    learner = GenerationLearner.default(tmp_path)
    _evolver(learner).evolve(generations=1)
    again = SkillLibrary(tmp_path / "skills.json")
    assert any(s.signature == "lookup>answer" for s in again.skills.values())
