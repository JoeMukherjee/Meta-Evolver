"""Query-adaptive retrieval_k -- retrieval_min_similarity makes the episode
graph's memory retrieval return fewer than retrieval_k items when a query
has few genuinely relevant memories, instead of padding the prompt with
weak matches. Covers the bank primitive, the config default, and the full
MetaEvolver wiring."""
from __future__ import annotations

from meta_evolver.benchmarks.custom import FunctionBenchmark, Task
from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.core.types import MemoryItem
from meta_evolver.graphs.episode import build_episode_graph, run_episode
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.memory.bank import ReasoningMemoryBank


def _item(**kwargs) -> MemoryItem:
    base = {"scenario": "general", "lesson": "a lesson", "title": "t"}
    return MemoryItem(**{**base, **kwargs})


def test_default_config_never_filters_by_similarity():
    assert EvolutionConfig().retrieval_min_similarity == 0.0


def test_bank_retrieve_drops_weak_candidates_instead_of_padding_to_k():
    bank = ReasoningMemoryBank(
        [
            _item(id="pool", lesson="connection pool exhausted, raise the limit"),
            _item(id="unrelated", lesson="check the fridge for expired milk"),
        ]
    )
    padded = bank.retrieve("the connection pool is exhausted", k=2, mode="cosine")
    assert len(padded) == 2  # today's behaviour: always fills to k

    adaptive = bank.retrieve(
        "the connection pool is exhausted", k=2, mode="cosine", min_similarity=0.35
    )
    assert [m.id for m in adaptive] == ["pool"]  # the weak match is dropped, not padded in


def _bench() -> FunctionBenchmark:
    def answer(text: str) -> dict:
        """Submit the final answer."""
        return {"answer": text}

    return FunctionBenchmark(
        name="adaptive-k-mini",
        tools={"answer": answer},
        tasks=[
            Task(
                id="t1",
                instruction="the connection pool is exhausted, fix it",
                split="train",
                terminal_tools=("answer",),
            )
        ],
        max_steps=2,
    )


def _bank() -> ReasoningMemoryBank:
    return ReasoningMemoryBank(
        [
            _item(id="pool", lesson="connection pool exhausted, raise the limit"),
            _item(id="unrelated", lesson="check the fridge for expired milk"),
        ]
    )


def _solve_and_retrieve(**episode_kwargs) -> list[str]:
    model = ScriptedChatModel(responder=lambda m, t: tool_call_message("answer", text="done"))
    graph = build_episode_graph(model=model, bank=_bank(), retrieval_k=2, **episode_kwargs)
    benchmark = _bench()
    env = benchmark.make_env("t1")
    trajectory = run_episode(
        graph,
        env=env,
        task_id="t1",
        benchmark=benchmark.name,
        instruction=benchmark.instruction_for("t1"),
        prompt_template=benchmark.system_prompt(),
        max_steps=2,
    )
    return trajectory.retrieved_memory_ids


def test_zero_min_similarity_retrieves_the_full_k_unchanged():
    assert sorted(_solve_and_retrieve(retrieval_min_similarity=0.0)) == ["pool", "unrelated"]


def test_high_min_similarity_drops_the_weak_match_from_a_live_episode():
    assert _solve_and_retrieve(retrieval_min_similarity=0.35) == ["pool"]


def test_metaevolver_wires_retrieval_min_similarity_from_config():
    evolver = MetaEvolver(
        benchmark=_bench(),
        chat_model=ScriptedChatModel(responder=lambda m, t: tool_call_message("answer", text="done")),
        bank=_bank(),
        config=EvolutionConfig(
            generations=1, max_steps=2, retrieval_k=2, retrieval_min_similarity=0.35
        ),
        telemetry=False,
    )
    env = evolver.benchmark.make_env("t1")
    trajectory = run_episode(
        evolver.episode_graph,
        env=env,
        task_id="t1",
        benchmark=evolver.benchmark.name,
        instruction=evolver.benchmark.instruction_for("t1"),
        prompt_template=evolver.benchmark.system_prompt(),
        max_steps=2,
    )
    assert trajectory.retrieved_memory_ids == ["pool"]
