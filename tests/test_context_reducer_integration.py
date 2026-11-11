"""EvidenceReducer wired into the real episode graph's act() node.

tests/test_reducer.py covers EvidenceReducer in isolation; this covers that
build_episode_graph() actually routes a large observation through it before
observation_chars truncation, and that None (the default) is untouched --
today's flat-truncation behaviour, exactly as the docstring promises.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from meta_evolver.benchmarks.custom import FunctionBenchmark, Task
from meta_evolver.graphs.episode import build_episode_graph, run_episode
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.tools.reducer import EvidenceReducer

_MARKER = "error: pool exhausted at 42 connections"


def _benchmark() -> FunctionBenchmark:
    def look(**_: object) -> dict:
        """Return a large observation to be compressed."""
        return {"log": "log line\n" * 500}

    def answer(text: str) -> dict:
        """Submit the final answer."""
        return {"answer": text}

    return FunctionBenchmark(
        name="reducer-mini",
        tools={"look": look, "answer": answer},
        tasks=[
            Task(
                id="t1",
                instruction="look then answer",
                split="train",
                terminal_tools=("answer",),
            )
        ],
        max_steps=3,
    )


def _stepwise_solver():
    calls = {"n": 0}

    def responder(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return tool_call_message("look")
        return tool_call_message("answer", text="done")

    return responder


def _solve(model, context_reducer=None, observation_chars=4000):
    graph = build_episode_graph(
        model=model, context_reducer=context_reducer, observation_chars=observation_chars
    )
    benchmark = _benchmark()
    env = benchmark.make_env("t1")
    return run_episode(
        graph,
        env=env,
        task_id="t1",
        benchmark=benchmark.name,
        instruction=benchmark.instruction_for("t1"),
        prompt_template=benchmark.system_prompt(),
        max_steps=3,
    )


def test_none_context_reducer_is_flat_truncation_unchanged():
    model = ScriptedChatModel(responder=_stepwise_solver())
    trajectory = _solve(model, context_reducer=None, observation_chars=50)
    look_step = trajectory.steps[0]
    assert len(look_step.observation) <= 50
    assert _MARKER not in look_step.observation


def test_context_reducer_compresses_the_full_observation_before_truncation():
    # min_chars=500: the fixture's own template text (~90 chars for the
    # "answer" step) must stay under threshold, so only the large "look"
    # observation triggers a real compression call.
    policy_model = ScriptedChatModel(responder=_stepwise_solver())
    reducer_model = ScriptedChatModel(responder=lambda m, t: AIMessage(content=_MARKER))
    reducer = EvidenceReducer(reducer_model, min_chars=500, max_output_chars=200)

    trajectory = _solve(policy_model, context_reducer=reducer, observation_chars=4000)

    look_step = trajectory.steps[0]
    assert _MARKER in look_step.observation
    assert reducer.n_calls == 1


def test_context_reducer_output_still_respects_observation_chars():
    policy_model = ScriptedChatModel(responder=_stepwise_solver())
    reducer_model = ScriptedChatModel(responder=lambda m, t: AIMessage(content=_MARKER))
    reducer = EvidenceReducer(reducer_model, min_chars=500, max_output_chars=200)

    trajectory = _solve(policy_model, context_reducer=reducer, observation_chars=10)

    look_step = trajectory.steps[0]
    assert look_step.observation == _MARKER[:10]
