"""EvidenceReducer wired into MetaEvolver's live episode graph --
default-off parity, and that turning it on compresses a large tool
observation instead of flat-truncating it."""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from meta_evolver.benchmarks.custom import FunctionBenchmark, Task
from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.graphs.episode import run_episode
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message

# No newlines/quotes: the raw text survives a dict repr unchanged, so a
# substring check on the rendered observation stays exact.
_LOG_DUMP = "x" * 3000
_COMPRESSED = "COMPRESSED nothing wrong found"


def _dump(**_: object) -> dict:
    """A tool whose result is large enough to cross a reducer threshold, so
    the reducer's presence is directly visible in the resulting observation
    text rather than inferred indirectly."""
    return {"value": _LOG_DUMP}


def _answer(text: str) -> dict:
    """Submit the final answer."""
    return {"answer": text}


def _bench() -> FunctionBenchmark:
    tasks = [
        Task(id="t1", instruction="dump then answer", split="train", terminal_tools=("answer",))
    ]
    return FunctionBenchmark(
        name="reducer-mini", tools={"dump": _dump, "answer": _answer}, tasks=tasks, max_steps=3
    )


def _responder(messages, tools):
    last = messages[-1].content if messages else ""
    if isinstance(last, str) and "Raw tool output" in last:
        return AIMessage(content=_COMPRESSED)
    called = any(isinstance(m, ToolMessage) for m in messages)
    if not called:
        return tool_call_message("dump")
    return tool_call_message("answer", text="done")


def _build(**config_overrides) -> MetaEvolver:
    defaults = dict(generations=1, max_steps=3)
    return MetaEvolver(
        benchmark=_bench(),
        chat_model=ScriptedChatModel(responder=_responder),
        config=EvolutionConfig(**{**defaults, **config_overrides}),
        telemetry=False,
    )


def _run(evolver: MetaEvolver):
    env = evolver.benchmark.make_env("t1")
    return run_episode(
        evolver.episode_graph,
        env=env,
        task_id="t1",
        benchmark=evolver.benchmark.name,
        instruction=evolver.benchmark.instruction_for("t1"),
        prompt_template=evolver.benchmark.system_prompt(),
        max_steps=3,
    )


def test_default_config_never_builds_a_reducer():
    assert EvolutionConfig().context_reducer is False


def test_context_reducer_off_by_default_leaves_flat_truncation():
    evolver = _build()
    trajectory = _run(evolver)
    dump_step = next(s for s in trajectory.steps if s.action.name == "dump")
    assert _LOG_DUMP in dump_step.observation
    assert _COMPRESSED not in dump_step.observation


def test_context_reducer_true_compresses_the_large_observation():
    evolver = _build(context_reducer=True, context_reducer_min_chars=50)
    trajectory = _run(evolver)
    dump_step = next(s for s in trajectory.steps if s.action.name == "dump")
    assert dump_step.observation == _COMPRESSED
    assert _LOG_DUMP not in dump_step.observation


def test_context_reducer_below_min_chars_never_calls_the_model():
    """A small observation must pass through unchanged even with the
    reducer turned on -- min_chars is still respected end to end."""
    evolver = _build(context_reducer=True, context_reducer_min_chars=10_000_000)
    trajectory = _run(evolver)
    dump_step = next(s for s in trajectory.steps if s.action.name == "dump")
    assert _LOG_DUMP in dump_step.observation
    assert _COMPRESSED not in dump_step.observation
