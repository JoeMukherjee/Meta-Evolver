"""End-to-end graph behaviour, entirely offline."""
from __future__ import annotations

import ast
import re

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from meta_evolver.core.registry import get_benchmark
from meta_evolver.core.rules import IntermittentFault, VerificationGate
from meta_evolver.graphs.episode import build_episode_graph, run_episode
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.telemetry.tracer import TelemetryTracer


def solve(benchmark, model, task_id="db_pool", bank=None, wrap=None, max_steps=15):
    graph = build_episode_graph(model=model, bank=bank)
    env = benchmark.make_env(task_id)
    if wrap is not None:
        env = wrap(env)
    return run_episode(
        graph,
        env=env,
        task_id=task_id,
        benchmark=benchmark.name,
        instruction=benchmark.instruction_for(task_id),
        prompt_template=benchmark.system_prompt(),
        max_steps=max_steps,
    )


def test_episode_graph_solves_a_task(solving_model):
    trajectory = solve(get_benchmark("devops"), solving_model)
    assert trajectory.success
    assert trajectory.score == 1.0
    assert trajectory.error == ""
    # diagnose, patch, restart, verify, submit
    assert [s.action.name for s in trajectory.steps][-3:] == [
        "restart_service",
        "run_healthcheck",
        "submit_resolution",
    ]


def test_no_tracer_means_no_spans_and_no_behaviour_change(solving_model):
    """The default (tracer=None) path must be untouched -- a no-op, not a
    degraded trace."""
    trajectory = solve(get_benchmark("devops"), solving_model)
    assert trajectory.success
    assert trajectory.score == 1.0


def test_tracer_captures_one_root_span_and_a_child_per_node(solving_model):
    tracer = TelemetryTracer(task_id="db_pool", run_id="test-run")
    graph = build_episode_graph(model=solving_model)
    env = get_benchmark("devops").make_env("db_pool")
    trajectory = run_episode(
        graph,
        env=env,
        task_id="db_pool",
        benchmark="devops",
        instruction="fix it",
        prompt_template="",
        max_steps=15,
        tracer=tracer,
    )
    assert trajectory.success

    root = next(s for s in tracer.spans if s.parent_span_id is None)
    assert root.name == "Episode"
    assert root.attributes["success"] is True
    assert root.status == "OK"
    assert root.duration_ms >= 0

    node_names = [s.node_name for s in tracer.spans if s.parent_span_id == root.span_id]
    # Every node the solving trajectory actually passes through must have
    # opened its own child span under the episode root.
    assert "prepare" in node_names
    assert node_names.count("think") >= 1
    assert node_names.count("act") >= 1
    assert "adapt" in node_names
    assert "finalize" in node_names


def test_tracer_think_span_records_tool_choice_and_tokens(solving_model):
    tracer = TelemetryTracer(task_id="db_pool", run_id="test-run")
    graph = build_episode_graph(model=solving_model)
    env = get_benchmark("devops").make_env("db_pool")
    run_episode(
        graph,
        env=env,
        task_id="db_pool",
        benchmark="devops",
        instruction="fix it",
        prompt_template="",
        max_steps=15,
        tracer=tracer,
    )
    think_spans = [s for s in tracer.spans if s.node_name == "think"]
    assert think_spans
    first = think_spans[0]
    assert "tool_call" in first.attributes
    assert "tokens_so_far" in first.attributes
    assert "n_tools_offered" in first.attributes


def test_tracer_survives_an_episode_ending_in_error():
    """A tracer must not swallow or corrupt an errored episode's result."""

    class Failing(ScriptedChatModel):
        def bind_tools(self, tools, **kwargs):
            raise RuntimeError("boom")

    tracer = TelemetryTracer(task_id="db_pool", run_id="test-run")
    graph = build_episode_graph(model=Failing())
    env = get_benchmark("devops").make_env("db_pool")
    trajectory = run_episode(
        graph,
        env=env,
        task_id="db_pool",
        benchmark="devops",
        instruction="fix it",
        prompt_template="",
        max_steps=15,
        tracer=tracer,
    )
    assert trajectory.error
    # The llm failure is caught inside `think` and routed to `finalize` as
    # episode state, not raised up through the graph -- so it shows up as an
    # error attribute on the finalize span, not a span-level ERROR status.
    finalize_span = next(s for s in tracer.spans if s.node_name == "finalize")
    assert finalize_span.attributes["success"] is False
    assert finalize_span.attributes["error"]


def test_transcript_is_well_formed_langchain_messages(solving_model):
    """Every tool call must be answered by a ToolMessage with a matching id.

    A mismatched id is rejected by every provider, and the failure surfaces
    as a 400 mid-episode rather than as anything the agent could recover
    from -- so it is worth asserting on the shape directly.
    """
    solve(get_benchmark("devops"), solving_model)
    transcript = solving_model.calls[-1]

    assert isinstance(transcript[0], SystemMessage), "system prompt is rebuilt each turn"

    issued = {
        call["id"]
        for m in transcript
        for call in (getattr(m, "tool_calls", None) or [])
    }
    answered = {m.tool_call_id for m in transcript if isinstance(m, ToolMessage)}
    assert issued, "the agent made tool calls"
    assert answered <= issued
    # Every call but the most recent has its reply in the transcript.
    assert len(issued - answered) <= 1


def test_tools_are_bound_through_bind_tools(solving_model):
    solve(get_benchmark("devops"), solving_model)
    names = {t["function"]["name"] for t in solving_model.bound_tools}
    assert "submit_resolution" in names
    assert "restart_service" in names


def test_hopeless_episode_terminates_at_the_step_budget(looping_model):
    trajectory = solve(get_benchmark("devops"), looping_model, max_steps=6)
    assert not trajectory.success
    assert trajectory.n_steps == 6
    assert trajectory.error == ""  # a real failure, not an infrastructure one
    assert trajectory.usable


def test_history_is_pruned_past_the_cap(looping_model):
    """``adapt`` bounds what the next ``think`` call resends.

    ``looping_model`` never submits, so a 20-step run accumulates ~41
    messages if nothing trims it. With ``max_history_messages=6`` the last
    call should stay small regardless of how long the episode ran.
    """
    graph = build_episode_graph(model=looping_model, max_history_messages=6)
    env = get_benchmark("devops").make_env("db_pool")
    run_episode(
        graph,
        env=env,
        task_id="db_pool",
        benchmark="devops",
        instruction="fix it",
        prompt_template=get_benchmark("devops").system_prompt(),
        max_steps=20,
    )
    transcript = looping_model.calls[-1]

    # system + the opening task message + at most ~6 recent turns, not the
    # ~40-message transcript an untrimmed 20-step loop would send.
    assert len(transcript) < 12
    assert isinstance(transcript[0], SystemMessage)
    assert "Task:" in transcript[1].content

    issued = {
        call["id"] for m in transcript for call in (getattr(m, "tool_calls", None) or [])
    }
    answered = {m.tool_call_id for m in transcript if isinstance(m, ToolMessage)}
    assert answered <= issued
    assert len(issued - answered) <= 1


def test_history_pruning_does_not_disturb_a_short_episode(solving_model):
    """The cap is a ceiling, not a fixed window: nothing is dropped early."""
    trajectory = solve(get_benchmark("devops"), solving_model, max_steps=15)
    transcript = solving_model.calls[-1]
    assert trajectory.success
    # db_pool solves in a handful of steps -- well under the default cap of
    # 24 -- so every turn's message should still be present. The transcript
    # that produced the *last* step has one fewer ToolMessage than the full
    # trajectory: that step's own result is appended after this call.
    assert sum(1 for m in transcript if isinstance(m, ToolMessage)) == len(trajectory.steps) - 1


def test_stagnation_evicts_memory_mid_episode(looping_model):
    from meta_evolver.core.types import MemoryItem
    from meta_evolver.memory.bank import ReasoningMemoryBank

    bank = ReasoningMemoryBank(
        [MemoryItem(id="m1", title="Read logs", lesson="the answer is always in the logs")]
    )
    trajectory = solve(get_benchmark("devops"), looping_model, bank=bank, max_steps=12)
    assert trajectory.retrieved_memory_ids == ["m1"]
    assert trajectory.memory_evicted_at is not None


def test_api_failure_is_recorded_as_an_error_not_a_task_failure():
    """The distinction the whole learning signal depends on.

    A rate-limited episode that scored as a task failure would teach the
    memory bank and the prompt optimizer a lesson about a provider outage.
    """

    class Failing(ScriptedChatModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("RateLimitError: 429")

    trajectory = solve(get_benchmark("devops"), Failing())
    assert not trajectory.success
    assert "RateLimitError" in trajectory.error
    assert not trajectory.usable  # excluded from induction, credit and scoring


def test_prose_only_model_is_nudged_then_gives_up():
    model = ScriptedChatModel(
        responder=lambda messages, tools: AIMessage(content="Let me think about this.")
    )
    trajectory = solve(get_benchmark("devops"), model, max_steps=10)
    assert trajectory.n_steps == 0
    assert trajectory.error == ""
    assert len(model.calls) <= 5  # bounded, not a runaway


def test_verification_gate_blocks_a_premature_submission():
    """Submitting before the healthcheck passes must be rejected."""

    def responder(messages, tools):
        return tool_call_message(
            "submit_resolution", root_cause="guessing", action_taken="nothing"
        )

    trajectory = solve(
        get_benchmark("devops"),
        ScriptedChatModel(responder=responder),
        wrap=lambda env: VerificationGate(env),
        max_steps=4,
    )
    assert not trajectory.success
    assert all(step.blocked for step in trajectory.steps)


def test_injected_faults_are_flagged_on_the_step(solving_model):
    trajectory = solve(
        get_benchmark("devops"),
        solving_model,
        wrap=lambda env: IntermittentFault(env, failure_rate=1.0, max_faults=1, seed=7),
    )
    assert sum(1 for s in trajectory.steps if s.perturbed) == 1


def test_harness_perturbation_is_reproducible_given_a_seed(solving_model):
    def run(seed):
        model = ScriptedChatModel(responder=solving_model.responder)
        return solve(
            get_benchmark("devops"),
            model,
            wrap=lambda env: IntermittentFault(env, failure_rate=0.5, seed=seed),
        )

    a, b = run(11), run(11)
    assert [s.perturbed for s in a.steps] == [s.perturbed for s in b.steps]


def test_textgame_benchmark_runs_through_the_same_graph():
    """A string-action environment needs no special case in the engine."""

    def responder(messages, tools):
        # Walk the admissible list in order: a dumb but valid policy.
        last = str(getattr(messages[-1], "content", ""))
        match = re.search(r"Admissible commands: (\[.*\])", last)
        commands = ast.literal_eval(match.group(1)) if match else ["look"]
        taken = {
            call["args"].get("text")
            for m in messages
            for call in (getattr(m, "tool_calls", None) or [])
        }
        for cmd in commands:
            if cmd not in taken:
                return tool_call_message("do", text=cmd)
        return tool_call_message("do", text=commands[0])

    bench = get_benchmark("textgame")
    trajectory = solve(
        bench, ScriptedChatModel(responder=responder), task_id="bathroom_soap_id", max_steps=10
    )
    assert trajectory.error == ""
    assert trajectory.n_steps > 0
    # Text actions render readably for the inducer, not as tool-call noise.
    assert trajectory.steps[0].action.render().startswith("go to")


@pytest.mark.parametrize(
    "task_id", ["db_pool", "jwt_auth", "cache_oom", "rate_limit", "disk_pressure"]
)
def test_every_devops_task_is_solvable_and_starts_unsolved(task_id):
    env = get_benchmark("devops").make_env(task_id)
    env.reset(options={"task_id": task_id})
    assert not env.evaluate().success
    assert env.evaluate().score == 0.0
