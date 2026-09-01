"""Unit and integration tests for the five advanced scaffolding features:
1. In-flight assertions and intra-step backtracking (ScaffoldAssert)
2. Feedback-driven multi-component reflection (GEPAPromptOptimizer)
3. Sandboxed variable REPL & sub-LLM context navigation (ScaffoldRLM)
4. Dynamic code scaffolding and evolving Python rules (FlexScaffold)
5. OpenTelemetry and MLflow trace collection and export (TelemetryTracer)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from meta_evolver.benchmarks.devops import DevOpsBenchmark
from meta_evolver.core.flex import FlexModule, FlexProposer, FlexResult, FlexRule
from meta_evolver.core.types import Action, EnvResponse, Observation, StepRecord, Trajectory
from meta_evolver.graphs.episode import arun_episode, build_episode_graph, run_episode
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.prompts.gepa import (
    GEPAPromptOptimizer,
    ModularPrompt,
    ParetoCandidate,
    ParetoFrontier,
)
from meta_evolver.telemetry.tracer import TelemetryTracer, TraceSpan
from meta_evolver.tools.assertions import (
    AdmissibleCommandAssertion,
    AssertionResult,
    AssertionRunner,
    CustomAssertion,
    NonEmptyArgsAssertion,
    NumericRangeAssertion,
    ValidToolAssertion,
)
from meta_evolver.tools.repl import REPLExecutionResult, REPLSession, REPLTools, VariableDescriptor

# ---------------------------------------------------------------------------
# 1. In-Flight Assertions & Backtracking
# ---------------------------------------------------------------------------


def test_non_empty_args_assertion():
    assert_fn = NonEmptyArgsAssertion(required_keys=["service_name"])
    res_ok = assert_fn.evaluate(Action(name="test", kwargs={"service_name": "auth"}))
    assert res_ok.passed is True

    res_fail = assert_fn.evaluate(Action(name="test", kwargs={"service_name": ""}))
    assert res_fail.passed is False
    assert "must not be empty" in res_fail.message


def test_valid_tool_assertion():
    assert_fn = ValidToolAssertion(allowed_tools=["read_logs", "restart"])
    assert assert_fn.evaluate(Action(name="read_logs")).passed is True
    res = assert_fn.evaluate(Action(name="drop_database"))
    assert res.passed is False
    assert "not in the allowed tools" in res.message


def test_admissible_command_assertion():
    assert_fn = AdmissibleCommandAssertion()
    state = {"admissible": ["go north", "take key"]}
    assert assert_fn.evaluate(Action(name="do", kwargs={"text": "go north"}), state=state).passed is True
    assert assert_fn.evaluate(Action(name="do", kwargs={"text": "GO NORTH"}), state=state).passed is True
    res = assert_fn.evaluate(Action(name="do", kwargs={"text": "fly away"}), state=state)
    assert res.passed is False
    assert "not in admissible" in res.message


def test_numeric_range_assertion():
    assert_fn = NumericRangeAssertion(key="replicas", min_val=1, max_val=10)
    assert assert_fn.evaluate(Action(name="scale", kwargs={"replicas": 5})).passed is True
    assert assert_fn.evaluate(Action(name="scale", kwargs={"replicas": 0})).passed is False
    assert assert_fn.evaluate(Action(name="scale", kwargs={"replicas": 15})).passed is False


def test_custom_assertion():
    assert_fn = CustomAssertion(
        predicate=lambda act, state, env_st: (
            bool(act.kwargs.get("endpoint", "").startswith("api/")),
            "Endpoint must start with api/",
        ),
        name="EndpointPrefix",
    )
    assert assert_fn.evaluate(Action(name="curl", kwargs={"endpoint": "api/v1/users"})).passed is True
    res = assert_fn.evaluate(Action(name="curl", kwargs={"endpoint": "internal/admin"}))
    assert res.passed is False
    assert "Endpoint must start with api/" in res.message


def test_assertion_runner_hard_and_soft():
    runner = AssertionRunner(
        [
            NonEmptyArgsAssertion(required_keys=["service_name"], is_hard=True),
            CustomAssertion(
                predicate=lambda act, state, env_st: False,
                failure_message="Advisory suggestion",
                is_hard=False,
            ),
        ]
    )
    results = runner.evaluate(Action(name="inspect", kwargs={"service_name": ""}))
    assert len(runner.hard_failures(results)) == 1
    assert len(runner.soft_warnings(results)) == 1


def test_episode_assertion_backtrack_retry():
    """Agent emits an empty parameter, assertion fails and triggers intra-step retry; agent corrects it."""
    bench = DevOpsBenchmark()
    env = bench.make_env("db_pool")

    # Responder: first call is invalid (empty service_name), second call corrects it
    def retry_responder(messages, tools):
        for msg in messages:
            if "Action validation assertion failed" in str(getattr(msg, "content", "")):
                return tool_call_message("inspect_service_logs", service_name="payment-gateway")
        # First call has empty service_name
        return tool_call_message("inspect_service_logs", service_name="")

    model = ScriptedChatModel(responder=retry_responder)
    runner = AssertionRunner([NonEmptyArgsAssertion(required_keys=["service_name"], is_hard=True)])

    graph = build_episode_graph(model=model, assertion_runner=runner, max_assertion_retries=3)
    traj = run_episode(
        graph,
        env=env,
        task_id="db_pool",
        benchmark=bench.name,
        instruction=bench.instruction_for("db_pool"),
        prompt_template=bench.system_prompt(),
        max_steps=4,
    )

    assert traj.error == ""
    # Should have executed the corrected step
    assert len(traj.steps) > 0
    assert traj.steps[0].action.kwargs["service_name"] == "payment-gateway"


# ---------------------------------------------------------------------------
# 2. Feedback-Driven Multi-Component Reflection (GEPA)
# ---------------------------------------------------------------------------


def test_modular_prompt_render_and_parse():
    mod = ModularPrompt(
        core_role="Custom Role",
        planning="Step by step planning",
        tool_policy="Careful tool usage",
        error_recovery="Inspect on error",
    )
    rendered = mod.render()
    assert "{memory_section}" in rendered
    assert "{guidance_section}" in rendered
    assert "Custom Role" in rendered
    assert "Step by step planning" in rendered

    parsed = ModularPrompt.from_text(rendered)
    assert parsed.core_role != ""


def test_pareto_frontier_dominance():
    c1 = ParetoCandidate(id="c1", modular_prompt=ModularPrompt(), task_scores={"t1": 1.0, "t2": 0.5})
    c2 = ParetoCandidate(id="c2", modular_prompt=ModularPrompt(), task_scores={"t1": 0.5, "t2": 1.0})
    c3 = ParetoCandidate(id="c3", modular_prompt=ModularPrompt(), task_scores={"t1": 0.2, "t2": 0.2})

    frontier = ParetoFrontier.get_frontier([c1, c2, c3])
    frontier_ids = {c.id for c in frontier}
    assert "c1" in frontier_ids
    assert "c2" in frontier_ids
    assert "c3" not in frontier_ids  # Dominated by both c1 and c2


def test_gepa_prompt_optimizer_propose_and_merge():
    mod_reply = AIMessage(content="Always inspect logs before modifying service configuration.")
    model = ScriptedChatModel(script=[mod_reply, mod_reply, mod_reply, mod_reply])

    opt = GEPAPromptOptimizer(model=model, n_candidates=2, seed=123)
    base_cand = opt.seed_population()

    # Trajectories with 1 success and 1 failure
    step = StepRecord(
        step_idx=1,
        action=Action(name="patch_service_config", kwargs={"service_name": "db"}),
        observation="Service crashed",
    )
    traj_fail = Trajectory(task_id="t1", success=False, score=0.0, steps=[step])
    traj_ok = Trajectory(task_id="t2", success=True, score=1.0, steps=[step])

    candidates = opt.propose(
        current_prompt=base_cand.modular_prompt.render(),
        trajectories=[traj_fail, traj_ok],
        generation=1,
        benchmark="devops",
    )

    assert len(candidates) >= 2
    for cand in candidates:
        assert "{memory_section}" in cand.text
        assert "{guidance_section}" in cand.text


def test_gepa_crossover_merge():
    p1 = ModularPrompt(planning="Plan P1", tool_policy="Tool P1")
    p2 = ModularPrompt(planning="Plan P2", tool_policy="Tool P2")
    opt = GEPAPromptOptimizer(model=ScriptedChatModel(script=[]), seed=42)
    merged = opt.merge(p1, p2)
    assert merged.planning == "Plan P1"
    assert merged.tool_policy == "Tool P2"


# ---------------------------------------------------------------------------
# 3. Sandbox Variable REPL (ScaffoldRLM)
# ---------------------------------------------------------------------------


def test_repl_variable_management():
    session = REPLSession()
    desc = session.set_variable("log_data", "2026-09-01 Error connection refused at port 5432")
    assert desc.name == "log_data"
    assert desc.length == len("2026-09-01 Error connection refused at port 5432")
    assert "log_data" in session.describe_all()


def test_repl_execution():
    session = REPLSession()
    session.set_variable("raw_nums", [10, 20, 30, 40])
    res = session.execute("total = sum(raw_nums); print(f'SUM={total}')")
    assert res.success is True
    assert "SUM=100" in res.output
    assert session.get_variable("total") == 100
    assert "total" in res.new_vars


def test_repl_execution_error_handling():
    session = REPLSession()
    res = session.execute("x = 1 / 0")
    assert res.success is False
    assert "ZeroDivisionError" in res.error


def test_repl_sub_llm_query_and_budget():
    sub_model = ScriptedChatModel(script=[AIMessage(content="Root cause is database port timeout.")])
    session = REPLSession(sub_lm=sub_model, max_sub_llm_calls=2)
    session.set_variable("incident_log", "DB Proxy failed to connect to Postgres on port 5432")

    ans1 = session.query_llm("What is the root cause?", "incident_log")
    assert "database port timeout" in ans1

    # Second call
    session.query_llm("Query 2", "incident_log")
    # Third call should hit budget cap
    ans3 = session.query_llm("Query 3", "incident_log")
    assert "max_sub_llm_calls" in ans3


def test_repl_tools_definitions():
    defs = REPLTools.get_tool_definitions()
    names = [d["function"]["name"] for d in defs]
    assert "repl_exec" in names
    assert "repl_inspect" in names
    assert "repl_list_vars" in names


# ---------------------------------------------------------------------------
# 4. Dynamic Code Scaffolding (FlexScaffold)
# ---------------------------------------------------------------------------


def test_flex_module_compilation_and_execution():
    code = (
        "def compute_delta(a, b):\n"
        "    return a - b\n\n"
        "def filter_text(text):\n"
        "    return text.upper()\n"
    )
    mod = FlexModule(name="math_tools", module_src=code)
    assert mod.compile_error == ""

    res = mod.call("compute_delta", 10, 3)
    assert res.success is True
    assert res.result == 7

    res2 = mod.call("filter_text", "hello")
    assert res2.success is True
    assert res2.result == "HELLO"


def test_flex_module_syntax_error_interception():
    bad_code = "def broken(\n  return invalid"
    mod = FlexModule(name="broken_module", module_src=bad_code)
    assert mod.compile_error != ""
    res = mod.call("broken")
    assert res.success is False
    assert "Module compilation failed" in res.error


def test_flex_rule_in_environment_harness():
    bench = DevOpsBenchmark()
    base_env = bench.make_env("db_pool")

    # FlexModule that adds a prefix to observations
    rule_code = (
        "from meta_evolver.core.types import Observation\n"
        "def filter_observation(obs, env_state):\n"
        "    return Observation(text='[FLEX_PREFIX] ' + obs.text, data=obs.data)\n"
    )
    mod = FlexModule(name="prefix_rule", module_src=rule_code)
    flex_harness = FlexRule(inner=base_env, module=mod)

    reset_res = flex_harness.reset(options={"task_id": "db_pool"})
    assert "[FLEX_PREFIX]" in reset_res.observation.text


def test_flex_proposer_extraction():
    model = ScriptedChatModel(
        script=[
            AIMessage(
                content="Here is the module:\n```python\ndef sanitize(s):\n    return s.strip()\n```"
            )
        ]
    )
    proposer = FlexProposer(model=model)
    mod = proposer.propose("sanitizer", "Sanitize string inputs")
    assert mod.name == "sanitizer"
    res = mod.call("sanitize", "  hello world  ")
    assert res.success is True
    assert res.result == "hello world"


# ---------------------------------------------------------------------------
# 5. OpenTelemetry and MLflow Tracing (TelemetryTracer)
# ---------------------------------------------------------------------------


def test_telemetry_tracer_span_hierarchy(tmp_path: Path):
    tracer = TelemetryTracer(task_id="task_123", run_id="run_abc")

    with tracer.span("Episode", node_name="episode_root", attributes={"model": "gemini-3.7"}) as root_span:
        with tracer.span("prepare", node_name="prepare"):
            pass
        with tracer.span("think", node_name="think", attributes={"tokens": 120}) as think_span:
            think_span.add_event("llm_call", {"latency_ms": 45.0})
        with tracer.span("act", node_name="act", attributes={"tool": "inspect_logs"}):
            pass

    assert len(tracer.spans) == 4
    assert root_span.duration_ms >= 0.0
    assert root_span.parent_span_id is None

    # Check child spans have parent
    child_spans = [s for s in tracer.spans if s.parent_span_id == root_span.span_id]
    assert len(child_spans) == 3

    # Test MLflow export format
    mlflow_data = tracer.to_mlflow_trace()
    assert mlflow_data["task_id"] == "task_123"
    assert len(mlflow_data["spans"]) == 4

    # Test OTel export format
    otel_data = tracer.to_opentelemetry_format()
    assert "resourceSpans" in otel_data

    # Test file saving
    save_file = tmp_path / "traces" / "trace.json"
    tracer.save_to_file(save_file)
    assert save_file.exists()
    loaded = json.loads(save_file.read_text(encoding="utf-8"))
    assert loaded["trace_id"] == tracer.trace_id


def test_telemetry_tracer_async_context():
    async def _run():
        tracer = TelemetryTracer(task_id="async_task")
        async with tracer.aspan("async_step", node_name="think") as sp:
            await asyncio.sleep(0.01)
        assert sp.duration_ms > 0.0
        assert sp.status == "OK"

    asyncio.run(_run())
