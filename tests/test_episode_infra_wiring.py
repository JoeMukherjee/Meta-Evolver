"""ToolRouter and AssertionRunner, threaded from MetaEvolver's own
constructor through to the live episode graph -- both were already
supported by build_episode_graph, but MetaEvolver never passed them."""
from __future__ import annotations

from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.tools.assertions import AssertionRunner, NonEmptyArgsAssertion
from meta_evolver.tools.routing import ToolRouter


def test_metaevolver_never_passes_a_router_or_runner_by_default():
    evolver = MetaEvolver(
        benchmark="devops",
        chat_model=ScriptedChatModel(responder=lambda m, t: tool_call_message("inspect_service_logs", service_name="db-proxy")),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(generations=1, max_steps=1, optimize_prompt=False, curriculum=False),
        telemetry=False,
    )
    assert evolver._tool_router is None
    assert evolver._assertion_runner is None


def test_tool_router_passed_to_metaevolver_narrows_what_the_agent_sees():
    seen: list[int] = []

    def responder(messages, tools):
        seen.append(len(tools or []))
        return tool_call_message("inspect_service_logs", service_name="db-proxy")

    router = ToolRouter(k=2, min_tools_to_route=2)
    evolver = MetaEvolver(
        benchmark="devops",
        chat_model=ScriptedChatModel(responder=responder),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(generations=1, max_steps=1, optimize_prompt=False, curriculum=False),
        tool_router=router,
        telemetry=False,
    )
    env = evolver.benchmark.make_env("db_pool")
    from meta_evolver.graphs.episode import run_episode

    run_episode(
        evolver.episode_graph,
        env=env,
        task_id="db_pool",
        benchmark=evolver.benchmark.name,
        instruction=evolver.benchmark.instruction_for("db_pool"),
        prompt_template=evolver.benchmark.system_prompt(),
        max_steps=1,
    )
    full_count = len(env.available_tools())
    assert seen[0] < full_count
    assert seen[0] == 2


def test_assertion_runner_passed_to_metaevolver_triggers_a_real_retry():
    def retry_responder(messages, tools):
        for msg in messages:
            if "empty" in str(getattr(msg, "content", "")).lower():
                return tool_call_message("inspect_service_logs", service_name="payment-gateway")
        return tool_call_message("inspect_service_logs", service_name="")

    runner = AssertionRunner([NonEmptyArgsAssertion(required_keys=["service_name"], is_hard=True)])
    evolver = MetaEvolver(
        benchmark="devops",
        chat_model=ScriptedChatModel(responder=retry_responder),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(generations=1, max_steps=5, optimize_prompt=False, curriculum=False),
        assertion_runner=runner,
        telemetry=False,
    )
    env = evolver.benchmark.make_env("db_pool")
    from meta_evolver.graphs.episode import run_episode

    trajectory = run_episode(
        evolver.episode_graph,
        env=env,
        task_id="db_pool",
        benchmark=evolver.benchmark.name,
        instruction=evolver.benchmark.instruction_for("db_pool"),
        prompt_template=evolver.benchmark.system_prompt(),
        max_steps=5,
    )
    # The corrected call actually reached the environment -- proof the
    # assertion runner intercepted the first, empty-arg attempt rather than
    # letting it through unchecked.
    assert any(
        s.action.name == "inspect_service_logs" and s.action.kwargs.get("service_name")
        for s in trajectory.steps
    )
