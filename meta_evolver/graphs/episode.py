"""The episode graph -- one agent rollout, as an explicit state machine.

::

      START -> prepare -> think -> route
                            ^        |
                            |        +--(no tool call)--> nudge --+
                            |        |                            |
                            |        +--(tool call)--> act -> adapt
                            |                                    |
                            +-------------(continue)-------------+
                                                                 |
                                          (done / budget) --> finalize -> END

Why a graph rather than a ``while`` loop. The loop version of this agent had
its retrieval, its stagnation detection and its termination conditions
interleaved inside one function, and the consequences were the ordinary ones:
the eviction rule could not be tested without an LLM, a change to termination
risked breaking retrieval, and there was nowhere to hang an interrupt. As
nodes, each concern is separately testable, the transitions are the routing
functions, and every superstep is a checkpoint -- so an episode can be
resumed, inspected mid-flight, or paused for human review without any of the
nodes knowing about it.

``adapt`` is the node that earns the structure. It runs after every action,
owns the entire OOD-mitigation policy, and communicates with ``think`` only
through state.

Messages are LangChain ``AnyMessage`` objects under the ``add_messages``
reducer, so tool calls arrive already normalized on ``AIMessage.tool_calls``
and nothing here re-implements a provider's wire format.

Nodes are **async**. An episode is almost entirely waiting on a model, so a
synchronous node holds a thread for the duration of a call it is not using.
With async nodes, a generation's rollouts -- fanned out by the evolution graph
-- overlap on one event loop, and wall-clock drops to roughly the slowest
episode rather than the sum of all of them. :func:`run_episode` keeps a
synchronous entry point for callers that have no loop of their own.

One caveat worth naming: ``env.step`` is called directly rather than through a
thread. The benchmarks here are in-process simulators where that is free. An
environment that does real I/O should be stepped with ``asyncio.to_thread`` so
it does not stall the loop for every other rollout.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from meta_evolver.adaptive.controller import (
    AdaptiveControllerConfig,
    AdaptiveExplorationController,
)
from meta_evolver.core.aio import selector_loop_factory
from meta_evolver.core.types import Action, StepRecord, Trajectory
from meta_evolver.graphs.state import EpisodeRuntime, EpisodeState
from meta_evolver.llm.client import message_text
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.prompts.templates import render_system_prompt
from meta_evolver.tools.assertions import AssertionRunner
from meta_evolver.tools.routing import ToolRouter

#: Cap on consecutive turns where the model answers with prose instead of a
#: tool call. Without it a chatty model burns the whole step budget talking.
MAX_NUDGES = 3


def build_episode_graph(
    model: BaseChatModel,
    bank: ReasoningMemoryBank | None = None,
    retrieval_k: int = 4,
    retrieval_mode: str = "mmr",
    controller_config: AdaptiveControllerConfig | None = None,
    observation_chars: int = 4000,
    tool_router: ToolRouter | None = None,
    assertion_runner: AssertionRunner | None = None,
    max_assertion_retries: int = 3,
    checkpointer: Any = None,
):
    """Compile the episode graph.

    Live handles -- the chat model, the bank -- are closed over rather than
    placed in state: they are not serializable, and a checkpointer would choke
    on them.

    ``tool_router`` is optional. Supply one when the environment exposes a
    large registry and most of it is irrelevant to any single task; without it
    the agent sees every tool the environment offers.

    ``checkpointer`` persists state after every superstep, which is what makes
    an episode resumable and inspectable mid-flight. The environment and the
    adaptive controller travel in ``config`` rather than state (see
    :class:`~meta_evolver.graphs.state.EpisodeRuntime`) precisely so that they
    do not have to serialize: a resumed episode recovers the transcript and
    the step record and is handed a fresh environment, which is the right
    granularity here -- episodes are cheap to replay, generations are not.
    """
    cfg = controller_config or AdaptiveControllerConfig()

    # -- nodes -------------------------------------------------------------

    async def prepare(state: EpisodeState, config: RunnableConfig) -> dict[str, Any]:
        """Reset the environment, retrieve memories, seed the conversation."""
        runtime = _runtime(config)
        env = runtime.env
        task_id = state.get("task_id", "")
        reset = env.reset(options={"task_id": task_id})
        obs = reset.observation
        instruction = (
            state.get("instruction")
            or reset.info.get("instruction")
            or reset.info.get("title")
            or task_id
        )

        retrieved = []
        if bank is not None and retrieval_k > 0:
            query = f"{instruction}\n{obs.text}"[:2000]
            retrieved = bank.retrieve(query, k=retrieval_k, mode=retrieval_mode)

        controller = AdaptiveExplorationController(memories=retrieved, config=cfg)
        controller.record_candidates(_admissible(obs))
        runtime.controller = controller

        return {
            "instruction": instruction,
            "observation": obs.text[:observation_chars],
            "admissible": _admissible(obs),
            "retrieved_memory_ids": [m.id for m in retrieved],
            "messages": [
                HumanMessage(
                    content=(
                        f"Task: {instruction}\n\n"
                        f"Current observation:\n{obs.text[:observation_chars]}"
                    )
                )
            ],
            "steps": [],
            "step_idx": 0,
            "nudges": 0,
            "assertion_retries": 0,
            "assertion_warnings": [],
            "pending_assertion_failures": [],
            "terminated": False,
            "truncated": False,
            "reward": 0.0,
            "tokens": 0,
            "error": "",
            "pending_action": None,
            "pending_thought": "",
        }

    async def think(state: EpisodeState, config: RunnableConfig) -> dict[str, Any]:
        """Ask the model for the next action.

        The system prompt is rebuilt every turn, because the two injected
        sections are live: the memory block can vanish mid-episode when the
        controller evicts it, and the guidance block reflects the search state
        as of this step. A prompt built once at episode start would keep
        re-asserting a prior the controller has already retired.
        """
        runtime = _runtime(config)
        env, controller = runtime.env, runtime.controller
        system = render_system_prompt(
            state.get("prompt_template", ""),
            memory_section=controller.memory_block(),
            guidance_section=controller.guidance_block(state.get("admissible")),
        )
        messages = [SystemMessage(content=system), *state["messages"]]

        tools = env.available_tools()
        if tool_router is not None:
            # Re-routed every turn, not once at reset: what the agent needs
            # after a diagnosis differs from what it needed before one.
            query = " | ".join(
                [str(state.get("instruction", "")), str(state.get("observation", ""))]
            )
            tools = tool_router.select(tools, query)

        try:
            reply = await model.bind_tools(tools).ainvoke(messages)
        except Exception as exc:
            # An infrastructure failure is not a task failure. Recording it as
            # `error` keeps it out of the learning signal downstream.
            return {"error": f"llm: {type(exc).__name__}: {exc}", "terminated": True}

        thought = message_text(reply)
        tokens = state.get("tokens", 0) + _token_count(reply)

        if not reply.tool_calls:
            # The reply is not appended here: `nudge` owns that, so a prose
            # turn produces exactly one assistant message in the transcript.
            return {"pending_action": None, "pending_thought": thought, "tokens": tokens}

        call = reply.tool_calls[0]
        proposed_action = Action(name=call["name"], kwargs=call.get("args") or {})

        # Run in-flight assertions if configured
        if assertion_runner is not None:
            env_st = getattr(env, "get_env_state", lambda: {})()
            results = assertion_runner.evaluate(proposed_action, state, env_st)
            hard = assertion_runner.hard_failures(results)
            soft = assertion_runner.soft_warnings(results)

            if hard and state.get("assertion_retries", 0) < max_assertion_retries:
                return {
                    "pending_action": None,
                    "pending_thought": thought,
                    "pending_assertion_failures": [h.message for h in hard],
                    "assertion_warnings": [s.message for s in soft],
                    "tokens": tokens,
                    "nudges": 0,
                }

        return {
            "messages": [reply],
            "pending_action": {"name": call["name"], "kwargs": call.get("args") or {}},
            "pending_thought": thought,
            "pending_tool_call_id": call.get("id") or f"call_{state.get('step_idx', 0) + 1}",
            "pending_assertion_failures": [],
            "tokens": tokens,
            "nudges": 0,
        }

    async def assert_retry(state: EpisodeState) -> dict[str, Any]:
        """An assertion failed. Feed back the error without advancing the environment."""
        failures = state.get("pending_assertion_failures") or ["Action validation check failed."]
        thought = state.get("pending_thought", "")
        feedback_text = "; ".join(failures)
        return {
            "assertion_retries": state.get("assertion_retries", 0) + 1,
            "pending_assertion_failures": [],
            "messages": [
                AIMessage(content=thought or "(attempted action)"),
                HumanMessage(
                    content=(
                        f"Action validation assertion failed: {feedback_text}\n"
                        f"Please correct your action call or parameters and try again."
                    )
                ),
            ],
        }

    async def nudge(state: EpisodeState) -> dict[str, Any]:
        """The model answered in prose. Record it and ask again."""
        thought = state.get("pending_thought", "")
        return {
            "nudges": state.get("nudges", 0) + 1,
            "messages": [
                AIMessage(content=thought or "(no content)"),
                HumanMessage(
                    content=(
                        "Respond with a tool call, not prose. Choose the single "
                        "action that best advances the task from the current state."
                    )
                ),
            ],
        }

    async def act(state: EpisodeState, config: RunnableConfig) -> dict[str, Any]:
        """Execute the chosen action against the environment."""
        env = _runtime(config).env
        pending = state["pending_action"] or {}
        action = Action(name=pending.get("name", ""), kwargs=pending.get("kwargs") or {})
        step_idx = state.get("step_idx", 0) + 1
        call_id = state.get("pending_tool_call_id") or f"call_{step_idx}"

        started = time.time()
        try:
            resp = env.step(action)
        except Exception as exc:
            # A crash inside the environment is the benchmark's bug, not the
            # agent's. Surface it as an episode error rather than a low score.
            return {"error": f"env: {type(exc).__name__}: {exc}", "terminated": True}
        latency_ms = (time.time() - started) * 1000.0

        info = resp.info or {}
        result = info.get("result") if isinstance(info.get("result"), dict) else {}
        obs_text = (resp.observation.text or "")[:observation_chars]
        tool_content = obs_text or json.dumps(result, default=str)

        record = StepRecord(
            step_idx=step_idx,
            thought=state.get("pending_thought", "") or "",
            action=action,
            observation=obs_text,
            result=result,
            reward=resp.reward,
            terminated=resp.terminated,
            truncated=resp.truncated,
            blocked=bool(info.get("blocked")),
            perturbed=bool(info.get("perturbed")),
            latency_ms=latency_ms,
        )

        return {
            "steps": [record],
            "step_idx": step_idx,
            "observation": obs_text,
            "admissible": _admissible(resp.observation),
            "reward": max(state.get("reward", 0.0), resp.reward),
            "terminated": bool(resp.terminated),
            "truncated": bool(resp.truncated),
            "pending_action": None,
            "assertion_retries": 0,
            "pending_assertion_failures": [],
            "messages": [
                ToolMessage(
                    content=str(tool_content), tool_call_id=call_id, name=action.name
                )
            ],
        }

    async def adapt(state: EpisodeState, config: RunnableConfig) -> dict[str, Any]:
        """Update exploration state; evict the memory prior if it has stalled.

        Mutates the controller in place -- it is the same object for the whole
        episode -- and returns nothing, because its influence reaches ``think``
        through the prompt sections rather than through state.
        """
        controller = _runtime(config).controller
        steps = state.get("steps") or []
        if steps and controller is not None:
            last = steps[-1]
            controller.record_step(
                action_text=last.action.render(),
                observation=last.observation,
                reward=last.reward,
                admissible=state.get("admissible"),
            )
        return {}

    async def finalize(state: EpisodeState, config: RunnableConfig) -> dict[str, Any]:
        """Score the episode and assemble the trajectory."""
        runtime = _runtime(config)
        env, controller = runtime.env, runtime.controller
        steps = state.get("steps") or []
        error = state.get("error", "")

        try:
            result = env.evaluate()
            success, score, metrics = result.success, result.score, dict(result.metrics)
        except Exception as exc:
            success, score, metrics = False, 0.0, {}
            error = error or f"evaluate: {type(exc).__name__}: {exc}"

        if error:
            # An episode that never produced a verdict must not be scored as a
            # loss; `usable` is False and every learner skips it.
            success, score = False, 0.0

        trajectory = Trajectory(
            task_id=state.get("task_id", ""),
            benchmark=state.get("benchmark", ""),
            instruction=state.get("instruction", ""),
            steps=steps,
            success=bool(success),
            score=float(score),
            final_reward=float(state.get("reward", 0.0)),
            metrics=metrics,
            error=error,
            retrieved_memory_ids=list(state.get("retrieved_memory_ids") or []),
            memory_evicted_at=getattr(controller, "eviction_step", None),
            prompt_version=state.get("prompt_version", "base"),
            generation=int(state.get("generation", 0)),
            duration_ms=sum(s.latency_ms for s in steps),
            tokens=int(state.get("tokens", 0)),
            rollout_index=int(state.get("rollout_index", 0)),
        )
        return {"trajectory": trajectory}

    # -- routing -----------------------------------------------------------

    def route_after_think(
        state: EpisodeState,
    ) -> Literal["act", "nudge", "assert_retry", "finalize"]:
        if state.get("error"):
            return "finalize"
        if state.get("pending_assertion_failures"):
            return "assert_retry"
        if state.get("pending_action"):
            return "act"
        if state.get("nudges", 0) >= MAX_NUDGES:
            return "finalize"
        return "nudge"

    def route_after_adapt(state: EpisodeState) -> Literal["think", "finalize"]:
        if state.get("error") or state.get("terminated") or state.get("truncated"):
            return "finalize"
        if state.get("step_idx", 0) >= state.get("max_steps", 15):
            return "finalize"
        return "think"

    def route_after_nudge(state: EpisodeState) -> Literal["think", "finalize"]:
        return "finalize" if state.get("nudges", 0) >= MAX_NUDGES else "think"

    def route_after_act(state: EpisodeState) -> Literal["adapt", "finalize"]:
        return "finalize" if state.get("error") else "adapt"

    # -- assembly ----------------------------------------------------------

    graph = StateGraph(EpisodeState)
    graph.add_node("prepare", prepare)
    graph.add_node("think", think)
    graph.add_node("nudge", nudge)
    graph.add_node("assert_retry", assert_retry)
    graph.add_node("act", act)
    graph.add_node("adapt", adapt)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "think")
    graph.add_conditional_edges(
        "think", route_after_think, ["act", "nudge", "assert_retry", "finalize"]
    )
    graph.add_edge("assert_retry", "think")
    graph.add_conditional_edges("nudge", route_after_nudge, ["think", "finalize"])
    graph.add_conditional_edges("act", route_after_act, ["adapt", "finalize"])
    graph.add_conditional_edges("adapt", route_after_adapt, ["think", "finalize"])
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def _runtime(config: RunnableConfig) -> EpisodeRuntime:
    """The live objects for this episode, from LangGraph's config channel."""
    runtime = (config or {}).get("configurable", {}).get("runtime")
    if runtime is None:
        raise RuntimeError(
            "no EpisodeRuntime in config; call arun_episode() rather than "
            "invoking the episode graph directly"
        )
    return runtime


def _admissible(observation: Any) -> list[str]:
    """Pull the admissible-command list out of an observation, if any.

    Text-game benchmarks expose it; tool-calling ones do not. The controller
    treats an empty list as "unknown action space" and simply tracks fewer
    entities, so both work through the same path.
    """
    data = getattr(observation, "data", None) or {}
    commands = data.get("admissible_commands") or data.get("admissible") or []
    return [str(c) for c in commands] if isinstance(commands, Sequence) else []


def _token_count(message: AIMessage) -> int:
    """Total tokens for a reply, when the provider reported them."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("total_tokens") or 0)
    return 0


async def arun_episode(
    graph: Any,
    env: Any,
    task_id: str,
    prompt_template: str,
    benchmark: str = "",
    instruction: str = "",
    max_steps: int = 15,
    generation: int = 0,
    prompt_version: str = "base",
    rollout_index: int = 0,
    thread_id: str | None = None,
    recursion_limit: int | None = None,
) -> Trajectory:
    """Run one episode to completion and return its trajectory.

    ``recursion_limit`` guards LangGraph's superstep counter. Each agent step
    costs roughly three supersteps (think, act, adapt), plus nudges, so the
    default is derived from ``max_steps`` rather than left at LangGraph's 25 --
    which a 15-step episode would otherwise trip long before finishing.

    ``thread_id`` addresses the checkpoint stream. It must be unique per
    episode -- including per rollout when a task is attempted several times --
    or concurrent rollouts of one task would write over each other's state.
    """
    limit = recursion_limit or (max_steps * 4 + 12)
    initial: EpisodeState = {
        "task_id": task_id,
        "benchmark": benchmark,
        "instruction": instruction,
        "prompt_template": prompt_template,
        "prompt_version": prompt_version,
        "generation": generation,
        "rollout_index": rollout_index,
        "max_steps": max_steps,
        "messages": [],
        "steps": [],
    }
    configurable: dict[str, Any] = {"runtime": EpisodeRuntime(env=env)}
    if thread_id is not None:
        configurable["thread_id"] = thread_id
    else:
        # A checkpointer requires a thread id. Deriving one keeps the graph
        # usable without a caller having to care whether it is checkpointed.
        configurable["thread_id"] = f"{benchmark}:{task_id}:{generation}:{rollout_index}"
    config: dict[str, Any] = {"recursion_limit": limit, "configurable": configurable}

    try:
        final = await graph.ainvoke(initial, config=config)
    except Exception as exc:
        return Trajectory(
            task_id=task_id,
            benchmark=benchmark,
            instruction=instruction,
            error=f"graph: {type(exc).__name__}: {exc}",
            generation=generation,
            prompt_version=prompt_version,
            rollout_index=rollout_index,
        )

    trajectory = final.get("trajectory")
    if trajectory is None:  # pragma: no cover - finalize always sets it
        return Trajectory(
            task_id=task_id,
            benchmark=benchmark,
            instruction=instruction,
            error="episode graph produced no trajectory",
            generation=generation,
            rollout_index=rollout_index,
        )
    return trajectory


def run_episode(*args: Any, **kwargs: Any) -> Trajectory:
    """Synchronous :func:`arun_episode`, for callers with no event loop.

    A one-shot loop, since a single episode owns nothing that has to outlive
    the call. An object that holds a connection pool across calls -- like
    :class:`~meta_evolver.core.evolver.MetaEvolver` -- needs a persistent one
    instead; see :mod:`meta_evolver.core.aio`.
    """
    with asyncio.Runner(loop_factory=selector_loop_factory()) as runner:
        return runner.run(arun_episode(*args, **kwargs))


__all__ = ["MAX_NUDGES", "arun_episode", "build_episode_graph", "run_episode"]
