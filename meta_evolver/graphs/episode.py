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
"""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from meta_evolver.adaptive.controller import (
    AdaptiveControllerConfig,
    AdaptiveExplorationController,
)
from meta_evolver.core.types import Action, StepRecord, Trajectory
from meta_evolver.graphs.state import EpisodeState
from meta_evolver.llm.client import message_text
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.prompts.templates import render_system_prompt
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
):
    """Compile the episode graph.

    Live handles -- the chat model, the bank -- are closed over rather than
    placed in state: they are not serializable, and a checkpointer would choke
    on them.

    ``tool_router`` is optional. Supply one when the environment exposes a
    large registry and most of it is irrelevant to any single task; without it
    the agent sees every tool the environment offers.
    """
    cfg = controller_config or AdaptiveControllerConfig()

    # -- nodes -------------------------------------------------------------

    def prepare(state: EpisodeState) -> dict[str, Any]:
        """Reset the environment, retrieve memories, seed the conversation."""
        env = state["env"]
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

        return {
            "env": env,
            "controller": controller,
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
            "terminated": False,
            "truncated": False,
            "reward": 0.0,
            "tokens": 0,
            "error": "",
            "pending_action": None,
            "pending_thought": "",
        }

    def think(state: EpisodeState) -> dict[str, Any]:
        """Ask the model for the next action.

        The system prompt is rebuilt every turn, because the two injected
        sections are live: the memory block can vanish mid-episode when the
        controller evicts it, and the guidance block reflects the search state
        as of this step. A prompt built once at episode start would keep
        re-asserting a prior the controller has already retired.
        """
        env, controller = state["env"], state["controller"]
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
            reply = model.bind_tools(tools).invoke(messages)
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
        return {
            "messages": [reply],
            "pending_action": {"name": call["name"], "kwargs": call.get("args") or {}},
            "pending_thought": thought,
            "pending_tool_call_id": call.get("id") or f"call_{state.get('step_idx', 0) + 1}",
            "tokens": tokens,
            "nudges": 0,
        }

    def nudge(state: EpisodeState) -> dict[str, Any]:
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

    def act(state: EpisodeState) -> dict[str, Any]:
        """Execute the chosen action against the environment."""
        env = state["env"]
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
            "messages": [
                ToolMessage(
                    content=str(tool_content), tool_call_id=call_id, name=action.name
                )
            ],
        }

    def adapt(state: EpisodeState) -> dict[str, Any]:
        """Update exploration state; evict the memory prior if it has stalled.

        Mutates the controller in place -- it is the same object for the whole
        episode -- and returns nothing, because its influence reaches ``think``
        through the prompt sections rather than through state.
        """
        controller = state["controller"]
        steps = state.get("steps") or []
        if steps:
            last = steps[-1]
            controller.record_step(
                action_text=last.action.render(),
                observation=last.observation,
                reward=last.reward,
                admissible=state.get("admissible"),
            )
        return {}

    def finalize(state: EpisodeState) -> dict[str, Any]:
        """Score the episode and assemble the trajectory."""
        env, controller = state["env"], state.get("controller")
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
        )
        return {"trajectory": trajectory}

    # -- routing -----------------------------------------------------------

    def route_after_think(state: EpisodeState) -> Literal["act", "nudge", "finalize"]:
        if state.get("error"):
            return "finalize"
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
    graph.add_node("act", act)
    graph.add_node("adapt", adapt)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "think")
    graph.add_conditional_edges("think", route_after_think, ["act", "nudge", "finalize"])
    graph.add_conditional_edges("nudge", route_after_nudge, ["think", "finalize"])
    graph.add_conditional_edges("act", route_after_act, ["adapt", "finalize"])
    graph.add_conditional_edges("adapt", route_after_adapt, ["think", "finalize"])
    graph.add_edge("finalize", END)

    return graph.compile()


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


def run_episode(
    graph: Any,
    env: Any,
    task_id: str,
    prompt_template: str,
    benchmark: str = "",
    instruction: str = "",
    max_steps: int = 15,
    generation: int = 0,
    prompt_version: str = "base",
    recursion_limit: int | None = None,
) -> Trajectory:
    """Run one episode to completion and return its trajectory.

    ``recursion_limit`` guards LangGraph's superstep counter. Each agent step
    costs roughly three supersteps (think, act, adapt), plus nudges, so the
    default is derived from ``max_steps`` rather than left at LangGraph's 25 --
    which a 15-step episode would otherwise trip long before finishing.
    """
    limit = recursion_limit or (max_steps * 4 + 12)
    initial: EpisodeState = {
        "env": env,
        "task_id": task_id,
        "benchmark": benchmark,
        "instruction": instruction,
        "prompt_template": prompt_template,
        "prompt_version": prompt_version,
        "generation": generation,
        "max_steps": max_steps,
        "messages": [],
        "steps": [],
    }
    try:
        final = graph.invoke(initial, config={"recursion_limit": limit})
    except Exception as exc:
        return Trajectory(
            task_id=task_id,
            benchmark=benchmark,
            instruction=instruction,
            error=f"graph: {type(exc).__name__}: {exc}",
            generation=generation,
            prompt_version=prompt_version,
        )

    trajectory = final.get("trajectory")
    if trajectory is None:  # pragma: no cover - finalize always sets it
        return Trajectory(
            task_id=task_id,
            benchmark=benchmark,
            instruction=instruction,
            error="episode graph produced no trajectory",
            generation=generation,
        )
    return trajectory


__all__ = ["MAX_NUDGES", "build_episode_graph", "run_episode"]
