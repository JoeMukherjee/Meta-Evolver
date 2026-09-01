"""Graph state schemas.

Two graphs, two states. Both are ``TypedDict`` with ``Annotated`` reducers on
the accumulating fields, which is what lets LangGraph merge concurrent node
returns instead of last-writer-wins.

A note on what lives in state and what does not. Live objects -- the
environment, the LLM client, the memory bank -- are *not* state. They are not
serializable, so putting them in state breaks checkpointing, and they are
shared rather than per-branch, so a reducer over them is meaningless. They are
bound into the node closures at build time instead; state carries only the
data that describes the run.

The exception is ``env`` and ``controller`` on ``EpisodeState``: an episode is
inseparable from its environment instance, and the whole episode graph runs
within a single ``invoke`` on one thread. They are marked as such below.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from meta_evolver.core.types import (
    GenerationReport,
    MemoryItem,
    StepRecord,
    Trajectory,
)


def _last(_old: Any, new: Any) -> Any:
    """Reducer for scalars written by exactly one node per superstep."""
    return new


class EpisodeState(TypedDict, total=False):
    """State of one agent episode.

    The flow is ``prepare -> think -> act -> adapt -> (think | finalize)``.
    ``adapt`` is what distinguishes this from a stock ReAct loop: it is a real
    node, so stagnation detection and memory eviction are inspectable,
    checkpointable, and testable in isolation rather than buried in a while
    loop.
    """

    # -- identity / inputs (set once by the caller) ------------------------
    task_id: str
    benchmark: str
    instruction: str
    generation: int
    prompt_template: str
    prompt_version: str
    max_steps: int

    # -- live handles (single-threaded within one episode invoke) ----------
    env: Any
    controller: Any

    # -- accumulating ------------------------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]
    """LangChain messages under LangGraph's own reducer.

    ``add_messages`` rather than ``operator.add`` because it de-duplicates by
    message id and supports in-place updates -- which is what lets a node
    revise or remove a message later (trimming history, redacting a tool
    result) instead of only ever appending."""

    steps: Annotated[list[StepRecord], operator.add]

    # -- rolling scalars ---------------------------------------------------
    step_idx: int
    observation: str
    admissible: list[str]
    pending_action: dict[str, Any] | None
    pending_thought: str
    pending_tool_call_id: str
    """Echoed back on the ToolMessage. A tool result whose id does not match
    the call that produced it is rejected by every provider."""
    nudges: int
    terminated: bool
    truncated: bool
    reward: float
    tokens: int
    error: str

    # -- outputs -----------------------------------------------------------
    retrieved_memory_ids: list[str]
    trajectory: Trajectory | None


class EvolutionState(TypedDict, total=False):
    """State of the outer self-improvement loop.

    One pass through the graph is one generation. Everything that carries
    across generations -- the prompt, the bank's contents, the curriculum
    level -- either lives here or lives in an object the nodes hold; the
    reports list is the audit trail of what each generation changed.
    """

    benchmark: str
    generation: int
    max_generations: int

    # -- the evolving artifacts -------------------------------------------
    prompt_template: str
    prompt_version: str
    curriculum_level: float

    # -- this generation's work --------------------------------------------
    task_ids: list[str]
    validation_task_ids: list[str]
    trajectories: Annotated[list[Trajectory], operator.add]
    induced: list[MemoryItem]

    # -- what this generation's learners did (read by ``checkpoint``) ------
    memories_before: int
    memories_added: int
    memories_pruned: int
    validation_pass_rate: float | None
    prompt_note: str

    # -- history / control -------------------------------------------------
    last_outcomes: dict[str, bool]
    """``task_id -> success`` from the previous generation, for regression
    detection. Kept in state rather than on a node closure so a resumed run
    picks up the comparison instead of silently reporting zero churn."""

    reports: Annotated[list[GenerationReport], operator.add]
    best_pass_rate: float
    generations_without_gain: int
    stop_reason: str


class RolloutInput(TypedDict, total=False):
    """The slice of :class:`EvolutionState` a fanned-out rollout needs.

    Deliberately narrow: ``Send`` copies its payload into every branch, and
    forwarding the whole accumulated state would duplicate every trajectory
    collected so far once per concurrent task.
    """

    task_id: str
    generation: int
    curriculum_level: float
    prompt_template: str
    prompt_version: str
