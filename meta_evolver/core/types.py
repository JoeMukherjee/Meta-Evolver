"""Core data contracts.

Every component in Meta-Evolver -- benchmarks, graphs, memory, telemetry --
exchanges only the types defined here. Keeping the contracts in one module is
what lets a new benchmark be plugged in without touching the engine: the
engine never sees a benchmark-specific type.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Action / Observation -- deliberately generic. A benchmark gives them meaning.
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """A tool call: a name plus JSON-serializable arguments.

    Text-action environments (ALFWorld, text games) use the convention
    ``Action(name="do", kwargs={"text": "go to fridge 1"})`` so a single
    tool-calling agent loop covers both tool-based and text-based benchmarks.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        if self.name == "do" and set(self.kwargs) == {"text"}:
            return str(self.kwargs["text"])
        return f"{self.name}({json.dumps(self.kwargs, default=str, sort_keys=True)})"


class Blocked(BaseModel):
    """Returned by ``Rules.filter_action`` when a harness rejects an action."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["blocked"] = "blocked"
    reason: str = ""


class Observation(BaseModel):
    """Everything the agent sees. ``data`` carries structured extras such as
    ``admissible_commands`` that a benchmark wants the agent to reason over."""

    model_config = ConfigDict(extra="allow")
    text: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step / episode results
# ---------------------------------------------------------------------------


class EnvResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: Observation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class EnvResetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: Observation
    info: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    success: bool = False
    score: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)


class StepRecord(BaseModel):
    """One agent/environment interaction, rich enough for offline reflection.

    The memory inducer and the prompt optimizer both read trajectories, and
    both need the *observation* rather than only the raw tool return value --
    an earlier iteration of this project recorded only ``result`` while the
    optimizer read ``observation``, so every failure trace it saw was blank.
    Both are kept here, under fixed names, so the two readers cannot drift.
    """

    model_config = ConfigDict(extra="forbid")
    step_idx: int
    thought: str = ""
    action: Action
    observation: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    blocked: bool = False
    perturbed: bool = False
    latency_ms: float = 0.0
    tokens: int = 0

    def render(self) -> str:
        head = f"  step {self.step_idx}: {self.action.render()}"
        obs = (self.observation or json.dumps(self.result, default=str))[:400]
        flags = []
        if self.blocked:
            flags.append("BLOCKED")
        if self.perturbed:
            flags.append("PERTURBED")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        return f"{head}{suffix}\n    -> {obs}"


class Trajectory(BaseModel):
    """A complete episode. The unit of learning."""

    model_config = ConfigDict(extra="forbid")
    task_id: str
    benchmark: str = ""
    instruction: str = ""
    steps: list[StepRecord] = Field(default_factory=list)
    success: bool = False
    score: float = 0.0
    final_reward: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    """Non-empty when the episode ended from an infrastructure failure (an API
    error, a crash) rather than from the agent being wrong. Scoring keeps these
    separate: counting a rate-limit as a task failure poisons every downstream
    signal -- the memory bank, the prompt optimizer, and the curriculum."""
    duration_ms: float = 0.0
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    """Which memories were in the prompt. The credit-assignment pass reads
    this to decide which memories actually earn their retrieval slot."""
    memory_evicted_at: int | None = None
    prompt_version: str = "base"
    generation: int = 0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def usable(self) -> bool:
        """Did this episode produce a signal worth learning from?"""
        return not self.error and bool(self.steps)

    def render(self, max_steps: int = 12) -> str:
        head = (
            f"Task [{self.task_id}] ({self.benchmark}): {self.instruction}\n"
            f"Outcome: {'SUCCESS' if self.success else 'FAILURE'} "
            f"(score={self.score:.2f}, steps={self.n_steps})"
        )
        shown = self.steps[-max_steps:]
        if len(self.steps) > max_steps:
            head += f"\n  ... {len(self.steps) - max_steps} earlier steps elided"
        return head + "\n" + "\n".join(s.render() for s in shown)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class TaskSpec(BaseModel):
    """A single unit of work a benchmark can hand the engine."""

    model_config = ConfigDict(extra="allow")
    task_id: str
    instruction: str = ""
    split: str = "train"
    difficulty: float = 0.5
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MemoryItem(BaseModel):
    """One distilled, reusable strategy.

    Beyond the text, an item carries its own performance record. ``uses`` and
    ``wins`` are updated by credit assignment after every generation, and
    ``utility`` turns them into a score the retriever and the pruner both read.
    That record is what makes the bank improve rather than merely grow.
    """

    model_config = ConfigDict(extra="forbid")
    id: str = ""
    title: str = ""
    scenario: str = "general"
    lesson: str = ""
    procedure: str = ""
    triggers: list[str] = Field(default_factory=list)
    polarity: Literal["success", "failure"] = "success"
    """``failure`` items are anti-patterns -- "this is what went wrong" --
    which are as valuable as positive strategies and are rendered differently
    in the prompt so the agent does not read them as instructions."""
    source_task_ids: list[str] = Field(default_factory=list)
    benchmark: str = ""
    embedding: list[float] | None = None
    uses: int = 0
    wins: int = 0
    created_generation: int = 0
    last_used_generation: int = 0

    @property
    def utility(self) -> float:
        """Posterior mean of a Beta(1,1) prior over "episodes citing this
        memory succeed". An unused item sits at 0.5 -- neither trusted nor
        pruned -- so a new memory gets a fair number of trials before the
        pruner can judge it."""
        return (self.wins + 1.0) / (self.uses + 2.0)

    def key(self) -> str:
        """Stable identity for dedup: scenario plus normalized lesson text."""
        norm = " ".join((self.scenario + " " + self.lesson).lower().split())
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]

    def embed_text(self) -> str:
        return " ".join(
            p for p in (self.scenario, self.title, self.lesson, " ".join(self.triggers)) if p
        )

    def render(self, index: int | None = None) -> str:
        label = f"#{index}" if index is not None else ""
        kind = "ANTI-PATTERN" if self.polarity == "failure" else "STRATEGY"
        head = f"[{kind} {label} | {self.scenario}] {self.title}".strip()
        lines = [head]
        if self.lesson:
            lines.append(f"  Insight: {self.lesson}")
        if self.procedure:
            lines.append(f"  Procedure: {self.procedure}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evolution reporting
# ---------------------------------------------------------------------------


class GenerationReport(BaseModel):
    """What one generation of the outer loop changed, and what it bought."""

    model_config = ConfigDict(extra="forbid")
    generation: int
    benchmark: str = ""
    n_tasks: int = 0
    n_errors: int = 0
    pass_rate: float = 0.0
    avg_steps: float = 0.0
    avg_score: float = 0.0
    regressions: int = 0
    """Tasks that passed last generation and fail now.

    The headline pass rate hides this: a generation that fixes two tasks and
    breaks two looks identical to one that changed nothing. Since every
    channel here rewrites shared state -- one prompt, one bank, for all tasks
    -- churn of exactly that kind is the expected failure mode, so it is
    counted rather than inferred."""

    recoveries: int = 0
    """Tasks that failed last generation and pass now."""

    memories_before: int = 0
    memories_added: int = 0
    memories_pruned: int = 0
    prompt_changed: bool = False
    prompt_version: str = "base"
    curriculum_level: float = 0.0
    validation_pass_rate: float | None = None
    duration_s: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    notes: list[str] = Field(default_factory=list)

    def render(self) -> str:
        return (
            f"gen {self.generation:>2} | pass {self.pass_rate * 100:5.1f}% "
            f"| steps {self.avg_steps:5.2f} | mem {self.memories_before}"
            f"(+{self.memories_added}/-{self.memories_pruned}) "
            f"| prompt {self.prompt_version}"
            f"{' *' if self.prompt_changed else '  '}"
            f"| curriculum {self.curriculum_level:.2f}"
            f"{f' | +{self.recoveries}/-{self.regressions}' if (self.recoveries or self.regressions) else ''}"
            f"{f' | errors {self.n_errors}' if self.n_errors else ''}"
        )
