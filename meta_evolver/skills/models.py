"""The :class:`Skill` record.

A *memory* (``MemoryItem``) is a lesson: "when X, beware Y". A *skill* is a
reusable procedure the agent can run: an ordered sequence of abstract actions
that has, in fact, been seen to work. The distinction matters for lifecycle:
a skill has preconditions, a version, a lineage, and a status it must *earn* --
``candidate`` on creation, ``active`` only after it has won on more than one
distinct task, ``deprecated`` once the evidence says it does not hold up.

Evidence uses the same Beta(1,1) posterior as the memory bank, so a skill and
a memory are scored on one scale. :meth:`Skill.lcb` / :meth:`Skill.ucb` are
credible-interval bounds (normal approximation to the Beta): promotion keys on
the lower bound ("we are confident it works") and deprecation on the upper
("we are confident it does not"), which is what stops a lucky first trial
from promoting a skill or an unlucky one from killing it.
"""
from __future__ import annotations

import hashlib
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from meta_evolver.core.types import MemoryItem

Status = Literal["candidate", "active", "deprecated"]


class Skill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = ""
    name: str
    description: str = ""
    when_to_use: list[str] = Field(default_factory=list)
    """Concept words that signal this skill applies; matched against a new
    task's instruction at retrieval time."""
    steps: list[str] = Field(default_factory=list)
    """The procedure, generalised: arguments are placeholders, not the
    specific objects of the episode it was distilled from."""
    tools: list[str] = Field(default_factory=list)
    signature: str = ""
    """The verb sequence (``go>take>heat>put``). Identity for dedup."""
    status: Status = "candidate"
    version: int = 1
    parent_id: str | None = None
    superseded_by: str | None = None
    """Set when a promoted child version replaces this skill. Retiring a skill
    always leaves a pointer to what replaced it, so nothing is a dead end."""
    merged_into: str | None = None
    locked: bool = False
    """A locked skill is never changed by an autonomous actor."""
    patch_count: int = 0
    patched_at_uses: int = 0
    """``uses`` when the last patch landed: ``reused_since_patch`` is then
    'did anyone run the patched version', the signal that tells a patch that
    was exercised from one that was merely written."""
    benchmark: str = ""
    origin_task_ids: list[str] = Field(default_factory=list)
    uses: int = 0
    wins: int = 0
    win_tasks: list[str] = Field(default_factory=list)
    """Distinct tasks this skill has won on (capped). Promotion requires more
    than one: a procedure that worked on a single task has not shown it
    generalises, only that it memorised."""
    domain_stats: dict[str, list[float]] = Field(default_factory=dict)
    created_generation: int = 0
    last_used_generation: int = 0
    promoted_generation: int | None = None
    embedding: list[float] | None = None

    def model_post_init(self, __context) -> None:
        if not self.id:
            self.id = "sk-" + hashlib.sha1(
                f"{self.benchmark}|{self.signature}|{self.version}".encode()
            ).hexdigest()[:10]

    # -- evidence ------------------------------------------------------------

    @property
    def utility(self) -> float:
        return (self.wins + 1.0) / (self.uses + 2.0)

    def _beta_sd(self) -> float:
        a, b = self.wins + 1.0, (self.uses - self.wins) + 1.0
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))

    @property
    def reused_since_patch(self) -> bool:
        return self.patch_count > 0 and self.uses > self.patched_at_uses

    def lcb(self, z: float = 1.0) -> float:
        return max(0.0, self.utility - z * self._beta_sd())

    def ucb(self, z: float = 1.0) -> float:
        return min(1.0, self.utility + z * self._beta_sd())

    def record(self, success: bool, task_id: str = "", domain: str | None = None,
               generation: int = 0, gamma: float = 1.0, cap: int = 50) -> None:
        self.uses += 1
        self.last_used_generation = generation
        if success:
            self.wins += 1
            if task_id and task_id not in self.win_tasks and len(self.win_tasks) < cap:
                self.win_tasks.append(task_id)
        if domain is not None:
            w, u = self.domain_stats.get(domain, [0.0, 0.0])
            w, u = w * gamma, u * gamma + 1.0
            self.domain_stats[domain] = [w + (1.0 if success else 0.0), u]

    # -- views ---------------------------------------------------------------

    def embed_text(self) -> str:
        return " ".join([self.name, self.description, " ".join(self.when_to_use), " ".join(self.tools)])

    def as_memory_item(self) -> MemoryItem:
        """The skill as the prompt layer already knows how to render and evict
        a memory -- so a skill gets the same soft-prior framing and the same
        stagnation-eviction safety net, with no change to the controller."""
        return MemoryItem(
            id=self.id,
            title=self.name,
            scenario="skill",
            lesson=self.description,
            procedure=" -> ".join(self.steps),
            triggers=list(self.when_to_use),
            polarity="success",
            benchmark=self.benchmark,
            uses=self.uses,
            wins=self.wins,
        )

    def to_markdown(self) -> str:
        """``SKILL.md`` in the Agent Skills layout (name/description frontmatter)."""
        trig = ", ".join(self.when_to_use) or "the task matches the steps below"
        # Trigger-first: the first words are what a retriever, or a human
        # skimming an index, actually reads.
        desc = f"Use when the task involves: {trig}. {self.description.rstrip('.')}.".replace("\n", " ")[:240]
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(self.steps, 1))
        return (
            f"---\nname: {self.name}\ndescription: {desc}\n---\n\n"
            f"# {self.name}\n\n## When to use\n\n{trig}\n\n## Steps\n\n{steps}\n\n"
            f"## Evidence\n\n- status: {self.status} (v{self.version})\n"
            f"- won {self.wins} of {self.uses} uses across {len(self.win_tasks)} distinct task(s)\n"
            f"- signature: `{self.signature}`\n"
        )
