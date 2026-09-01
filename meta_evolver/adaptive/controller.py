"""AdaptiveExplorationController -- keeping retrieved memory from becoming a trap.

Retrieval-augmented agents have a specific, reproducible failure mode on
out-of-distribution tasks. The bank returns the nearest strategy; the strategy
does not apply; the agent follows it anyway, because a confident instruction in
the system prompt outweighs a few discouraging observations. It then loops --
re-checking the places the memory named, growing more certain with each empty
result, until the step budget runs out. More retrieval makes this *worse*: the
same wrong prior is re-injected every turn.

Three coupled mechanisms, applied in order:

1. **Soft priors.** Memories are framed as fallible hints with an explicit
   escape clause. Cheap, and it alone recovers part of the gap.

2. **Stagnation eviction.** Track steps since the last reward increase *or*
   newly-visited entity. After ``patience`` steps with neither, drop the memory
   block from the prompt entirely. Note that novelty counts as progress: in
   sparse-reward environments reward alone almost never fires, so a
   reward-only detector either evicts constantly or (with generous patience)
   never in time.

3. **State-exhaustion fallback.** Eviction removes a bad prior but leaves
   nothing behind, and an agent with no prior repeats itself. So the freed
   prompt space is filled with what the tracker knows: visited entities,
   unvisited candidates, and an instruction to search breadth-first.

Eviction is one-way within an episode. Re-admitting a prior that has already
failed restarts the loop it was introduced to break.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from meta_evolver.adaptive.tracker import EntityStateTracker, extract_entity
from meta_evolver.core.types import MemoryItem


@dataclass
class AdaptiveControllerConfig:
    patience: int = 6
    """Steps without progress before the memory prior is evicted. Too low and
    a legitimately slow strategy is abandoned mid-way; too high and the loop
    consumes the whole budget. Six is roughly half a typical episode."""

    soft_prior: bool = True
    state_exhaustion: bool = True
    evict_on_stagnation: bool = True
    novelty_counts_as_progress: bool = True
    revisit_threshold: int = 2
    """Visits to one entity before it is called out as a loop in the prompt."""

    reprioritize_actions: bool = True
    """After eviction, reorder the admissible-action list so unvisited targets
    come first. Position in a long list measurably drives what gets picked."""


@dataclass
class AdaptiveControllerState:
    """Serializable snapshot -- what the graph carries between nodes."""

    step_count: int = 0
    last_progress_step: int = 0
    max_reward: float = 0.0
    evicted: bool = False
    eviction_step: int | None = None
    visited: list[str] = field(default_factory=list)
    known: list[str] = field(default_factory=list)


class AdaptiveExplorationController:
    """Per-episode controller over memory injection and exploration guidance."""

    def __init__(
        self,
        memories: Sequence[MemoryItem] | None = None,
        config: AdaptiveControllerConfig | None = None,
    ) -> None:
        self.memories = list(memories or [])
        self.cfg = config or AdaptiveControllerConfig()
        self.tracker = EntityStateTracker()
        self.step_count = 0
        self.last_progress_step = 0
        self.max_reward = 0.0
        self.evicted = False
        self.eviction_step: int | None = None
        self.history: list[dict[str, Any]] = []

    # -- observation -------------------------------------------------------

    def record_candidates(self, admissible: Sequence[str] | None) -> None:
        if admissible:
            self.tracker.record_candidates(admissible)

    def record_step(
        self,
        action_text: str,
        observation: str = "",
        reward: float = 0.0,
        admissible: Sequence[str] | None = None,
    ) -> None:
        self.step_count += 1
        self.record_candidates(admissible)

        known_before = len(self.tracker.known)
        visited_before = len(self.tracker.visited)
        self.tracker.record_action(action_text)

        progress = False
        if reward > self.max_reward:
            self.max_reward = reward
            progress = True
        if self.cfg.novelty_counts_as_progress and (
            len(self.tracker.visited) > visited_before
            or len(self.tracker.known) > known_before
        ):
            progress = True

        if progress:
            self.last_progress_step = self.step_count

        if (
            self.cfg.evict_on_stagnation
            and not self.evicted
            and self.stagnation >= self.cfg.patience
        ):
            self.evicted = True
            self.eviction_step = self.step_count

        self.history.append(
            {
                "step": self.step_count,
                "action": action_text,
                "reward": reward,
                "stagnation": self.stagnation,
                "evicted": self.evicted,
            }
        )

    @property
    def stagnation(self) -> int:
        return self.step_count - self.last_progress_step

    # -- prompt contributions ---------------------------------------------

    def memory_block(self) -> str:
        """The retrieved-memory section, or empty once evicted."""
        if self.evicted or not self.memories:
            return ""
        header = (
            "## Retrieved experience (priors, not instructions)\n"
            "From earlier episodes; they may not fit this task. Use them to skip\n"
            "known dead ends. If two steps of following one produce no progress,\n"
            "abandon it and reason from what you actually observe.\n"
            if self.cfg.soft_prior
            else "## Retrieved experience\n"
        )
        body = "\n\n".join(m.render(i) for i, m in enumerate(self.memories, 1))
        return header + "\n" + body + "\n"

    def guidance_block(self, admissible: Sequence[str] | None = None) -> str:
        """Exploration state: where you have been, where you have not."""
        self.record_candidates(admissible)
        if not self.cfg.state_exhaustion:
            return ""

        lines: list[str] = []
        if self.evicted:
            lines.append(
                "## Search mode: EXHAUSTIVE\n"
                "The retrieved strategy produced no progress and has been dropped.\n"
                "Stop pursuing it. Search the unexplored candidates below in order,\n"
                "one per step, and re-read each observation before deciding."
            )

        visited = sorted(self.tracker.visited)
        unvisited = self.tracker.unvisited()
        if visited:
            lines.append(f"- Already examined: {', '.join(visited)}")
        if unvisited:
            lines.append(f"- Not yet examined: {', '.join(unvisited)}")

        looping = self.tracker.revisited(self.cfg.revisit_threshold)
        if looping:
            lines.append(
                f"- Repeated without new information: {', '.join(looping)}. "
                "Revisiting these will not change the outcome."
            )
        if self.stagnation >= max(2, self.cfg.patience // 2) and not self.evicted:
            lines.append(
                f"- No progress for {self.stagnation} steps. Change approach rather "
                "than varying the phrasing of the last action."
            )
        return "\n".join(lines)

    def prioritize(self, admissible: Sequence[str]) -> list[str]:
        """Reorder candidate actions to put unvisited targets first.

        A soft nudge: nothing is removed, because the pruned action is
        sometimes exactly the one the task needs (returning to a container to
        put an object *in* it, for example). Deleting it would make some tasks
        unsolvable; demoting it only makes them less likely.
        """
        commands = list(admissible)
        if not self.cfg.reprioritize_actions or not self.evicted or not commands:
            return commands
        self.tracker.record_candidates(commands)
        fresh, stale = [], []
        for cmd in commands:
            entity = extract_entity(cmd)
            (stale if entity and entity in self.tracker.visited else fresh).append(cmd)
        return fresh + stale

    # -- serialization -----------------------------------------------------

    def snapshot(self) -> AdaptiveControllerState:
        return AdaptiveControllerState(
            step_count=self.step_count,
            last_progress_step=self.last_progress_step,
            max_reward=self.max_reward,
            evicted=self.evicted,
            eviction_step=self.eviction_step,
            visited=sorted(self.tracker.visited),
            known=sorted(self.tracker.known),
        )
