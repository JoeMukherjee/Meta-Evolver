"""Adaptive Exploration Controller: Solves OOD confirmation bias & memory stagnation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from meta_evolver.adaptive.tracker import EntityStateTracker

@dataclass
class AdaptiveControllerConfig:
    patience: int = 6
    soft_prior: bool = True
    state_exhaustion: bool = True
    evict_on_stagnation: bool = True

class AdaptiveExplorationController:
    """Dynamically balances memory prior exploitation and BFS state exhaustion."""
    def __init__(
        self,
        retrieved_memories: List[Dict[str, Any]],
        config: Optional[AdaptiveControllerConfig] = None,
    ) -> None:
        self.raw_memories = retrieved_memories
        self.cfg = config or AdaptiveControllerConfig()
        self.tracker = EntityStateTracker()
        
        self.stagnation_counter: int = 0
        self.last_reward: float = 0.0
        self.memory_evicted: bool = False
        self.eviction_step: Optional[int] = None
        self.step_history: List[Dict[str, Any]] = []

    def record_step(
        self,
        action: str,
        observation: str,
        reward: float,
        admissible_commands: Optional[List[str]] = None,
    ) -> None:
        self.tracker.update_from_action(action)
        if admissible_commands:
            self.tracker.update_from_admissible(admissible_commands)

        if reward > self.last_reward:
            self.stagnation_counter = 0
            self.last_reward = reward
        else:
            self.stagnation_counter += 1

        if (
            self.cfg.evict_on_stagnation
            and not self.memory_evicted
            and self.stagnation_counter >= self.cfg.patience
        ):
            self.memory_evicted = True
            self.eviction_step = len(self.step_history)

        self.step_history.append({
            "action": action,
            "observation": observation,
            "reward": reward,
            "stagnation": self.stagnation_counter,
            "evicted": self.memory_evicted,
        })

    def get_effective_memory_prompt(self) -> str:
        if self.memory_evicted or not self.raw_memories:
            return ""
        
        lines = ["[RETRIEVED STRATEGY HINTS (Flexible Priors)]:"]
        for i, mem in enumerate(self.raw_memories, 1):
            title = mem.get("title", f"Strategy #{i}")
            rule = mem.get("rule") or mem.get("strategy_rule") or ""
            lines.append(f"{i}. {title}: {rule}")
        return "\n".join(lines)

    def get_exploration_guidance(self, admissible_commands: List[str]) -> str:
        if not self.cfg.state_exhaustion:
            return ""
        
        unvisited = self.tracker.get_unvisited_entities()
        visited = sorted(list(self.tracker.visited_entities))
        
        if not self.memory_evicted:
            return ""
        
        guidance = [
            "\n[SYSTEM EXPLORATION INVARIANT]: Prior search memory has stagnated. Do not repeat visited locations.",
            f"Already Explored: {visited or 'None'}",
            f"Unexplored Candidates: {unvisited or 'All visited'}",
            "Prioritize exploring unvisited candidates in breadth-first order."
        ]
        return "\n".join(guidance)

    def filter_redundant_actions(self, admissible_commands: List[str]) -> List[str]:
        if not self.memory_evicted:
            return admissible_commands
        
        unvisited = set(self.tracker.get_unvisited_entities())
        priority_actions = []
        regular_actions = []
        
        for cmd in admissible_commands:
            is_priority = any(u in cmd.lower() for u in unvisited)
            if is_priority:
                priority_actions.append(cmd)
            else:
                regular_actions.append(cmd)
                
        return priority_actions + regular_actions if priority_actions else admissible_commands
