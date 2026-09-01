"""MetaEvolver: Autonomous evolutionary loop for agent strategies & memories."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.adaptive.controller import AdaptiveExplorationController, AdaptiveControllerConfig

@dataclass
class EvolutionConfig:
    generations: int = 3
    population_size: int = 10
    mutation_rate: float = 0.3
    patience: int = 6
    retrieval_k: int = 5
    retrieval_mode: str = "mmr"

class MetaEvolver:
    """Self-improving agent evolution engine across multi-shard environment harnesses."""
    def __init__(
        self,
        memory_bank: Optional[ReasoningMemoryBank] = None,
        config: Optional[EvolutionConfig] = None,
    ) -> None:
        self.bank = memory_bank or ReasoningMemoryBank()
        self.cfg = config or EvolutionConfig()

    def build_controller_for_task(self, task_description: str) -> AdaptiveExplorationController:
        retrieved = self.bank.retrieve(
            query=task_description,
            top_k=self.cfg.retrieval_k,
            mode=self.cfg.retrieval_mode,
        )
        ctrl_cfg = AdaptiveControllerConfig(patience=self.cfg.patience)
        return AdaptiveExplorationController(retrieved_memories=retrieved, config=ctrl_cfg)
