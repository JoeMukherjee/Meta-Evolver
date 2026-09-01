"""Meta-Evolver: Environment-Harness-based Evolutionary Meta-Agent."""
from meta_evolver.core.evolver import MetaEvolver, EvolutionConfig
from meta_evolver.core.types import MemoryItem, StepRecord, Trajectory
from meta_evolver.adaptive.controller import (
    AdaptiveExplorationController,
    AdaptiveControllerConfig,
)
from meta_evolver.memory.bank import ReasoningMemoryBank

__version__ = "0.1.0"
__all__ = [
    "MetaEvolver",
    "EvolutionConfig",
    "AdaptiveExplorationController",
    "AdaptiveControllerConfig",
    "ReasoningMemoryBank",
    "MemoryItem",
    "StepRecord",
    "Trajectory",
]

