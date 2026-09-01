"""Adaptive exploration: stagnation eviction and state-exhaustion fallback."""
from meta_evolver.adaptive.controller import (
    AdaptiveControllerConfig,
    AdaptiveControllerState,
    AdaptiveExplorationController,
)
from meta_evolver.adaptive.tracker import EntityStateTracker, extract_entity

__all__ = [
    "AdaptiveControllerConfig",
    "AdaptiveControllerState",
    "AdaptiveExplorationController",
    "EntityStateTracker",
    "extract_entity",
]
