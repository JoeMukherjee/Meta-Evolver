"""Adaptive exploration: stagnation eviction and state-exhaustion fallback."""
from __future__ import annotations

from meta_evolver.adaptive.controller import (
    AdaptiveControllerConfig,
    AdaptiveExplorationController,
)
from meta_evolver.adaptive.tracker import EntityStateTracker, extract_entity
from meta_evolver.core.types import MemoryItem


def memory(**kwargs) -> MemoryItem:
    base = {"id": "m1", "title": "Search countertops", "lesson": "knives live on countertops"}
    return MemoryItem(**{**base, **kwargs})


# -- entity extraction -------------------------------------------------------


def test_extracts_entities_from_text_commands():
    assert extract_entity("go to fridge 1") == "fridge 1"
    assert extract_entity("open drawer 3") == "drawer 3"
    assert extract_entity("take knife 1 from countertop 2") == "countertop 2"
    assert extract_entity("put mug 1 in/on cabinet 1") == "cabinet 1"


def test_extracts_entities_from_tool_calls():
    assert extract_entity('inspect_service_logs({"service_name": "auth-service"})') == (
        "inspect_service_logs:auth-service"
    )
    # A mutating call is not a "visit" -- patching a service is not looking at it.
    assert extract_entity('patch_service_config({"service_name": "db-proxy"})') is None


def test_non_targeting_actions_extract_nothing():
    assert extract_entity("") is None
    assert extract_entity("think about the problem") is None


def test_tracker_reports_coverage():
    tracker = EntityStateTracker()
    tracker.record_candidates(["go to drawer 1", "go to drawer 2", "go to shelf 1"])
    tracker.record_action("go to drawer 1")
    assert tracker.unvisited() == ["drawer 2", "shelf 1"]
    assert abs(tracker.coverage() - 1 / 3) < 1e-9


def test_tracker_flags_repeat_visits():
    tracker = EntityStateTracker()
    for _ in range(3):
        tracker.record_action("go to fridge 1")
    assert tracker.revisited(threshold=2) == ["fridge 1"]


# -- eviction ----------------------------------------------------------------


def test_memory_is_evicted_after_patience_without_progress():
    ctrl = AdaptiveExplorationController(
        [memory()], AdaptiveControllerConfig(patience=3, novelty_counts_as_progress=False)
    )
    assert "countertops" in ctrl.memory_block()

    for _ in range(3):
        ctrl.record_step("examine countertop 1", "nothing useful", reward=0.0)

    assert ctrl.evicted
    assert ctrl.eviction_step == 3
    assert ctrl.memory_block() == ""


def test_reward_progress_resets_the_stagnation_counter():
    ctrl = AdaptiveExplorationController([memory()], AdaptiveControllerConfig(patience=3))
    ctrl.record_step("examine countertop 1", "nothing", reward=0.0)
    ctrl.record_step("examine countertop 2", "found it", reward=0.5)
    assert ctrl.stagnation == 0
    assert not ctrl.evicted


def test_novelty_counts_as_progress_in_sparse_reward_settings():
    """A reward-only detector never fires where reward is terminal-only.

    Visiting a new receptacle is progress even at zero reward; without this the
    controller would evict a strategy that is working, just slowly.
    """
    ctrl = AdaptiveExplorationController([memory()], AdaptiveControllerConfig(patience=2))
    for i in range(5):
        ctrl.record_step(f"go to drawer {i}", "empty", reward=0.0)
    assert not ctrl.evicted

    for _ in range(2):
        ctrl.record_step("go to drawer 4", "empty again", reward=0.0)
    assert ctrl.evicted


def test_eviction_is_one_way_within_an_episode():
    """Re-admitting a failed prior restarts the loop eviction exists to break."""
    ctrl = AdaptiveExplorationController(
        [memory()], AdaptiveControllerConfig(patience=2, novelty_counts_as_progress=False)
    )
    for _ in range(2):
        ctrl.record_step("examine countertop 1", "nothing", reward=0.0)
    assert ctrl.evicted

    ctrl.record_step("go to drawer 1", "found the knife", reward=1.0)
    assert ctrl.evicted
    assert ctrl.memory_block() == ""


# -- guidance ----------------------------------------------------------------


def test_guidance_names_visited_and_unvisited_after_eviction():
    ctrl = AdaptiveExplorationController(
        [memory()], AdaptiveControllerConfig(patience=2, novelty_counts_as_progress=False)
    )
    admissible = ["go to countertop 1", "go to drawer 1", "go to garbagecan 1"]
    for _ in range(2):
        ctrl.record_step("go to countertop 1", "nothing", admissible=admissible)

    guidance = ctrl.guidance_block(admissible)
    assert "EXHAUSTIVE" in guidance
    assert "countertop 1" in guidance
    assert "drawer 1" in guidance and "garbagecan 1" in guidance


def test_guidance_warns_before_eviction_too():
    """Half-way to eviction the agent is told it is stalling, but keeps the prior."""
    ctrl = AdaptiveExplorationController([memory()], AdaptiveControllerConfig(patience=6))
    # The first visit is novel, so it counts as progress; the next three do not.
    for _ in range(4):
        ctrl.record_step("examine countertop 1", "nothing", reward=0.0)
    assert ctrl.stagnation == 3

    guidance = ctrl.guidance_block(["go to countertop 1"])
    assert "No progress for" in guidance
    assert not ctrl.evicted
    assert ctrl.memory_block() != ""


def test_prioritize_demotes_visited_targets_without_removing_them():
    """A pruned action can be exactly the one the task needs.

    Returning to a container to *put* something in it looks like a revisit.
    Demoting is safe; deleting would make some tasks unsolvable.
    """
    ctrl = AdaptiveExplorationController(
        [memory()], AdaptiveControllerConfig(patience=1, novelty_counts_as_progress=False)
    )
    ctrl.record_step("go to countertop 1", "nothing")
    assert ctrl.evicted

    ordered = ctrl.prioritize(["go to countertop 1", "go to drawer 1"])
    assert ordered == ["go to drawer 1", "go to countertop 1"]
    assert len(ordered) == 2


def test_no_reordering_before_eviction():
    ctrl = AdaptiveExplorationController([memory()], AdaptiveControllerConfig(patience=99))
    ctrl.record_step("go to countertop 1", "nothing")
    commands = ["go to countertop 1", "go to drawer 1"]
    assert ctrl.prioritize(commands) == commands


def test_soft_prior_framing_is_present_by_default():
    ctrl = AdaptiveExplorationController([memory()])
    assert "may not fit this task" in ctrl.memory_block()


def test_snapshot_is_serializable():
    ctrl = AdaptiveExplorationController([memory()])
    ctrl.record_step("go to drawer 1", "empty", admissible=["go to drawer 1"])
    snap = ctrl.snapshot()
    assert snap.step_count == 1
    assert snap.visited == ["drawer 1"]
