"""Unit tests for adaptive exploration controller."""
import unittest
import sys
from pathlib import Path

# Add package root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_evolver.adaptive.controller import (
    AdaptiveExplorationController,
    AdaptiveControllerConfig,
)

class TestAdaptiveExploration(unittest.TestCase):
    def test_adaptive_eviction(self):
        memories = [{"title": "Search Fridge", "rule": "Always look inside fridge"}]
        ctrl = AdaptiveExplorationController(memories, AdaptiveControllerConfig(patience=3))
        
        self.assertFalse(ctrl.memory_evicted)
        self.assertIn("Search Fridge", ctrl.get_effective_memory_prompt())
        
        ctrl.record_step("go to fridge 1", "fridge 1 is empty", 0.0, ["go to cabinet 1"])
        ctrl.record_step("open fridge 1", "nothing inside", 0.0, ["go to cabinet 1"])
        ctrl.record_step("examine fridge 1", "still nothing", 0.0, ["go to cabinet 1"])
        
        self.assertTrue(ctrl.memory_evicted)
        self.assertEqual(ctrl.get_effective_memory_prompt(), "")
        guidance = ctrl.get_exploration_guidance(["go to cabinet 1"])
        self.assertIn("Unexplored Candidates", guidance)

    def test_progress_resets_stagnation(self):
        memories = [{"title": "Strat", "rule": "Rule"}]
        ctrl = AdaptiveExplorationController(memories, AdaptiveControllerConfig(patience=3))
        
        ctrl.record_step("go to fridge 1", "ok", 0.0)
        ctrl.record_step("open fridge 1", "found target", 0.5)
        self.assertEqual(ctrl.stagnation_counter, 0)
        self.assertFalse(ctrl.memory_evicted)

if __name__ == "__main__":
    unittest.main()

