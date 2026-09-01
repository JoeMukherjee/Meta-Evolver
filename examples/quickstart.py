"""Minimal 2-minute quickstart demo for Meta-Evolver."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_evolver import MetaEvolver, ReasoningMemoryBank


# 1. Initialize Memory Bank with strategies
bank = ReasoningMemoryBank([
    {
        "id": "strat-001",
        "title": "Kitchen Utensil Search Order",
        "task_pattern": "find and clean knife in kitchen",
        "strategy_rule": "Check countertop 1-3 first, then dishwasher 1, then drawers 1-3.",
    }
])

# 2. Instantiate MetaEvolver
evolver = MetaEvolver(memory_bank=bank)

# 3. Build Adaptive Controller for an embodied task
controller = evolver.build_controller_for_task("put a clean knife in drawer 1")
print("Retrieved Prompt:\n", controller.get_effective_memory_prompt())
