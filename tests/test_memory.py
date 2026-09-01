"""Unit tests for ReasoningMemoryBank."""
import unittest
import sys
from pathlib import Path

# Add package root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_evolver.memory.bank import ReasoningMemoryBank

class TestReasoningMemoryBank(unittest.TestCase):
    def test_cosine_and_mmr_retrieval(self):
        items = [
            {"id": "1", "title": "Strat 1", "embedding": [1.0, 0.0, 0.0]},
            {"id": "2", "title": "Strat 2", "embedding": [0.9, 0.1, 0.0]},
            {"id": "3", "title": "Strat 3", "embedding": [0.6, 0.8, 0.0]},
        ]
        bank = ReasoningMemoryBank(items)
        
        # Plain query
        results = bank.retrieve("find knife", top_k=2)
        self.assertEqual(len(results), 2)
        
        # Cosine mode (picks top 2 similar: 1 and 2)
        cos_results = bank.retrieve("find knife", query_embedding=[1.0, 0.0, 0.0], top_k=2, mode="cosine")
        self.assertEqual(cos_results[0]["id"], "1")
        self.assertEqual(cos_results[1]["id"], "2")
        
        # MMR mode with diversity penalty (picks 1, then penalizes redundant 2 and selects diverse 3)
        mmr_results = bank.retrieve("find knife", query_embedding=[1.0, 0.0, 0.0], top_k=2, mode="mmr", mmr_lambda=0.3)
        self.assertEqual(mmr_results[0]["id"], "1")
        self.assertEqual(mmr_results[1]["id"], "3")


if __name__ == "__main__":
    unittest.main()

