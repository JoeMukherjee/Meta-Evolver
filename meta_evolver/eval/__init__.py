"""Measurement that is not part of the learning loop.

The evolution graph scores candidates to *decide* things; this package scores
the whole system to *report* it. Keeping them apart is the same held-out
discipline the loop itself follows.
"""
from meta_evolver.eval.baseline import ARMS, run_arm, run_baseline, summarize

__all__ = ["ARMS", "run_arm", "run_baseline", "summarize"]
