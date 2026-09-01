"""Deterministic seed derivation.

Every stochastic component here -- fault injection, observation noise, task
sampling, receptacle ordering -- must be reproducible from a run's seed, or an
A/B comparison between two generations measures RNG drift instead of learning.

The obvious spelling, ``random.Random((seed, episode))``, is wrong twice.
Python 3.11 removed support for arbitrary hashable seeds, so a tuple raises
``TypeError`` on any modern interpreter. And where a tuple *was* accepted, it
went through ``hash()``, which is randomized per process for strings -- so a
seed derived from a task id would differ between two runs of the same command.

``derive_seed`` hashes the parts through a stable digest instead: same inputs,
same stream, on any process and any platform.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

_MASK = 0xFFFFFFFF


def derive_seed(*parts: Any) -> int:
    """A stable 32-bit seed from any combination of values."""
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big") & _MASK


def derive_rng(*parts: Any) -> random.Random:
    """A ``random.Random`` seeded from ``parts``, reproducibly."""
    return random.Random(derive_seed(*parts))
