"""Curriculum -- the environment side of the evolutionary loop.

A fixed benchmark stops teaching the moment it is solved. Every subsequent
generation runs the same tasks, produces the same successes, and the memory
bank and prompt optimizer have nothing left to learn from: the curve goes flat
and stays flat, and it is easy to mistake that plateau for convergence.

So difficulty is a variable, not a constant. ``level`` runs from 0 (the bare
benchmark) to 1 (every perturbation at full strength), and the harness stack
is *derived* from it: at 0.3 the agent meets occasional transient faults; at
0.6 it also faces a verification gate and distractor observations; at 0.85 the
step budget tightens as well. Each band adds a distinct failure mode rather
than turning one dial up, because an agent that has learned to retry has not
thereby learned to verify.

Promotion and demotion are hysteretic -- promote above 0.7, demote below 0.3 --
so a run does not oscillate across a single threshold, alternating between two
difficulties and learning neither.

The measured pass rate is always relative to the current level. A flat rate at
a rising level is improvement, and the evolution graph's stall detector knows
that.
"""
from __future__ import annotations

from dataclasses import dataclass

from meta_evolver.core.env import ActionableEnv
from meta_evolver.core.rules import (
    ActionBudget,
    IntermittentFault,
    ObservationNoise,
    VerificationGate,
)


@dataclass
class CurriculumBand:
    """One difficulty band and the perturbations it introduces."""

    threshold: float
    name: str
    fault_rate: float = 0.0
    noise_rate: float = 0.0
    verification_gate: bool = False
    budget: int | None = None


DEFAULT_BANDS: tuple[CurriculumBand, ...] = (
    CurriculumBand(0.00, "clean"),
    CurriculumBand(0.20, "flaky", fault_rate=0.25),
    CurriculumBand(0.45, "flaky+gated", fault_rate=0.3, verification_gate=True),
    CurriculumBand(
        0.65, "noisy+gated", fault_rate=0.3, noise_rate=0.35, verification_gate=True
    ),
    CurriculumBand(
        0.85,
        "adversarial",
        fault_rate=0.4,
        noise_rate=0.5,
        verification_gate=True,
        budget=12,
    ),
)


class Curriculum:
    """Maps a difficulty level onto a harness stack, and advances the level."""

    def __init__(
        self,
        bands: tuple[CurriculumBand, ...] = DEFAULT_BANDS,
        step: float = 0.2,
        max_level: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.bands = tuple(sorted(bands, key=lambda b: b.threshold))
        self.step = float(step)
        self.max_level = float(max_level)
        self.enabled = bool(enabled)

    # -- level -> environment ---------------------------------------------

    def band_for(self, level: float) -> CurriculumBand:
        active = self.bands[0]
        for band in self.bands:
            if level >= band.threshold:
                active = band
        return active

    def wrap(self, env: ActionableEnv, level: float, seed: int = 0) -> ActionableEnv:
        """Apply the harness stack for ``level``.

        Layers are applied innermost-first so the outermost -- the budget --
        sees the fully perturbed transition and can count it. Every stochastic
        layer gets a distinct derived seed: sharing one would correlate the
        fault draw with the noise draw, and a "hard" episode would be hard in
        every dimension at once rather than in the mix the level describes.
        """
        if not self.enabled or level <= 0:
            return env
        band = self.band_for(level)

        if band.fault_rate > 0:
            env = IntermittentFault(env, failure_rate=band.fault_rate, seed=seed + 101)
        if band.verification_gate:
            env = VerificationGate(env, seed=seed + 202)
        if band.noise_rate > 0:
            env = ObservationNoise(env, rate=band.noise_rate, seed=seed + 303)
        if band.budget:
            env = ActionBudget(env, budget=band.budget, seed=seed + 404)
        return env

    # -- level advancement -------------------------------------------------

    def adjust(
        self,
        level: float,
        pass_rate: float,
        promote_at: float = 0.7,
        demote_at: float = 0.3,
    ) -> float:
        """Next difficulty level given this generation's pass rate.

        The gap between ``promote_at`` and ``demote_at`` is deliberate
        hysteresis: with a single threshold, a run sitting near it flips band
        every generation and never accumulates enough experience at either to
        improve.
        """
        if not self.enabled:
            return level
        if pass_rate >= promote_at:
            return min(self.max_level, level + self.step)
        if pass_rate < demote_at:
            return max(0.0, level - self.step)
        return level

    def describe(self, level: float) -> str:
        band = self.band_for(level)
        parts = [f"level {level:.2f} ({band.name})"]
        if band.fault_rate:
            parts.append(f"faults {band.fault_rate:.0%}")
        if band.noise_rate:
            parts.append(f"noise {band.noise_rate:.0%}")
        if band.verification_gate:
            parts.append("verification gate")
        if band.budget:
            parts.append(f"budget {band.budget}")
        return ", ".join(parts)
