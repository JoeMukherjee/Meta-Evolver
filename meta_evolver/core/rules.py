"""Rules -- the environment-mutation harness.

A ``Rules`` layer wraps an environment and applies up to three pure hooks per
step:

    filter_action(action, env_state)                -- the A axis
    modify_transition(action, response, env_state)  -- the T axis
    filter_observation(obs, env_state)              -- the O axis

All three default to pass-through, so a useful harness is a subclass that
overrides one of them. Composing effects means stacking layers; the stack is
ordinary object composition, and ``EnvHarness.layers()`` reports it.

Determinism matters here. A harness that injects a random fault must draw from
its own seeded RNG rather than the global one, or two runs of the same
generation are not comparable and every measured improvement is noise. Every
stochastic hook in this module takes a seed.
"""
from __future__ import annotations

from typing import Any

from meta_evolver.core.env import ActionableEnv, EnvHarness
from meta_evolver.core.seeding import derive_rng
from meta_evolver.core.types import (
    Action,
    Blocked,
    EnvResetResponse,
    EnvResponse,
    Observation,
)


class Rules(EnvHarness):
    """A/T/O harness. Subclass and override the hooks you need."""

    env_type = "rules"

    def __init__(self, inner: ActionableEnv, seed: int = 0) -> None:
        super().__init__(inner)
        self.seed = int(seed)
        self.rng = derive_rng(self.seed)
        self.n_blocked = 0
        self.n_perturbed = 0

    # -- hooks (override these) --------------------------------------------

    def filter_action(self, action: Action, env_state: dict[str, Any]) -> Action | Blocked:
        return action

    def modify_transition(
        self, action: Action, response: EnvResponse, env_state: dict[str, Any]
    ) -> EnvResponse:
        return response

    def filter_observation(self, obs: Observation, env_state: dict[str, Any]) -> Observation:
        return obs

    # -- plumbing (usually leave alone) ------------------------------------

    def reset(self, seed=None, options=None) -> EnvResetResponse:
        # Re-seed per episode so a given (harness seed, episode seed) pair
        # reproduces exactly, which is what makes A/B comparisons meaningful.
        self.rng = derive_rng(self.seed, seed)
        self.n_blocked = 0
        self.n_perturbed = 0
        resp = self.inner.reset(seed=seed, options=options)
        return EnvResetResponse(
            observation=self.filter_observation(resp.observation, self.get_env_state()),
            info=resp.info,
        )

    def step(self, action: Action) -> EnvResponse:
        state = self.get_env_state()
        filtered = self.filter_action(action, state)

        if isinstance(filtered, Blocked):
            self.n_blocked += 1
            obs = self.filter_observation(self.inner.observe(), state)
            return EnvResponse(
                observation=Observation(
                    text=f"{obs.text}\n[HARNESS] Action rejected: {filtered.reason}",
                    data=obs.data,
                ),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "result": {"error": filtered.reason},
                    "blocked": True,
                    "harness": type(self).__name__,
                },
            )

        resp = self.inner.step(filtered)
        state = self.get_env_state()
        resp = self.modify_transition(filtered, resp, state)
        return EnvResponse(
            observation=self.filter_observation(resp.observation, state),
            reward=resp.reward,
            terminated=resp.terminated,
            truncated=resp.truncated,
            info=resp.info,
        )

    def observe(self) -> Observation:
        return self.filter_observation(self.inner.observe(), self.get_env_state())


# ---------------------------------------------------------------------------
# Concrete, benchmark-agnostic perturbations
# ---------------------------------------------------------------------------


class IntermittentFault(Rules):
    """Fails a fraction of read-like actions with a transient error.

    Trains retry and fallback behaviour: an agent that treats the first
    timeout as ground truth will conclude the wrong thing about the world.
    ``read_actions=None`` perturbs every action.
    """

    def __init__(
        self,
        inner: ActionableEnv,
        failure_rate: float = 0.25,
        read_actions: tuple[str, ...] | None = None,
        max_faults: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__(inner, seed=seed)
        self.failure_rate = float(failure_rate)
        self.read_actions = read_actions
        self.max_faults = int(max_faults)

    def modify_transition(self, action, response, env_state):
        if self.n_perturbed >= self.max_faults:
            return response
        if self.read_actions is not None and action.name not in self.read_actions:
            return response
        if self.rng.random() >= self.failure_rate:
            return response

        self.n_perturbed += 1
        msg = (
            "ConnectionTimeoutError: ETIMEDOUT. The read did not complete; its "
            "result is unknown, not empty. Retry or reach the same fact another way."
        )
        return EnvResponse(
            observation=Observation(
                text=f"{response.observation.text}\n[SYSTEM] {action.render()} timed out.",
                data=response.observation.data,
            ),
            reward=0.0,
            terminated=False,
            truncated=response.truncated,
            info={
                "result": {"error": msg},
                "perturbed": True,
                "harness": type(self).__name__,
            },
        )


class VerificationGate(Rules):
    """Blocks a submit-style action until the environment reports verification.

    Enforces the discipline that separates a plausible fix from a confirmed
    one. The gate reads a boolean out of ``get_env_state()`` rather than
    naming any specific benchmark, so it transfers.
    """

    def __init__(
        self,
        inner: ActionableEnv,
        submit_actions: tuple[str, ...] = ("submit_resolution", "submit", "finish"),
        state_flag: str = "verified",
        seed: int = 0,
    ) -> None:
        super().__init__(inner, seed=seed)
        self.submit_actions = tuple(submit_actions)
        self.state_flag = state_flag

    def filter_action(self, action, env_state):
        if action.name in self.submit_actions and not env_state.get(self.state_flag, False):
            return Blocked(
                reason=(
                    f"Submission rejected: '{self.state_flag}' is still false. "
                    "Verify the fix end to end before submitting."
                )
            )
        return action


class ObservationNoise(Rules):
    """Injects irrelevant-but-plausible lines into observations.

    Tests whether the agent's attention survives distractors -- the failure
    mode where an agent chases a red herring it was handed for free.
    """

    DISTRACTORS = (
        "[INFO] Routine background compaction finished in 812ms.",
        "[WARN] Deprecated field 'legacy_mode' present in config; ignored.",
        "[INFO] Heartbeat OK from 3 peers.",
        "[DEBUG] Trace sampling rate adjusted to 0.05.",
    )

    def __init__(self, inner: ActionableEnv, rate: float = 0.4, seed: int = 0) -> None:
        super().__init__(inner, seed=seed)
        self.rate = float(rate)

    def filter_observation(self, obs, env_state):
        if self.rate <= 0 or self.rng.random() >= self.rate:
            return obs
        self.n_perturbed += 1
        line = self.rng.choice(self.DISTRACTORS)
        return Observation(text=f"{obs.text}\n{line}", data=obs.data)


class ActionBudget(Rules):
    """Truncates the episode after N steps, announcing the remaining budget.

    Rewards planning over exhaustive search, and gives the memory bank a
    reason to prefer short procedures.
    """

    def __init__(self, inner: ActionableEnv, budget: int = 12, seed: int = 0) -> None:
        super().__init__(inner, seed=seed)
        self.budget = int(budget)
        self.used = 0

    def reset(self, seed=None, options=None):
        self.used = 0
        return super().reset(seed=seed, options=options)

    def modify_transition(self, action, response, env_state):
        self.used += 1
        remaining = self.budget - self.used
        if remaining <= 0:
            return EnvResponse(
                observation=response.observation,
                reward=response.reward,
                terminated=response.terminated,
                truncated=True,
                info={**response.info, "budget_exhausted": True},
            )
        if remaining <= 3:
            return EnvResponse(
                observation=Observation(
                    text=f"{response.observation.text}\n[BUDGET] {remaining} actions remaining.",
                    data=response.observation.data,
                ),
                reward=response.reward,
                terminated=response.terminated,
                truncated=response.truncated,
                info=response.info,
            )
        return response
