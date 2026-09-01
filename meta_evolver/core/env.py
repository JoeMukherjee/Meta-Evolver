"""ActionableEnv -- the one environment interface the engine programs against.

Every benchmark implements this directly. A harness (see
``meta_evolver.core.rules``) *wraps* an ActionableEnv and is itself an
ActionableEnv, so the agent loop cannot tell a raw benchmark from one buried
under three layers of perturbation. That indistinguishability is the whole
point: the curriculum can make an environment arbitrarily harsher without the
agent, the graph, or the memory bank needing to know.

The interface is deliberately the Gymnasium five-tuple plus two things a
learning system needs and Gymnasium does not provide:

  * ``get_env_state()`` -- a serializable view harness hooks can read.
  * ``evaluate()``       -- ground truth, independent of accumulated reward.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from meta_evolver.core.types import (
    Action,
    EnvResetResponse,
    EnvResponse,
    EvaluationResult,
    Observation,
)


class ActionableEnv(ABC):
    """The universal environment contract."""

    #: OpenAI-style tool schemas this env accepts. Scoping tools per-env
    #: (rather than handing the agent one global registry) keeps the action
    #: space honest: an agent offered 18 tools for a 7-tool task wastes steps
    #: rediscovering which ones do anything.
    tool_schemas: ClassVar[list[dict[str, Any]]] = []

    env_type: ClassVar[str] = "actionable_env"

    # -- step loop ---------------------------------------------------------

    @abstractmethod
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> EnvResetResponse:
        """Start a new episode. ``options["task_id"]`` selects a task."""

    @abstractmethod
    def step(self, action: Action) -> EnvResponse: ...

    @abstractmethod
    def observe(self) -> Observation:
        """Re-read current state without stepping.

        Distinct from ``reset`` because a harness may mutate state between the
        reset returning and the agent's first action.
        """

    @abstractmethod
    def evaluate(self) -> EvaluationResult:
        """Ground-truth verdict for the episode so far."""

    def get_env_state(self) -> dict[str, Any]:
        """Serializable view of internals that harness hooks may read.

        Must not contain live handles (sockets, browsers, containers).
        """
        return {}

    # -- optional -----------------------------------------------------------

    def available_tools(self) -> list[dict[str, Any]]:
        """Tool schemas for *this instance*. Overridable when the action space
        varies per task; defaults to the class-level registry."""
        return list(self.tool_schemas)

    def close(self) -> None:
        """Release runtime resources. Subprocess death does not free a
        container or a browser, so envs owning those must clean up here."""
        return None


class EnvHarness(ActionableEnv):
    """An ActionableEnv that wraps another one (the decorator pattern).

    Everything not overridden delegates inward, so a subclass overriding a
    single hook is a complete, valid environment.
    """

    env_type: ClassVar[str] = "harness"

    def __init__(self, inner: ActionableEnv) -> None:
        self.inner = inner

    # Delegation -----------------------------------------------------------

    def reset(self, seed=None, options=None) -> EnvResetResponse:
        return self.inner.reset(seed=seed, options=options)

    def step(self, action: Action) -> EnvResponse:
        return self.inner.step(action)

    def observe(self) -> Observation:
        return self.inner.observe()

    def evaluate(self) -> EvaluationResult:
        return self.inner.evaluate()

    def get_env_state(self) -> dict[str, Any]:
        return self.inner.get_env_state()

    def available_tools(self) -> list[dict[str, Any]]:
        return self.inner.available_tools()

    def close(self) -> None:
        self.inner.close()

    # Introspection --------------------------------------------------------

    @property
    def base(self) -> ActionableEnv:
        """The innermost non-harness environment."""
        cur: ActionableEnv = self
        while isinstance(cur, EnvHarness):
            cur = cur.inner
        return cur

    def layers(self) -> list[str]:
        """Names of the harness stack, outermost first."""
        out: list[str] = []
        cur: ActionableEnv = self
        while isinstance(cur, EnvHarness):
            out.append(type(cur).__name__)
            cur = cur.inner
        out.append(type(cur).__name__)
        return out
