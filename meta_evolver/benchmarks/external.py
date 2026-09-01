"""Bridges to environments Meta-Evolver did not define.

Two of them, covering most of what exists in the wild:

``ExternalEnvAdapter``
    Wraps anything with a Gymnasium-shaped ``reset``/``step`` -- including
    EnvHarness ``ActionableEnv`` implementations, whose types are structurally
    identical to this package's but come from a different module and so fail
    an ``isinstance`` check. Duck typing rather than inheritance is what makes
    that work without importing a dependency that may not be installed.

``TextEnvAdapter``
    For environments whose action space is a string and whose observation is
    a string -- text games, REPL-ish tools, terminal harnesses. It supplies the
    single ``do(text=...)`` tool so a text environment reaches the same
    tool-calling agent loop as everything else, with no special-casing in the
    graph.

Both exist so that "adapt to any benchmark" is a thing you do in ten lines,
not a claim in a README.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from meta_evolver.benchmarks.base import BenchmarkAdapter, tool_schema
from meta_evolver.core.env import ActionableEnv
from meta_evolver.core.types import (
    Action,
    EnvResetResponse,
    EnvResponse,
    EvaluationResult,
    Observation,
    TaskSpec,
)


def _coerce_observation(raw: Any) -> Observation:
    """Normalize whatever an environment returns into an ``Observation``."""
    if isinstance(raw, Observation):
        return raw
    if hasattr(raw, "text"):  # a structurally-compatible foreign Observation
        return Observation(text=str(raw.text or ""), data=dict(getattr(raw, "data", {}) or {}))
    if isinstance(raw, dict):
        text = raw.get("text") or raw.get("observation") or ""
        return Observation(text=str(text), data=raw)
    return Observation(text=str(raw))


def _coerce_step(raw: Any) -> EnvResponse:
    """Accept either an ``EnvResponse``-shaped object or a Gym 4/5-tuple."""
    if isinstance(raw, EnvResponse):
        return raw
    if hasattr(raw, "observation") and hasattr(raw, "reward"):
        return EnvResponse(
            observation=_coerce_observation(raw.observation),
            reward=float(getattr(raw, "reward", 0.0) or 0.0),
            terminated=bool(getattr(raw, "terminated", False)),
            truncated=bool(getattr(raw, "truncated", False)),
            info=dict(getattr(raw, "info", {}) or {}),
        )
    if isinstance(raw, tuple):
        if len(raw) == 5:
            obs, reward, terminated, truncated, info = raw
        elif len(raw) == 4:
            obs, reward, terminated, info = raw
            truncated = False
        else:  # pragma: no cover - unusual shape
            raise TypeError(f"cannot interpret a {len(raw)}-tuple as a step result")
        return EnvResponse(
            observation=_coerce_observation(obs),
            reward=float(reward or 0.0),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}),
        )
    raise TypeError(f"cannot interpret {type(raw).__name__} as a step result")


class ExternalEnvAdapter(ActionableEnv):
    """Presents a foreign environment through this package's interface."""

    env_type = "external"

    def __init__(
        self,
        env: Any,
        tool_schemas: Sequence[dict[str, Any]] | None = None,
        success_key: str = "won",
        action_builder: Callable[[Action], Any] | None = None,
    ) -> None:
        self.env = env
        self.success_key = success_key
        self.action_builder = action_builder
        self._last_info: dict[str, Any] = {}
        self._last_reward = 0.0

        if tool_schemas is not None:
            self.tool_schemas = list(tool_schemas)
        else:
            found = getattr(env, "tool_schemas", None)
            if callable(found):
                found = found()
            if not found:
                registry = getattr(env, "tool_registry", None) or []
                found = [
                    t.get_info() if hasattr(t, "get_info") else t for t in registry
                ]
            self.tool_schemas = [_normalize_schema(s) for s in (found or [])]

    def reset(self, seed=None, options=None) -> EnvResetResponse:
        self._last_info = {}
        self._last_reward = 0.0
        raw = self.env.reset(seed=seed, options=options)
        if hasattr(raw, "observation"):
            return EnvResetResponse(
                observation=_coerce_observation(raw.observation),
                info=dict(getattr(raw, "info", {}) or {}),
            )
        if isinstance(raw, tuple) and len(raw) == 2:
            obs, info = raw
            return EnvResetResponse(
                observation=_coerce_observation(obs), info=dict(info or {})
            )
        return EnvResetResponse(observation=_coerce_observation(raw))

    def step(self, action: Action) -> EnvResponse:
        payload = self.action_builder(action) if self.action_builder else action
        resp = _coerce_step(self.env.step(payload))
        self._last_info = resp.info
        self._last_reward = max(self._last_reward, resp.reward)
        return resp

    def observe(self) -> Observation:
        if hasattr(self.env, "observe"):
            return _coerce_observation(self.env.observe())
        return Observation(text="")

    def evaluate(self) -> EvaluationResult:
        """Prefer the environment's own verdict; fall back to reward.

        Reward is a poor success proxy for shaped environments, so it is used
        only when nothing better exists -- and the fallback is recorded in the
        metrics so a surprising number can be traced to it.
        """
        if hasattr(self.env, "evaluate"):
            raw = self.env.evaluate()
            if hasattr(raw, "success"):
                return EvaluationResult(
                    success=bool(raw.success),
                    score=float(getattr(raw, "score", 0.0) or 0.0),
                    metrics=dict(getattr(raw, "metrics", {}) or {}),
                )
        won = bool(self._last_info.get(self.success_key, False)) or self._last_reward >= 1.0
        return EvaluationResult(
            success=won,
            score=1.0 if won else float(self._last_reward),
            metrics={"source": "reward_fallback"},
        )

    def get_env_state(self) -> dict[str, Any]:
        if hasattr(self.env, "get_env_state"):
            state = self.env.get_env_state()
            if isinstance(state, dict):
                state.setdefault("verified", self.evaluate().success)
                return state
        return {"verified": self.evaluate().success}

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


DO_TOOL = tool_schema(
    "do",
    (
        "Perform one action in the environment, written exactly as the "
        "environment expects it. If admissible commands are listed, copy one verbatim."
    ),
    {"text": {"type": "string"}},
    ["text"],
)


class TextEnvAdapter(ExternalEnvAdapter):
    """External adapter for string-in / string-out environments."""

    env_type = "external_text"

    def __init__(self, env: Any, success_key: str = "won") -> None:
        super().__init__(
            env,
            tool_schemas=[DO_TOOL],
            success_key=success_key,
            action_builder=lambda a: str(a.kwargs.get("text", a.name)),
        )


def _normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Accept both the flat and the OpenAI-nested tool-schema shapes."""
    if isinstance(schema, dict) and schema.get("type") == "function" and "function" in schema:
        return schema
    if isinstance(schema, dict) and "name" in schema:
        return {"type": "function", "function": schema}
    return schema


class ExternalBenchmark(BenchmarkAdapter):
    """Adapter over a factory that produces foreign environments.

    ::

        bench = ExternalBenchmark(
            name="alfworld",
            task_ids_fn=lambda split: alfworld_ids(split),
            env_factory=lambda task_id, **_: TextEnvAdapter(AlfworldEnv(task_id)),
        )
    """

    def __init__(
        self,
        name: str,
        env_factory: Callable[..., ActionableEnv],
        task_ids_fn: Callable[[str], Sequence[str]],
        instruction_fn: Callable[[str], str] | None = None,
        description: str = "",
        system_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.description = description or f"External benchmark {name!r}."
        self.env_factory = env_factory
        self.task_ids_fn = task_ids_fn
        self.instruction_fn = instruction_fn
        self._system_prompt = system_prompt

    def task_ids(self, split: str = "train") -> list[str]:
        return list(self.task_ids_fn(split))

    def make_env(self, task_id: str, curriculum_level: float = 0.0, seed: int = 0):
        return self.env_factory(task_id, curriculum_level=curriculum_level, seed=seed)

    def instruction_for(self, task_id: str) -> str:
        return self.instruction_fn(task_id) if self.instruction_fn else task_id

    def specs(self) -> list[TaskSpec]:
        return [
            TaskSpec(task_id=t, instruction=self.instruction_for(t))
            for t in self.task_ids("all")
        ]

    def system_prompt(self) -> str:
        return self._system_prompt or super().system_prompt()
