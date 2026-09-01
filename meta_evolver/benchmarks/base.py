"""BenchmarkAdapter -- the one seam a new benchmark has to fill.

Everything above this line (the graphs, the memory bank, the prompt optimizer,
the curriculum) is benchmark-agnostic. Everything a benchmark knows lives
below it. Adding SWE-bench, WebArena, or an internal evaluation set means
writing one subclass and one decorator; no engine code changes.

The required surface is deliberately three methods:

    task_ids(split)      -- what tasks exist
    make_env(task_id)    -- an ActionableEnv for one of them
    instruction_for(id)  -- the natural-language goal

Everything else has a working default. ``sample`` in particular implements the
train/validation discipline the evolution loop depends on, so an adapter gets
held-out prompt selection for free and cannot accidentally leak validation
tasks into training by forgetting to.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from meta_evolver.core.env import ActionableEnv
from meta_evolver.core.seeding import derive_rng
from meta_evolver.core.types import TaskSpec
from meta_evolver.prompts.templates import BASE_SYSTEM_PROMPT


class BenchmarkAdapter(ABC):
    """Adapts an evaluation suite to the Meta-Evolver engine."""

    #: Set by ``@register_benchmark``.
    name: str = "unnamed"

    #: Shown in ``meta-evolver benchmarks``.
    description: str = ""

    # -- required ----------------------------------------------------------

    @abstractmethod
    def task_ids(self, split: str = "train") -> list[str]:
        """Task identifiers in ``split``.

        Recognized splits are ``train``, ``eval``, and ``all``. An adapter with
        no natural split may return the same list for each.
        """

    @abstractmethod
    def make_env(
        self, task_id: str, curriculum_level: float = 0.0, seed: int = 0
    ) -> ActionableEnv:
        """A fresh environment for ``task_id``.

        Must return a *new* instance each call: episodes run concurrently, and
        a shared environment would interleave two agents' mutations.

        ``curriculum_level`` is passed for benchmarks with intrinsic difficulty
        (a harder task variant). Generic perturbation is layered on separately
        by :class:`~meta_evolver.harness.curriculum.Curriculum`, so an adapter
        that has no notion of difficulty can ignore it entirely.
        """

    # -- defaults worth overriding ----------------------------------------

    def instruction_for(self, task_id: str) -> str:
        spec = self.spec_for(task_id)
        return spec.instruction if spec else task_id

    def spec_for(self, task_id: str) -> TaskSpec | None:
        for spec in self.specs():
            if spec.task_id == task_id:
                return spec
        return None

    def specs(self) -> list[TaskSpec]:
        """All tasks as :class:`TaskSpec`. Default derives them from ids."""
        return [TaskSpec(task_id=t, instruction=t) for t in self.task_ids("all")]

    def system_prompt(self) -> str:
        """The starting system prompt, before any evolution.

        Override to add domain framing the agent could not infer -- but keep it
        short. The optimizer will specialise it from evidence, and a long
        hand-written prompt gives it less room to.
        """
        return BASE_SYSTEM_PROMPT

    # -- provided ----------------------------------------------------------

    def sample(
        self,
        generation: int = 0,
        n: int | None = None,
        validation_fraction: float = 0.0,
        seed: int = 0,
    ) -> tuple[list[str], list[str]]:
        """``(train_ids, validation_ids)`` for one generation.

        Two properties matter and are easy to get wrong:

        * The two lists are disjoint. A prompt selected on tasks it was
          written from always looks like an improvement.
        * The validation set is *stable across generations* (seeded by ``seed``
          alone, not by ``generation``), so a pass rate at generation 4 is
          comparable with one at generation 1. Reshuffling it every generation
          turns the headline metric into noise.

        The training rotation does vary by generation, so a long run sees the
        whole task pool rather than overfitting one slice.
        """
        pool = list(self.task_ids("train"))
        if not pool:
            return [], []

        holdout: list[str] = []
        if validation_fraction > 0 and len(pool) >= 3:
            k = max(1, int(round(len(pool) * validation_fraction)))
            k = min(k, len(pool) - 1)  # never leave training empty
            holdout = derive_rng("holdout", seed).sample(sorted(pool), k)

        train_pool = [t for t in pool if t not in set(holdout)]
        if n and n < len(train_pool):
            train = derive_rng(seed, generation).sample(sorted(train_pool), n)
        else:
            train = train_pool
        return train, holdout

    def close(self) -> None:
        """Release suite-level resources (a dataset handle, a container pool)."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} tasks={len(self.task_ids('all'))}>"


# ---------------------------------------------------------------------------
# Tool-schema helper -- most adapters need this and it is fiddly to get right
# ---------------------------------------------------------------------------


def tool_schema(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]] | None = None,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    """Build one OpenAI-format tool schema.

    Wrapped in a helper because the nesting (``type``/``function``/
    ``parameters``/``properties``) is easy to get subtly wrong, and a
    malformed schema fails as "the model never calls this tool" rather than as
    an error.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": list(required),
            },
        },
    }
