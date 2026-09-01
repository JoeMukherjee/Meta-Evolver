"""FunctionBenchmark -- wire up your own tasks without writing an environment.

The fastest path from "I have an evaluation set" to "the evolver is improving
on it". You supply tasks, Python callables as tools, and a verifier; this
module supplies the ``ActionableEnv`` and the adapter.

::

    from meta_evolver.benchmarks.custom import FunctionBenchmark, Task

    def search(query: str) -> dict:
        return {"hits": my_index.search(query)}

    bench = FunctionBenchmark(
        name="support-triage",
        tools={"search": search, "answer": lambda text: {"answer": text}},
        tasks=[
            Task(id="t1", instruction="Which release broke SSO?",
                 verify=lambda calls: any(
                     "4.2.1" in str(c.result.get("answer", "")) for c in calls
                     if c.name == "answer")),
        ],
    )

That is the whole integration. Tool schemas are derived from each callable's
signature and docstring, so the description the model sees is the docstring you
already wrote.

Two conventions worth knowing. A tool raising an exception becomes an error
observation rather than crashing the episode -- an agent should be able to
recover from a bad argument. And ``verify`` receives the full list of calls,
not just the last, so a task can require that a check actually ran rather than
only that the final answer looks right.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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

#: Python annotation -> JSON Schema type.
#:
#: Keyed by the type's *name* rather than the type object, because a module
#: using ``from __future__ import annotations`` -- as this one does, and as
#: most user code will -- hands ``inspect.signature`` the string ``"int"``
#: instead of ``int``. Looking up the object would miss every annotation and
#: silently type everything as a string, which reaches the model as a tool
#: whose integer parameter is declared textual.
_JSON_TYPES: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "Any": "string",
}


def _json_type(annotation: Any) -> str:
    """JSON Schema type for a parameter annotation, defaulting to string.

    ``string`` is the safe fallback: every provider accepts it, and a wrong
    type declaration surfaces as a validation error the agent cannot fix.
    """
    if annotation is inspect.Parameter.empty:
        return "string"
    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    # Strip Optional[...] / list[...] wrappers down to the head type.
    head = str(name).split("[")[0].replace("typing.", "").strip()
    if head.startswith("Optional") or " | " in head:
        head = head.replace("Optional", "").strip("[] ").split("|")[0].strip()
    return _JSON_TYPES.get(head, "string")


@dataclass
class ToolCallRecord:
    """One executed tool call, as handed to a verifier."""

    name: str
    kwargs: dict[str, Any]
    result: dict[str, Any]
    error: str = ""


@dataclass
class Task:
    """One unit of work.

    ``verify`` receives the call log and returns either a bool or a
    ``(success, score)`` pair, so a task can express partial credit -- which
    the curriculum and the prompt optimizer both make better use of than a
    bare pass/fail.
    """

    id: str
    instruction: str = ""
    verify: Callable[[list[ToolCallRecord]], bool | tuple[bool, float]] | None = None
    setup: Callable[[], dict[str, Any]] | None = None
    split: str = "train"
    difficulty: float = 0.5
    context: dict[str, Any] = field(default_factory=dict)
    terminal_tools: tuple[str, ...] = ()
    """Tools that end the episode when called, e.g. ``("answer",)``."""


class FunctionEnv(ActionableEnv):
    """An environment whose actions are plain Python callables."""

    env_type = "function_env"

    def __init__(
        self,
        task: Task,
        tools: dict[str, Callable[..., Any]],
        schemas: list[dict[str, Any]],
        max_steps: int = 15,
    ) -> None:
        self.task = task
        self.tools = tools
        self.tool_schemas = schemas
        self.max_steps = int(max_steps)
        self.calls: list[ToolCallRecord] = []
        self.step_count = 0
        self.finished = False
        self.context: dict[str, Any] = {}

    def reset(self, seed=None, options=None) -> EnvResetResponse:
        self.calls = []
        self.step_count = 0
        self.finished = False
        self.context = dict(self.task.context)
        if self.task.setup is not None:
            self.context.update(self.task.setup() or {})
        return EnvResetResponse(
            observation=self.observe(),
            info={
                "task_id": self.task.id,
                "instruction": self.task.instruction,
                "title": self.task.instruction,
            },
        )

    def step(self, action: Action) -> EnvResponse:
        self.step_count += 1
        name, kwargs = action.name, dict(action.kwargs or {})

        fn = self.tools.get(name)
        if fn is None:
            record = ToolCallRecord(
                name=name,
                kwargs=kwargs,
                result={"error": f"unknown tool {name!r}", "available": sorted(self.tools)},
                error="unknown tool",
            )
        else:
            try:
                raw = fn(**kwargs)
                result = raw if isinstance(raw, dict) else {"value": raw}
                record = ToolCallRecord(name=name, kwargs=kwargs, result=result)
            except Exception as exc:
                # A tool error is information for the agent, not a crash. It
                # gets one more chance to call it correctly.
                record = ToolCallRecord(
                    name=name,
                    kwargs=kwargs,
                    result={"error": f"{type(exc).__name__}: {exc}"},
                    error=str(exc),
                )
        self.calls.append(record)

        if name in self.task.terminal_tools and not record.error:
            self.finished = True

        evaluation = self.evaluate()
        return EnvResponse(
            observation=self.observe(),
            reward=evaluation.score,
            terminated=self.finished,
            truncated=self.step_count >= self.max_steps,
            info={"result": record.result, "step": self.step_count},
        )

    def observe(self) -> Observation:
        lines = [f"Task: {self.task.instruction}", f"Step {self.step_count}/{self.max_steps}"]
        if self.context:
            lines.append(f"Context: {self.context}")
        if self.calls:
            last = self.calls[-1]
            lines.append(f"Last call {last.name}({last.kwargs}) -> {last.result}")
        return Observation(
            text="\n".join(lines),
            data={"task_id": self.task.id, "n_calls": len(self.calls)},
        )

    def evaluate(self) -> EvaluationResult:
        if self.task.verify is None:
            return EvaluationResult(success=self.finished, score=1.0 if self.finished else 0.0)
        try:
            verdict = self.task.verify(list(self.calls))
        except Exception as exc:
            # A broken verifier must not silently mark everything failed --
            # that would look like an agent regression and mislead the whole
            # evolution loop. Surface it in the metrics.
            return EvaluationResult(
                success=False, score=0.0, metrics={"verifier_error": str(exc)}
            )
        if isinstance(verdict, tuple):
            success, score = bool(verdict[0]), float(verdict[1])
        else:
            success, score = bool(verdict), 1.0 if verdict else 0.0
        return EvaluationResult(
            success=success, score=score, metrics={"n_calls": len(self.calls)}
        )

    def get_env_state(self) -> dict[str, Any]:
        return {
            "task_id": self.task.id,
            "step_count": self.step_count,
            "calls": [c.name for c in self.calls],
            "verified": self.evaluate().success,
        }


class FunctionBenchmark(BenchmarkAdapter):
    """Turns a list of :class:`Task` plus a dict of callables into a benchmark."""

    def __init__(
        self,
        name: str,
        tools: dict[str, Callable[..., Any]],
        tasks: Sequence[Task],
        max_steps: int = 15,
        description: str = "",
        system_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.description = description or f"User-defined benchmark {name!r}."
        self.tools = dict(tools)
        self.tasks = {t.id: t for t in tasks}
        self.max_steps = int(max_steps)
        self._system_prompt = system_prompt
        self.schemas = [derive_schema(n, f) for n, f in self.tools.items()]

    def task_ids(self, split: str = "train") -> list[str]:
        if split == "all":
            return list(self.tasks)
        return [tid for tid, task in self.tasks.items() if task.split == split]

    def make_env(self, task_id: str, curriculum_level: float = 0.0, seed: int = 0):
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task {task_id!r}; have {sorted(self.tasks)}")
        return FunctionEnv(task, self.tools, self.schemas, max_steps=self.max_steps)

    def instruction_for(self, task_id: str) -> str:
        task = self.tasks.get(task_id)
        return task.instruction if task else task_id

    def specs(self) -> list[TaskSpec]:
        return [
            TaskSpec(
                task_id=t.id,
                instruction=t.instruction,
                split=t.split,
                difficulty=t.difficulty,
            )
            for t in self.tasks.values()
        ]

    def system_prompt(self) -> str:
        return self._system_prompt or super().system_prompt()


def derive_schema(name: str, fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a tool schema from a callable's signature and docstring.

    Parameters without a default are marked required; annotations map to JSON
    Schema types where recognized and fall back to ``string``, which every
    provider accepts. The docstring's first line becomes the description the
    model actually reads, so an undocumented tool is a tool the agent will
    guess at.
    """
    doc = inspect.getdoc(fn) or f"Call {name}."
    summary = doc.strip().split("\n\n")[0].replace("\n", " ")

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins without introspectable signatures
        return tool_schema(name, summary)

    for param_name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[param_name] = {"type": _json_type(param.annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    return tool_schema(name, summary, properties, required)
