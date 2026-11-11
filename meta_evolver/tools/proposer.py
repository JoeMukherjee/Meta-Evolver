"""Gate 0 of tool self-modification: propose a pure-function tool from a
cluster of failed episodes.

DGM proposes changes to an agent's own code from its failure history; this is
the narrow, safe slice of that idea this project adopts -- research-only, as
the roadmap names it. Rather than letting a model rewrite scaffold code
directly, it is asked for one small, pure, side-effect-free helper function
(no file, network, or process access -- see :mod:`meta_evolver.tools.sandbox`
for what that excludes) that a cluster of failures suggests the agent was
missing, plus test cases proving it does what it claims. Nothing here decides
whether the tool is actually good; :mod:`meta_evolver.tools.sandbox` checks it
is safe to run at all, and the evolution graph's held-out validation (Gate 2)
decides whether it is worth keeping, exactly like a prompt candidate.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from meta_evolver.core.types import Trajectory
from meta_evolver.llm.client import LLMError, invoke_text
from meta_evolver.tools.sandbox import ALLOWED_IMPORTS, ToolTestCase

PROPOSER_SYSTEM = """\
You design one small Python tool for an autonomous agent that is failing at \
a task, based on traces of what went wrong.

The tool must be a single PURE function: given its arguments, it computes and \
returns a value with no side effects. It may NOT read or write files, make \
network calls, spawn processes, or use reflection -- only plain computation. \
Allowed imports: {allowed_imports}. If the failures do not suggest a missing \
*computation* (a calculation, a parse, a lookup, a transform) -- for example \
if the agent just needs to call an existing tool it already has -- say so \
instead of inventing one.

Reply with exactly one fenced ```json``` block, nothing else, shaped like:
{{
  "name": "snake_case_function_name",
  "description": "one sentence: what it computes and when the agent should call it",
  "code": "def snake_case_function_name(...):\\n    ...\\n    return ...",
  "tests": [
    {{"kwargs": {{...}}, "expected": <value>}},
    {{"kwargs": {{...}}, "assert_code": "RESULT > 0"}}
  ]
}}

"code" must define exactly one top-level function, named exactly "name". \
Give at least 2 test cases that a correct implementation must satisfy -- they \
will actually be executed, so they must be genuinely correct, not aspirational. \
Use "expected" for an exact value, "assert_code" for a Python boolean \
expression over the bound name RESULT. If no tool is warranted, reply with \
{{"name": "", "description": "", "code": "", "tests": []}} instead."""

PROPOSER_USER = """\
## Benchmark
{benchmark}

## Recurring failure signatures
{signatures}

## Failure traces
{failures}

Propose the tool now, as one fenced json block."""


@dataclass
class ToolProposal:
    name: str
    code: str
    description: str
    test_cases: list[ToolTestCase] = field(default_factory=list)
    version: str = "t0"
    score: float | None = None


class ToolProposer:
    """Proposes candidate pure-function tools from failure evidence.

    Mirrors :class:`~meta_evolver.prompts.optimizer.PromptOptimizer`'s shape
    on purpose: same ``propose(trajectories, generation, benchmark)``-style
    entry point, same "nothing to learn from -> empty list" contract, so the
    evolution graph's tool-adoption node reads exactly like its
    prompt-adoption counterpart.
    """

    def __init__(
        self,
        model: BaseChatModel,
        max_failures_in_prompt: int = 4,
        max_steps_per_trace: int = 10,
    ) -> None:
        self.model = model
        self.max_failures_in_prompt = int(max_failures_in_prompt)
        self.max_steps_per_trace = int(max_steps_per_trace)
        self.last_error: str = ""

    def propose(
        self,
        trajectories: Sequence[Trajectory],
        generation: int = 0,
        benchmark: str = "",
    ) -> list[ToolProposal]:
        """One candidate tool, or an empty list when there is nothing to
        propose -- no usable episodes, no failures, or the model itself
        declined (it is explicitly told it may)."""
        usable = [t for t in trajectories if t.usable]
        failures = [t for t in usable if not t.success]
        if not usable or not failures:
            return []

        by_task: dict[str, Trajectory] = {}
        for t in failures:
            by_task.setdefault(t.task_id, t)
        chosen = list(by_task.values())[: self.max_failures_in_prompt]

        body = PROPOSER_USER.format(
            benchmark=benchmark or "this benchmark",
            signatures=f"{len(failures)} failed episode(s) analysed.",
            failures="\n\n".join(t.render(max_steps=self.max_steps_per_trace) for t in chosen),
        )
        system = PROPOSER_SYSTEM.format(allowed_imports=", ".join(sorted(ALLOWED_IMPORTS)))

        try:
            content = invoke_text(
                self.model, [SystemMessage(content=system), HumanMessage(content=body)]
            )
        except LLMError as exc:
            self.last_error = str(exc)
            return []

        proposal = _parse(content, version=f"g{generation}t0")
        return [proposal] if proposal is not None else []


def _parse(content: str, version: str) -> ToolProposal | None:
    """Best-effort extraction of the fenced JSON block. Malformed or
    empty-by-design replies both return ``None`` -- a proposer that cannot
    parse its own model's reply must fail closed, not raise into the graph."""
    match = re.search(r"```(?:json)?\s*\n(.*?)\n?```", content, re.DOTALL)
    raw = match.group(1) if match else content
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    name = str(data.get("name") or "").strip()
    code = str(data.get("code") or "").strip()
    if not name or not code:
        return None  # the model declined, or gave an unusable reply

    tests_raw = data.get("tests") or []
    if not isinstance(tests_raw, list) or not tests_raw:
        return None  # an untested proposal is not one the sandbox gate can even run

    test_cases: list[ToolTestCase] = []
    for case in tests_raw:
        if not isinstance(case, dict):
            continue
        has_expected = "expected" in case
        test_cases.append(
            ToolTestCase(
                kwargs=dict(case.get("kwargs") or {}),
                expected=case.get("expected"),
                has_expected=has_expected,
                assert_code=str(case.get("assert_code") or ""),
                description=str(case.get("description") or ""),
            )
        )
    if not test_cases:
        return None

    return ToolProposal(
        name=name,
        code=code,
        description=str(data.get("description") or ""),
        test_cases=test_cases,
        version=version,
    )


__all__ = ["ToolProposal", "ToolProposer"]
