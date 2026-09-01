"""In-flight semantic assertions and intra-step backtracking.

Provides runtime verification of agent actions, parameters, and reasoning
before actions are committed to the environment. Inspired by DSPy's
``dspy.Assert`` and ``dspy.Suggest``, this allows the scaffolding to intercept
ill-formed actions, schema violations, out-of-bound arguments, or domain
rule breaks, and feed actionable feedback back to the agent in an intra-step
retry loop without advancing the environment step counter.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from meta_evolver.core.types import Action


class AssertionResult(BaseModel):
    """Verdict from evaluating an assertion against an action."""

    model_config = ConfigDict(extra="forbid")
    passed: bool
    message: str = ""
    is_hard: bool = True  # True = hard Assert (triggers backtrack), False = soft Suggest (warning only)
    assertion_name: str = ""


class HarnessAssertion(ABC):
    """Abstract base class for all in-flight assertions."""

    def __init__(self, name: str | None = None, is_hard: bool = True):
        self.name = name or self.__class__.__name__
        self.is_hard = is_hard

    @abstractmethod
    def evaluate(
        self,
        action: Action,
        state: dict[str, Any] | None = None,
        env_state: dict[str, Any] | None = None,
    ) -> AssertionResult:
        """Check the action. Returns AssertionResult with passed=True/False."""
        ...


class NonEmptyArgsAssertion(HarnessAssertion):
    """Verifies that specified required string or collection arguments are not empty."""

    def __init__(
        self,
        required_keys: list[str] | None = None,
        name: str | None = None,
        is_hard: bool = True,
    ):
        super().__init__(name=name or "NonEmptyArgsAssertion", is_hard=is_hard)
        self.required_keys = required_keys or []

    def evaluate(
        self,
        action: Action,
        state: dict[str, Any] | None = None,
        env_state: dict[str, Any] | None = None,
    ) -> AssertionResult:
        kwargs = action.kwargs or {}
        keys_to_check = self.required_keys or list(kwargs.keys())
        for key in keys_to_check:
            val = kwargs.get(key)
            if val is None or (isinstance(val, (str, list, dict)) and len(val) == 0):
                return AssertionResult(
                    passed=False,
                    message=f"Argument '{key}' in tool '{action.name}' must not be empty.",
                    is_hard=self.is_hard,
                    assertion_name=self.name,
                )
        return AssertionResult(passed=True, is_hard=self.is_hard, assertion_name=self.name)


class ValidToolAssertion(HarnessAssertion):
    """Verifies that the called tool is in the list of available/valid tools."""

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        name: str | None = None,
        is_hard: bool = True,
    ):
        super().__init__(name=name or "ValidToolAssertion", is_hard=is_hard)
        self.allowed_tools = set(allowed_tools) if allowed_tools else set()

    def evaluate(
        self,
        action: Action,
        state: dict[str, Any] | None = None,
        env_state: dict[str, Any] | None = None,
    ) -> AssertionResult:
        if self.allowed_tools and action.name not in self.allowed_tools:
            return AssertionResult(
                passed=False,
                message=(
                    f"Tool '{action.name}' is not in the allowed tools list: "
                    f"{sorted(self.allowed_tools)}."
                ),
                is_hard=self.is_hard,
                assertion_name=self.name,
            )
        return AssertionResult(passed=True, is_hard=self.is_hard, assertion_name=self.name)


class AdmissibleCommandAssertion(HarnessAssertion):
    """Verifies that a text command ('do') matches one of the admissible commands."""

    def __init__(self, name: str | None = None, is_hard: bool = True):
        super().__init__(name=name or "AdmissibleCommandAssertion", is_hard=is_hard)

    def evaluate(
        self,
        action: Action,
        state: dict[str, Any] | None = None,
        env_state: dict[str, Any] | None = None,
    ) -> AssertionResult:
        if action.name != "do":
            return AssertionResult(passed=True, is_hard=self.is_hard, assertion_name=self.name)

        cmd = str(action.kwargs.get("text", "")).strip()
        admissible = (state or {}).get("admissible") or []
        if admissible and cmd not in admissible:
            # Check if case-insensitive match exists
            cmd_lower = cmd.lower()
            matching = [a for a in admissible if a.lower() == cmd_lower]
            if matching:
                return AssertionResult(passed=True, is_hard=self.is_hard, assertion_name=self.name)
            # Find close matches for feedback
            sample = admissible[:6]
            return AssertionResult(
                passed=False,
                message=(
                    f"Command '{cmd}' is not in admissible actions. "
                    f"Valid examples include: {sample}"
                ),
                is_hard=self.is_hard,
                assertion_name=self.name,
            )
        return AssertionResult(passed=True, is_hard=self.is_hard, assertion_name=self.name)


class NumericRangeAssertion(HarnessAssertion):
    """Verifies that numeric parameters fall within specified [min_val, max_val] range."""

    def __init__(
        self,
        key: str,
        min_val: float | None = None,
        max_val: float | None = None,
        name: str | None = None,
        is_hard: bool = True,
    ):
        super().__init__(name=name or f"RangeAssertion[{key}]", is_hard=is_hard)
        self.key = key
        self.min_val = min_val
        self.max_val = max_val

    def evaluate(
        self,
        action: Action,
        state: dict[str, Any] | None = None,
        env_state: dict[str, Any] | None = None,
    ) -> AssertionResult:
        kwargs = action.kwargs or {}
        if self.key in kwargs:
            val = kwargs[self.key]
            try:
                num = float(val)
                if self.min_val is not None and num < self.min_val:
                    return AssertionResult(
                        passed=False,
                        message=f"Parameter '{self.key}' ({num}) is below minimum {self.min_val}.",
                        is_hard=self.is_hard,
                        assertion_name=self.name,
                    )
                if self.max_val is not None and num > self.max_val:
                    return AssertionResult(
                        passed=False,
                        message=f"Parameter '{self.key}' ({num}) exceeds maximum {self.max_val}.",
                        is_hard=self.is_hard,
                        assertion_name=self.name,
                    )
            except (ValueError, TypeError):
                return AssertionResult(
                    passed=False,
                    message=f"Parameter '{self.key}' value '{val}' must be numeric.",
                    is_hard=self.is_hard,
                    assertion_name=self.name,
                )
        return AssertionResult(passed=True, is_hard=self.is_hard, assertion_name=self.name)


class CustomAssertion(HarnessAssertion):
    """Wrap any callable predicate (action, state, env_state) -> bool | tuple[bool, str]."""

    def __init__(
        self,
        predicate: Callable[[Action, dict[str, Any] | None, dict[str, Any] | None], bool | tuple[bool, str]],
        name: str = "CustomAssertion",
        failure_message: str = "Custom assertion failed.",
        is_hard: bool = True,
    ):
        super().__init__(name=name, is_hard=is_hard)
        self.predicate = predicate
        self.failure_message = failure_message

    def evaluate(
        self,
        action: Action,
        state: dict[str, Any] | None = None,
        env_state: dict[str, Any] | None = None,
    ) -> AssertionResult:
        res = self.predicate(action, state, env_state)
        if isinstance(res, tuple):
            passed, msg = res
            return AssertionResult(
                passed=bool(passed),
                message=msg or self.failure_message,
                is_hard=self.is_hard,
                assertion_name=self.name,
            )
        return AssertionResult(
            passed=bool(res),
            message="" if res else self.failure_message,
            is_hard=self.is_hard,
            assertion_name=self.name,
        )


class AssertionRunner:
    """Manages a suite of assertions and evaluates actions against them."""

    def __init__(self, assertions: list[HarnessAssertion] | None = None):
        self.assertions: list[HarnessAssertion] = assertions or []

    def add(self, assertion: HarnessAssertion) -> AssertionRunner:
        self.assertions.append(assertion)
        return self

    def evaluate(
        self,
        action: Action,
        state: dict[str, Any] | None = None,
        env_state: dict[str, Any] | None = None,
    ) -> list[AssertionResult]:
        """Run all assertions and return their results."""
        results = []
        for assertion in self.assertions:
            try:
                res = assertion.evaluate(action, state, env_state)
                results.append(res)
            except Exception as exc:
                # If assertion code raises, record failure
                results.append(
                    AssertionResult(
                        passed=False,
                        message=f"Assertion '{assertion.name}' raised {type(exc).__name__}: {exc}",
                        is_hard=assertion.is_hard,
                        assertion_name=assertion.name,
                    )
                )
        return results

    def hard_failures(self, results: list[AssertionResult]) -> list[AssertionResult]:
        return [r for r in results if not r.passed and r.is_hard]

    def soft_warnings(self, results: list[AssertionResult]) -> list[AssertionResult]:
        return [r for r in results if not r.passed and not r.is_hard]
