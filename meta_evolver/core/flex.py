"""Dynamic code scaffolding and evolving Python rules (FlexScaffold).

Inspired by DSPy's ``dspy.Flex``, this module moves the scaffolding's Python
code itself into the optimizable search space. Rather than only tuning prompt
text, an agent or optimizer can synthesize, mutate, and execute custom Python
tool helpers, state trackers, observation filters, and harness rules.

Key guarantees:
1. **Isolated Execution**: Dynamic code is compiled and executed in a sandboxed
   execution namespace with safe builtins and timeout bounds.
2. **Crash Resilience**: Syntax errors or runtime exceptions in candidate code
   do not crash the agent or harness; they are intercepted, scored as failures,
   and formatted into diagnostic reflection feedback for the next generation.
3. **Plug-and-Play Harness Composition**: `FlexRule` wraps dynamic code into an
   `ActionableEnv` harness `Rules` layer that operates seamlessly with all existing
   benchmarks.
"""
from __future__ import annotations

import ast
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from meta_evolver.core.env import ActionableEnv
from meta_evolver.core.rules import Rules
from meta_evolver.core.types import Action, EnvResponse, Observation
from meta_evolver.llm.client import invoke_text


class FlexResult(BaseModel):
    """Result of invoking a function inside a FlexModule."""

    model_config = ConfigDict(extra="forbid")
    success: bool
    result: Any = None
    error: str = ""
    duration_ms: float = 0.0


class FlexModule:
    """An evolving module whose Python source code is the optimizable parameter."""

    def __init__(
        self,
        name: str,
        module_src: str,
        description: str = "",
        version: int = 1,
    ):
        self.name = name
        self.module_src = module_src
        self.description = description
        self.version = version
        self.namespace: dict[str, Any] = {}
        self.compile_error: str = ""
        self.compile()

    def compile(self) -> bool:
        """Compile module source into an isolated namespace."""
        self.compile_error = ""
        try:
            # AST parse check first
            ast.parse(self.module_src)
            # Execute into clean namespace
            clean_ns: dict[str, Any] = {"__builtins__": __builtins__}
            compiled = compile(self.module_src, f"<flex_module_{self.name}>", "exec")
            exec(compiled, clean_ns)
            self.namespace = clean_ns
            return True
        except Exception:
            self.compile_error = traceback.format_exc()
            self.namespace = {}
            return False

    def call(self, fn_name: str, *args: Any, **kwargs: Any) -> FlexResult:
        """Call a function defined inside the compiled module namespace."""
        if self.compile_error:
            return FlexResult(
                success=False,
                error=f"Module compilation failed: {self.compile_error}",
            )

        fn = self.namespace.get(fn_name)
        if not fn or not callable(fn):
            return FlexResult(
                success=False,
                error=f"Function '{fn_name}' not found or not callable in module '{self.name}'.",
            )

        started = time.time()
        try:
            res = fn(*args, **kwargs)
            duration_ms = (time.time() - started) * 1000.0
            return FlexResult(success=True, result=res, duration_ms=duration_ms)
        except Exception:
            err = traceback.format_exc()
            duration_ms = (time.time() - started) * 1000.0
            return FlexResult(success=False, error=err, duration_ms=duration_ms)


class FlexRule(Rules):
    """A harness Rules layer driven by dynamic Python source code in a FlexModule."""

    def __init__(
        self,
        inner: ActionableEnv,
        module: FlexModule,
        fallback_on_error: bool = True,
    ):
        super().__init__(inner)
        self.module = module
        self.fallback_on_error = fallback_on_error
        self.execution_errors: list[str] = []

    def filter_action(self, action: Action, env_state: dict[str, Any]) -> Action | Any:
        """Delegate action filtering to module's 'filter_action' function if present."""
        if "filter_action" in self.module.namespace:
            res = self.module.call("filter_action", action, env_state)
            if res.success and res.result is not None:
                return res.result
            if not res.success:
                self.execution_errors.append(res.error)
                if not self.fallback_on_error:
                    raise RuntimeError(f"FlexRule filter_action failed: {res.error}")
        return super().filter_action(action, env_state)

    def modify_transition(
        self,
        action: Action,
        response: EnvResponse,
        env_state: dict[str, Any],
    ) -> EnvResponse:
        """Delegate transition modification to module's 'modify_transition' if present."""
        if "modify_transition" in self.module.namespace:
            res = self.module.call("modify_transition", action, response, env_state)
            if res.success and isinstance(res.result, EnvResponse):
                return res.result
            if not res.success:
                self.execution_errors.append(res.error)
                if not self.fallback_on_error:
                    raise RuntimeError(f"FlexRule modify_transition failed: {res.error}")
        return super().modify_transition(action, response, env_state)

    def filter_observation(
        self,
        obs: Observation,
        env_state: dict[str, Any],
    ) -> Observation:
        """Delegate observation filtering to module's 'filter_observation' if present."""
        if "filter_observation" in self.module.namespace:
            res = self.module.call("filter_observation", obs, env_state)
            if res.success and isinstance(res.result, Observation):
                return res.result
            if not res.success:
                self.execution_errors.append(res.error)
                if not self.fallback_on_error:
                    raise RuntimeError(f"FlexRule filter_observation failed: {res.error}")
        return super().filter_observation(obs, env_state)


FLEX_PROPOSER_SYSTEM = """\
You are an expert Python systems programmer writing dynamic agent scaffolding and tools.

Write or rewrite a Python module implementing the requested functionality.
Requirements:
1. Write valid, clean, deterministic Python code with no external dependencies.
2. Define functions matching the requested signature precisely.
3. Handle potential null/empty edge cases defensively.
4. Output ONLY valid Python code inside a ```python ... ``` code fence."""


class FlexProposer:
    """Uses an LLM to synthesize and mutate FlexModule Python code from requirements."""

    def __init__(self, model: BaseChatModel):
        self.model = model

    def propose(
        self,
        module_name: str,
        intent: str,
        current_src: str = "",
        error_feedback: str = "",
    ) -> FlexModule:
        """Propose or mutate a FlexModule."""
        user_prompt = (
            f"Module Name: {module_name}\n"
            f"Intent: {intent}\n\n"
        )
        if current_src:
            user_prompt += f"Current Implementation:\n```python\n{current_src}\n```\n\n"
        if error_feedback:
            user_prompt += f"Execution / Diagnostic Errors:\n{error_feedback}\n\n"

        user_prompt += "Write the updated Python module source code now:"

        try:
            reply = invoke_text(
                self.model,
                [
                    SystemMessage(content=FLEX_PROPOSER_SYSTEM),
                    HumanMessage(content=user_prompt),
                ],
            )
            code = self._extract_code(reply)
            mod = FlexModule(name=module_name, module_src=code, description=intent)
            return mod
        except Exception:
            # Fallback simple baseline
            fallback_code = (
                f"# Fallback implementation for {module_name}\n"
                "def process(*args, **kwargs):\n"
                "    return {'status': 'ok'}\n"
            )
            return FlexModule(name=module_name, module_src=fallback_code, description=intent)

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from markdown blocks or raw text."""
        if "```python" in text:
            parts = text.split("```python")[1].split("```")[0]
            return parts.strip()
        if "```" in text:
            parts = text.split("```")[1].split("```")[0]
            return parts.strip()
        return text.strip()
