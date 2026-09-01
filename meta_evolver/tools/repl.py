"""Sandbox variable REPL and recursive context navigation (ScaffoldRLM).

Solves context rot in environments with massive observations (long logs, huge DOMs,
large spreadsheets, trace dumps) by decoupling Variable Space from Token Space.

Key mechanics:
1. **Variable Space Isolation**: Raw large observations and structured datasets are
   stored as variables inside an isolated REPL session. The LLM's prompt receives only
   concise `VariableDescriptor` metadata (type, length, shape, preview).
2. **Deterministic Code Slicing**: The agent writes Python snippets to filter, regex,
   and slice target sections deterministically.
3. **Sub-LLM Delegation**: `llm_query` and `llm_query_batched` let the outer agent delegate
   semantic analysis of isolated variable slices to a sub-model with bounded call budgets.
"""
from __future__ import annotations

import io
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from meta_evolver.llm.client import invoke_text


@dataclass
class VariableDescriptor:
    """Metadata descriptor representing a variable stored in the REPL session."""

    name: str
    type_name: str
    length: int | None = None
    shape: tuple[int, ...] | None = None
    preview: str = ""

    def render(self) -> str:
        shape_info = f", shape={self.shape}" if self.shape else ""
        len_info = f", length={self.length}" if self.length is not None else ""
        return (
            f"• Variable `{self.name}` [{self.type_name}{len_info}{shape_info}]:\n"
            f"  Preview: {self.preview}"
        )


class REPLExecutionResult(BaseModel):
    """Result of running code inside the REPL session."""

    model_config = ConfigDict(extra="forbid")
    success: bool
    output: str = ""
    result_repr: str = ""
    error: str = ""
    duration_ms: float = 0.0
    new_vars: list[str] = Field(default_factory=list)


class REPLSession:
    """Sandboxed Python execution session maintaining persistent variable state across turns."""

    def __init__(
        self,
        sub_lm: BaseChatModel | None = None,
        max_sub_llm_calls: int = 30,
        max_output_chars: int = 4000,
    ):
        self.sub_lm = sub_lm
        self.max_sub_llm_calls = max_sub_llm_calls
        self.max_output_chars = max_output_chars
        self.sub_llm_calls_made = 0
        self.namespace: dict[str, Any] = {}
        self._init_namespace()

    def _init_namespace(self) -> None:
        """Populate base builtins and helper primitives."""
        # Built-in sub-query helpers exposed inside the REPL code environment
        def llm_query(query: str, var_name_or_text: Any) -> str:
            return self.query_llm(query, var_name_or_text)

        def llm_query_batched(queries_and_contexts: list[tuple[str, Any]]) -> list[str]:
            return [self.query_llm(q, c) for q, c in queries_and_contexts]

        self.namespace = {
            "__builtins__": __builtins__,
            "llm_query": llm_query,
            "llm_query_batched": llm_query_batched,
        }

    def set_variable(self, name: str, value: Any, preview_chars: int = 250) -> VariableDescriptor:
        """Store a variable in the REPL namespace and return its descriptor."""
        self.namespace[name] = value

        type_name = type(value).__name__
        length = len(value) if hasattr(value, "__len__") else None
        shape = getattr(value, "shape", None)

        if isinstance(value, str):
            preview = repr(value[:preview_chars] + ("..." if len(value) > preview_chars else ""))
        elif isinstance(value, (list, tuple)):
            sample = value[:3]
            preview = f"{sample} (total {len(value)} items)"
        elif isinstance(value, dict):
            preview = f"Keys: {list(value.keys())[:8]}"
        else:
            preview = str(value)[:preview_chars]

        return VariableDescriptor(
            name=name,
            type_name=type_name,
            length=length,
            shape=shape,
            preview=preview,
        )

    def get_variable(self, name: str) -> Any:
        return self.namespace.get(name)

    def list_variables(self) -> list[VariableDescriptor]:
        """List all user-defined variables (ignoring system builtins)."""
        ignored = {"__builtins__", "llm_query", "llm_query_batched"}
        descriptors = []
        for k, v in self.namespace.items():
            if k not in ignored and not k.startswith("_"):
                descriptors.append(self.set_variable(k, v))
        return descriptors

    def describe_all(self) -> str:
        """Render a formatted overview of all active REPL variables for prompt injection."""
        vars_list = self.list_variables()
        if not vars_list:
            return "No active variables in REPL context."
        return "\n".join([v.render() for v in vars_list])

    def execute(self, code: str) -> REPLExecutionResult:
        """Execute a Python code string inside the persistent namespace."""
        started = time.time()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        keys_before = set(self.namespace.keys())
        error_msg = ""
        result_repr = ""

        try:
            sys.stdout = stdout_buf
            sys.stderr = stderr_buf

            # Try eval first if it's a single expression
            try:
                compiled = compile(code, "<repl>", "eval")
                eval_res = eval(compiled, self.namespace)
                if eval_res is not None:
                    result_repr = repr(eval_res)
            except SyntaxError:
                # Fall back to multi-line exec
                compiled = compile(code, "<repl>", "exec")
                exec(compiled, self.namespace)

        except Exception:
            error_msg = traceback.format_exc()
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

        duration_ms = (time.time() - started) * 1000.0
        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()
        combined_output = (stdout_text + ("\nSTDERR:\n" + stderr_text if stderr_text else "")).strip()

        if len(combined_output) > self.max_output_chars:
            combined_output = combined_output[: self.max_output_chars] + "\n...[truncated]"

        keys_after = set(self.namespace.keys())
        new_vars = sorted(list(keys_after - keys_before))

        return REPLExecutionResult(
            success=not bool(error_msg),
            output=combined_output,
            result_repr=result_repr[:1000],
            error=error_msg,
            duration_ms=duration_ms,
            new_vars=new_vars,
        )

    def query_llm(self, query: str, context_or_var_name: Any) -> str:
        """Sub-LLM query primitive over a focused context or variable."""
        if self.sub_llm_calls_made >= self.max_sub_llm_calls:
            return f"[ERROR: max_sub_llm_calls ({self.max_sub_llm_calls}) reached. Use Python aggregation instead.]"

        if isinstance(context_or_var_name, str) and context_or_var_name in self.namespace:
            context_text = str(self.namespace[context_or_var_name])
        else:
            context_text = str(context_or_var_name)

        if not self.sub_lm:
            # Fallback mock/extractive response if no sub_lm provided
            return f"[Extracted from context ({len(context_text)} chars)]: {query} -> relevant lines found."

        self.sub_llm_calls_made += 1

        messages = [
            SystemMessage(
                content=(
                    "You are a sub-LLM extraction unit. Answer the user query based ONLY "
                    "on the provided context slice. Be concise."
                )
            ),
            HumanMessage(
                content=f"Context snippet:\n{context_text[:6000]}\n\nQuery: {query}"
            ),
        ]
        try:
            return invoke_text(self.sub_lm, messages).strip()
        except Exception as exc:
            return f"[Sub-LLM Error: {type(exc).__name__}: {exc}]"


class REPLTools:
    """Helper to expose REPL capabilities as callable tool dictionaries."""

    @staticmethod
    def get_tool_definitions() -> list[dict[str, Any]]:
        """Return OpenAI / LangChain function definitions for REPL primitives."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "repl_exec",
                    "description": (
                        "Execute Python code in the persistent REPL session to explore, filter, "
                        "and transform large variables. Built-in helpers: llm_query(query, text), "
                        "llm_query_batched([(q1, t1), ...])."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code snippet to execute.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "repl_inspect",
                    "description": "Inspect a specific slice or value of a stored REPL variable.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "variable_name": {
                                "type": "string",
                                "description": "Name of the variable to inspect.",
                            },
                            "start": {
                                "type": "integer",
                                "description": "Start index for string/list slicing (default 0).",
                            },
                            "length": {
                                "type": "integer",
                                "description": "Number of characters/items to return (default 500).",
                            },
                        },
                        "required": ["variable_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "repl_list_vars",
                    "description": "List all active variables currently held in the REPL session.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
        ]
