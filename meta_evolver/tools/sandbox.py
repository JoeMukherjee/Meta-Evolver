"""Gate 1 of tool self-modification: AST allowlist, compile check, and
generated assertions run in a real subprocess with a timeout.

DGM's own safety story is isolation plus a compile filter plus a held-out
benchmark, run before a self-modification is ever adopted (see the vault's
DGM analysis). This module is the first two of those three for meta-evolver:
a proposed tool is source code an LLM wrote, so before it runs against any
real task it must (1) pass a structural check that rejects anything capable
of file, network, process, or interpreter-internals access, and (2) actually
work, verified by executing it -- with its own LLM-generated test cases --
in a fresh subprocess with a hard timeout, not the calling process.

The AST check is an allowlist of imports plus a denylist of dangerous names
and dunder attribute access, not a formal purity proof -- purity is
undecidable in general, and a denylist alone rots as new builtins and
gadgets appear. Combined with subprocess isolation this raises the bar
substantially; it is not a sandbox in the container sense (this project's
plan document flags that limitation explicitly), and nothing here should be
trusted to run genuinely adversarial code outside a throwaway environment.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Module names a proposed tool's source may import. Deliberately tiny --
#: nothing that touches the filesystem, network, process table, or
#: interpreter internals.
ALLOWED_IMPORTS = frozenset(
    {"math", "re", "json", "datetime", "statistics", "itertools", "collections", "textwrap"}
)

#: Names that grant filesystem, process, network, or reflection access, or
#: let code reach outside the function it was found in. Denylist, not
#: allowlist, because legitimate helper code freely uses hundreds of
#: ordinary builtins (``len``, ``sorted``, ``str.format``, ...) that an
#: allowlist would have to enumerate and would inevitably miss.
FORBIDDEN_NAMES = frozenset(
    {
        "open", "exec", "eval", "compile", "__import__", "input",
        "globals", "locals", "vars", "getattr", "setattr", "delattr",
        "breakpoint", "memoryview", "exit", "quit", "help",
    }
)


@dataclass
class ASTCheckResult:
    ok: bool
    violations: list[str] = field(default_factory=list)
    function_names: list[str] = field(default_factory=list)
    """Top-level ``def`` names found -- the sandbox runner checks the
    proposed tool's name is exactly one of these."""


def check_ast(code: str) -> ASTCheckResult:
    """Structural safety check over ``code``. Never executes it."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ASTCheckResult(ok=False, violations=[f"SyntaxError: {exc}"])

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    violations.append(f"disallowed import: {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                violations.append(f"disallowed import: {node.module!r}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            violations.append(f"forbidden name: {node.id!r}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append(f"dunder attribute access: {node.attr!r}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append(f"{type(node).__name__.lower()} statement")

    function_names = [
        n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.FunctionDef)
    ]
    if not function_names:
        violations.append("no top-level function definition found")

    # De-duplicate while keeping the first occurrence's order -- several
    # identical violations (e.g. the same forbidden name used twice) are one
    # fact worth reporting, not a wall of repeats.
    seen: set[str] = set()
    deduped = [v for v in violations if not (v in seen or seen.add(v))]
    return ASTCheckResult(ok=not deduped, violations=deduped, function_names=function_names)


@dataclass
class ToolTestCase:
    """One check the proposed tool must satisfy.

    Exactly one of ``expected`` or ``assert_code`` is used: ``expected`` for
    a plain equality check, ``assert_code`` for anything richer (a Python
    expression evaluated with ``RESULT`` bound to the call's return value,
    itself run with no builtins available -- so a test case cannot become a
    second, unchecked way to escape the sandbox).
    """

    kwargs: dict[str, Any] = field(default_factory=dict)
    expected: Any = None
    has_expected: bool = False
    assert_code: str = ""
    description: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "kwargs": self.kwargs,
            "expected": self.expected,
            "has_expected": self.has_expected,
            "assert_code": self.assert_code,
            "description": self.description,
        }


@dataclass
class SandboxResult:
    ast_ok: bool
    ast_violations: list[str]
    ran: bool
    test_results: list[dict[str, Any]]
    error: str = ""

    @property
    def all_tests_passed(self) -> bool:
        return bool(self.test_results) and all(r.get("ok") for r in self.test_results)

    @property
    def passed(self) -> bool:
        return self.ast_ok and self.ran and self.all_tests_passed and not self.error

    def render(self) -> str:
        if not self.ast_ok:
            return "AST check failed: " + "; ".join(self.ast_violations)
        if not self.ran:
            return f"execution failed: {self.error}"
        n_ok = sum(1 for r in self.test_results if r.get("ok"))
        return f"{n_ok}/{len(self.test_results)} test cases passed" + (
            f"; {self.error}" if self.error else ""
        )


_RUNNER_TEMPLATE = """\
import json, sys

{code}

_cases = json.loads(sys.argv[1])
_results = []
for _i, _case in enumerate(_cases):
    try:
        RESULT = {tool_name}(**_case["kwargs"])
        if _case.get("has_expected"):
            _ok = RESULT == _case["expected"]
        elif _case.get("assert_code"):
            _ok = bool(eval(_case["assert_code"], {{"__builtins__": {{}}, "RESULT": RESULT}}, {{}}))
        else:
            _ok = True
        _results.append({{"index": _i, "ok": _ok, "result": repr(RESULT)[:500]}})
    except Exception as exc:
        _results.append({{"index": _i, "ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}})

print(json.dumps(_results))
"""


def run_sandboxed(
    code: str,
    tool_name: str,
    test_cases: list[ToolTestCase],
    timeout: float = 5.0,
) -> SandboxResult:
    """Gate 1: check ``code`` structurally, then actually run it.

    ``code`` is expected to define exactly ``tool_name`` at module level
    (plus whatever private helpers it needs); anything else -- multiple
    top-level functions, a name mismatch -- is rejected before a subprocess
    is ever started. Execution happens in a fresh ``python`` subprocess with
    a hard wall-clock ``timeout``; a hang or an infinite loop in generated
    code costs this one gate check, not the calling process.
    """
    check = check_ast(code)
    if check.ok and tool_name not in check.function_names:
        check = ASTCheckResult(
            ok=False,
            violations=[
                f"proposed name {tool_name!r} does not match the defined "
                f"function(s) {check.function_names!r}"
            ],
            function_names=check.function_names,
        )
    if not check.ok:
        return SandboxResult(ast_ok=False, ast_violations=check.violations, ran=False, test_results=[])

    script = _RUNNER_TEMPLATE.format(code=code, tool_name=tool_name)
    cases_json = json.dumps([c.to_json() for c in test_cases])

    with tempfile.TemporaryDirectory(prefix="meta_evolver_sandbox_") as tmp:
        script_path = Path(tmp) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path), cases_json],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ast_ok=True, ast_violations=[], ran=False, test_results=[],
                error=f"timed out after {timeout}s",
            )
        except Exception as exc:
            return SandboxResult(
                ast_ok=True, ast_violations=[], ran=False, test_results=[],
                error=f"{type(exc).__name__}: {exc}",
            )

        if proc.returncode != 0:
            return SandboxResult(
                ast_ok=True, ast_violations=[], ran=False, test_results=[],
                error=(proc.stderr or "nonzero exit").strip()[-2000:],
            )
        try:
            results = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:
            return SandboxResult(
                ast_ok=True, ast_violations=[], ran=False, test_results=[],
                error=f"could not parse subprocess output: {type(exc).__name__}: {exc}",
            )
        return SandboxResult(ast_ok=True, ast_violations=[], ran=True, test_results=results)


__all__ = ["ASTCheckResult", "ToolTestCase", "SandboxResult", "check_ast", "run_sandboxed"]
