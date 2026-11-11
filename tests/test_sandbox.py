"""Gate 1: AST allowlist/denylist, and real subprocess execution with a
hard timeout -- offline, no model calls."""
from __future__ import annotations

from meta_evolver.tools.sandbox import ToolTestCase, check_ast, run_sandboxed

_GOOD = """\
def add_tax(amount, rate):
    return round(amount * (1 + rate), 2)
"""


def test_ast_check_accepts_a_clean_pure_function():
    result = check_ast(_GOOD)
    assert result.ok
    assert result.function_names == ["add_tax"]


def test_ast_check_rejects_disallowed_import():
    code = "import os\n\ndef f():\n    return os.listdir('.')\n"
    result = check_ast(code)
    assert not result.ok
    assert any("os" in v for v in result.violations)


def test_ast_check_rejects_forbidden_builtin():
    code = "def f(x):\n    return eval(x)\n"
    result = check_ast(code)
    assert not result.ok
    assert any("eval" in v for v in result.violations)


def test_ast_check_rejects_dunder_attribute_access():
    code = "def f():\n    return (1).__class__.__bases__\n"
    result = check_ast(code)
    assert not result.ok
    assert any("dunder" in v for v in result.violations)


def test_ast_check_rejects_syntax_errors():
    result = check_ast("def f(:\n")
    assert not result.ok
    assert any("SyntaxError" in v for v in result.violations)


def test_ast_check_rejects_code_with_no_function():
    result = check_ast("x = 1\n")
    assert not result.ok
    assert any("no top-level function" in v for v in result.violations)


def test_run_sandboxed_passes_clean_function_and_equality_test_cases():
    result = run_sandboxed(
        _GOOD,
        "add_tax",
        [
            ToolTestCase(kwargs={"amount": 100, "rate": 0.1}, expected=110.0, has_expected=True),
            ToolTestCase(kwargs={"amount": 0, "rate": 0.5}, expected=0.0, has_expected=True),
        ],
    )
    assert result.passed
    assert result.all_tests_passed


def test_run_sandboxed_reports_a_failing_test_case_without_crashing():
    result = run_sandboxed(
        _GOOD,
        "add_tax",
        [ToolTestCase(kwargs={"amount": 100, "rate": 0.1}, expected=999.0, has_expected=True)],
    )
    assert result.ran
    assert not result.passed
    assert not result.test_results[0]["ok"]


def test_run_sandboxed_supports_assert_code_test_cases():
    result = run_sandboxed(
        _GOOD,
        "add_tax",
        [ToolTestCase(kwargs={"amount": 100, "rate": 0.0}, assert_code="RESULT == 100.0")],
    )
    assert result.passed


def test_run_sandboxed_never_executes_code_that_fails_the_ast_gate():
    result = run_sandboxed(
        "import os\n\ndef f():\n    return os.getcwd()\n",
        "f",
        [ToolTestCase(kwargs={})],
    )
    assert not result.ast_ok
    assert not result.ran  # the subprocess must never have started


def test_run_sandboxed_rejects_a_name_mismatch():
    result = run_sandboxed(_GOOD, "not_the_real_name", [ToolTestCase(kwargs={})])
    assert not result.ast_ok
    assert any("does not match" in v for v in result.ast_violations)


def test_run_sandboxed_catches_a_runtime_exception_inside_the_function():
    code = "def divide(a, b):\n    return a / b\n"
    result = run_sandboxed(code, "divide", [ToolTestCase(kwargs={"a": 1, "b": 0})])
    assert result.ran
    assert not result.passed
    assert "ZeroDivisionError" in result.test_results[0]["error"]


def test_run_sandboxed_enforces_a_hard_timeout():
    code = "def loop_forever():\n    while True:\n        pass\n"
    result = run_sandboxed(code, "loop_forever", [ToolTestCase(kwargs={})], timeout=0.5)
    assert not result.ran
    assert "timed out" in result.error


def test_assert_code_test_cases_cannot_escape_via_builtins():
    """The assert_code eval runs with no builtins -- a proposed test case
    trying to use it as a second sandbox-escape path must fail cleanly, not
    silently succeed."""
    code = "def f():\n    return 1\n"
    result = run_sandboxed(
        code, "f", [ToolTestCase(kwargs={}, assert_code="__import__('os').getcwd() or True")]
    )
    assert result.ran
    assert not result.test_results[0]["ok"]
