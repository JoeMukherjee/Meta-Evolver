"""Plug your own tasks in -- the whole integration, in one file.

    python examples/custom_benchmark.py

Supply tools as plain Python callables and tasks as a verifier over the call
log. Tool schemas are derived from each function's signature and docstring, so
the description the model reads is the docstring you already wrote.

Everything else -- the episode graph, memory induction, credit assignment,
prompt evolution, the difficulty curriculum -- applies unchanged. That is the
claim this example exists to make checkable.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_evolver import EvolutionConfig, MetaEvolver
from meta_evolver.benchmarks.custom import FunctionBenchmark, Task, ToolCallRecord

# --- a tiny world the tools operate on --------------------------------------

RELEASES = {
    "4.2.0": {"date": "2026-06-01", "changes": ["cache warmup", "new metrics"]},
    "4.2.1": {"date": "2026-06-14", "changes": ["SSO redirect rewrite", "log rotation"]},
    "4.2.2": {"date": "2026-07-02", "changes": ["docs", "typo fixes"]},
}

INCIDENTS = [
    {"id": "INC-880", "opened": "2026-06-15", "summary": "SSO login loop for enterprise tenants"},
    {"id": "INC-881", "opened": "2026-07-10", "summary": "slow dashboard load"},
]


# --- tools: ordinary functions, documented ----------------------------------


def list_releases() -> dict:
    """List every release with its date and changelog entries."""
    return {"releases": RELEASES}


def list_incidents() -> dict:
    """List open incidents with the date each was first reported."""
    return {"incidents": INCIDENTS}


def search_changelog(keyword: str) -> dict:
    """Find releases whose changelog mentions a keyword."""
    hits = [
        {"version": version, **meta}
        for version, meta in RELEASES.items()
        if any(keyword.lower() in change.lower() for change in meta["changes"])
    ]
    return {"keyword": keyword, "matches": hits}


def answer(version: str, reasoning: str) -> dict:
    """Submit the release you believe caused the incident, and why."""
    return {"version": version, "reasoning": reasoning}


# --- verification: a function over the call log ------------------------------


def broke_sso(calls: list[ToolCallRecord]) -> tuple[bool, float]:
    """Right answer, and evidence that it was actually looked up.

    Returning partial credit rather than a bare bool matters: the curriculum
    and the prompt optimizer both read the score, and a pass/fail signal
    cannot distinguish "nearly had it" from "never started".
    """
    answers = [c for c in calls if c.name == "answer" and not c.error]
    investigated = any(c.name in {"search_changelog", "list_releases"} for c in calls)

    if not answers:
        return False, 0.25 if investigated else 0.0
    correct = answers[-1].result.get("version") == "4.2.1"
    if not correct:
        return False, 0.4
    # A guess that happens to be right is not the behaviour worth reinforcing.
    return (True, 1.0) if investigated else (True, 0.7)


def main() -> int:
    bench = FunctionBenchmark(
        name="release-triage",
        description="Correlate an incident with the release that caused it.",
        tools={
            "list_releases": list_releases,
            "list_incidents": list_incidents,
            "search_changelog": search_changelog,
            "answer": answer,
        },
        tasks=[
            Task(
                id="sso-regression",
                instruction="Which release broke enterprise SSO? Answer with the version.",
                verify=broke_sso,
                terminal_tools=("answer",),
                split="train",
            ),
            Task(
                id="sso-regression-holdout",
                instruction="An enterprise tenant reports an SSO login loop. Which release is responsible?",
                verify=broke_sso,
                terminal_tools=("answer",),
                split="eval",
            ),
        ],
        max_steps=8,
    )

    evolver = MetaEvolver(
        benchmark=bench,
        model="google_genai:gemini-3-flash",
        config=EvolutionConfig(generations=2, max_steps=8, validate_prompt=False),
        telemetry=False,
    )
    evolver.evolve(on_report=lambda report: print(report.render()))
    print("\n" + evolver.render_progress())

    print("\nWhat it learned:")
    for memory in evolver.bank:
        print(f"  - {memory.render()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
