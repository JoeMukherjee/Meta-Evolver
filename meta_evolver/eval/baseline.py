"""Baseline measurement -- three arms, same held-out tasks, tokens attached.

Every later change to this project is judged against the table this module
produces, so it measures the three things a change could be credited to,
separately:

    no_memory     the base prompt, no retrieval, no learning
    bank_only     learn a memory bank on the train split; prompt frozen
    bank_evolve   learn the bank *and* evolve the prompt

All three are scored on the same ``eval`` split, which no arm ever trains on.
The difference between adjacent arms is one channel's contribution, which is a
claim worth being able to make with a number rather than an impression.

Two choices that keep the numbers comparable with published ones:

* **The curriculum is off.** It perturbs the environment (injected faults,
  noisy observations) to keep a solved task teaching. On an external QA
  benchmark that would be grading the agent on a harder task than the one
  everyone else reports.
* **The task subset is fixed across seeds.** ``seed`` varies the run, never
  the questions, so two arms -- or two commits -- are compared on identical
  task ids. With a 0.02 adoption margin and LLM sampling noise on top, an
  unpaired comparison cannot tell a real gain from a lucky draw.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.core.types import Trajectory
from meta_evolver.graphs.evolution import EvolutionConfig

ARMS: tuple[str, ...] = ("no_memory", "bank_only", "bank_evolve")


def _category(trajectory: Trajectory) -> str:
    """The benchmark's own label for a task, wherever its adapter put it."""
    metrics = trajectory.metrics or {}
    return str(metrics.get("category_name") or metrics.get("question_type") or "all")


def by_category(trajectories: Sequence[Trajectory]) -> dict[str, dict[str, float]]:
    """Pass rate and mean score per category, over usable episodes only."""
    groups: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        if t.usable:
            groups.setdefault(_category(t), []).append(t)
    return {
        label: {
            "n": len(ts),
            "pass_rate": sum(t.success for t in ts) / len(ts),
            "avg_score": sum(t.score for t in ts) / len(ts),
        }
        for label, ts in sorted(groups.items())
    }


def git_sha(cwd: str | Path | None = None) -> str:
    """Short commit id, or ``"nogit"`` -- a result must say what code made it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def run_arm(
    arm: str,
    benchmark_factory: Callable[[], Any],
    model_factory: Callable[[], BaseChatModel],
    seed: int,
    generations: int,
    max_steps: int,
    max_total_tokens: int | None,
    run_dir: Path,
    trace: bool = True,
) -> dict[str, Any]:
    """One arm, one seed. Fresh benchmark, model and bank every time, so no
    arm inherits anything another arm learned."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    bench = benchmark_factory()
    model = model_factory()
    if hasattr(bench, "judge_model") and bench.judge_model is None:
        bench.judge_model = model

    config = EvolutionConfig(
        generations=generations,
        max_steps=max_steps,
        curriculum=False,
        optimize_prompt=(arm == "bank_evolve"),
        seed=seed,
        max_total_tokens=max_total_tokens,
    )
    started = time.time()
    evolver = MetaEvolver(
        benchmark=bench,
        chat_model=model,
        config=config,
        run_dir=run_dir / f"seed{seed}" / arm,
        use_memory=(arm != "no_memory"),
        trace=trace,
    )
    try:
        train_tokens = 0
        if arm != "no_memory":
            reports = evolver.evolve()
            train_tokens = sum(r.tokens for r in reports)
        result = evolver.evaluate(split="eval", use_memory=(arm != "no_memory"))
    finally:
        evolver.close()

    return {
        "arm": arm,
        "seed": seed,
        "pass_rate": result["pass_rate"],
        "avg_score": result["avg_score"],
        "avg_steps": result["avg_steps"],
        "n_tasks": result["n_tasks"],
        "n_errors": result["n_errors"],
        "train_tokens": train_tokens,
        "eval_tokens": result["tokens"],
        "prompt_version": evolver.prompt_version,
        "memories": len(evolver.bank.items),
        "by_category": by_category(result["trajectories"]),
        "duration_s": round(time.time() - started, 1),
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and spread per arm across seeds.

    ``stdev`` is what later phases compare their gains against: an
    improvement smaller than this is not yet an improvement.
    """
    out: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm]
        if not mine:
            continue
        rates = [r["pass_rate"] for r in mine]
        scores = [r["avg_score"] for r in mine]
        out[arm] = {
            "n_seeds": len(mine),
            "pass_rate_mean": statistics.fmean(rates),
            "pass_rate_stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "avg_score_mean": statistics.fmean(scores),
            "avg_score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "tokens_mean": statistics.fmean(
                [r["train_tokens"] + r["eval_tokens"] for r in mine]
            ),
        }
    return out


def run_baseline(
    benchmark_name: str,
    benchmark_factory: Callable[[], Any],
    model_factory: Callable[[], BaseChatModel],
    model_name: str,
    seeds: Sequence[int] = (0, 1),
    arms: Sequence[str] = ARMS,
    generations: int = 2,
    max_steps: int = 15,
    max_total_tokens: int | None = None,
    out_dir: str | Path = "results",
    trace: bool = True,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Run every (seed, arm) pair and write ``<out_dir>/<bench>/<sha>.json``.

    The file is rewritten after each row, so a run that dies -- or is stopped
    because the numbers are already clearly bad -- keeps what it measured.
    """
    sha = git_sha()
    target_dir = Path(out_dir) / benchmark_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{sha}.json"

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            row = run_arm(
                arm,
                benchmark_factory,
                model_factory,
                seed=seed,
                generations=generations,
                max_steps=max_steps,
                max_total_tokens=max_total_tokens,
                run_dir=target_dir / f"{sha}_runs",
                trace=trace,
            )
            rows.append(row)
            if on_row is not None:
                on_row(row)
            target.write_text(
                json.dumps(
                    {
                        "benchmark": benchmark_name,
                        "model": model_name,
                        "git_sha": sha,
                        "generations": generations,
                        "max_steps": max_steps,
                        "max_total_tokens": max_total_tokens,
                        "summary": summarize(rows),
                        "rows": rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    return target


__all__ = ["ARMS", "by_category", "git_sha", "run_arm", "run_baseline", "summarize"]
