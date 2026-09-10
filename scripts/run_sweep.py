"""Run the seed sweep that turns our claims into measurements.

The project's central claim is that the agent improves generation over
generation, and the supporting claim is that three scaffold channels each
contribute. Neither is currently measured: the numbers on hand come from a
single seed per condition, which is enough to demonstrate a mechanism and not
enough to put in a results table.

This runs the grid that fixes that:

    conditions x seeds x generations

with one condition per channel switched off, so the ablation answers "which
channel does the work" rather than only "does the whole thing help".

Three design points, each of which is the difference between a number you can
publish and one you cannot:

**Seeds vary the environment, not just the sampler.** ``seed`` reaches the
harness perturbations, the task rotation and the held-out split, so two seeds
are genuinely different runs rather than the same run with different noise.

**Every run is independent.** Each gets a fresh bank, a fresh prompt and its
own ``run_id``, so nothing leaks between conditions. A shared memory bank
would make the ablation meaningless -- the no-memory arm would still benefit
from memories another arm wrote.

**It is resumable and it is honest about cost.** Results append to JSONL as
they complete, a re-run skips what is already there, and ``--dry-run`` proves
the whole grid works against a scripted model before a single paid call.

    python scripts/run_sweep.py --dry-run                  # free, validates the grid
    python scripts/run_sweep.py --seeds 5 --generations 6  # the real thing
    python scripts/run_sweep.py --summarise                # aggregate what exists
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO.parents[1] / "runs" / "sweeps"

# `meta_evolver` is a source package here, not an installed one, so it is only
# importable once this directory is on sys.path. The dry-run path used to get
# this for free as a side effect of importing offline_demo.py (which does its
# own path insertion); the real path imported meta_evolver directly and never
# got it, so a live run failed at the first API-backed model construction --
# after the grid had already printed "N to run" and looked like it started.
sys.path.insert(0, str(REPO))

#: Difficulty at which the second held-out measurement is taken. 0.65 is the
#: "noisy + gated" band: transient faults, a verification gate and distractor
#: observations all active -- the lowest level where all three failure modes
#: the scaffold is meant to handle are present at once.
HARD_LEVEL = 0.65


@dataclass
class Condition:
    """One arm of the ablation.

    ``overrides`` are applied to :class:`EvolutionConfig`; ``use_memory``
    controls retrieval itself rather than induction, which is a different
    thing: an arm can still *write* memories while never *reading* them.
    """

    name: str
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)
    use_memory: bool = True
    why: str = ""


CONDITIONS: list[Condition] = [
    Condition(
        "full", "Full system",
        why="all three channels plus curriculum",
    ),
    Condition(
        "no_memory", "No memory",
        overrides={"induce_memories": False},
        use_memory=False,
        why="isolates what the reasoning bank contributes",
    ),
    Condition(
        "no_prompt", "No prompt evolution",
        overrides={"optimize_prompt": False},
        why="isolates what OPRO contributes",
    ),
    Condition(
        "no_curriculum", "No curriculum",
        overrides={"curriculum": False},
        why="isolates whether escalating difficulty helps or merely looks hard",
    ),
    Condition(
        "static", "Static baseline",
        overrides={"induce_memories": False, "optimize_prompt": False, "curriculum": False},
        use_memory=False,
        why="the same agent with nothing evolving: the floor every arm must beat",
    ),
]


def run_key(condition: str, seed: int, generations: int) -> str:
    """Identity of one cached result.

    Includes ``generations`` deliberately: resuming a sweep after changing
    --generations must not reuse a row computed under the old budget, or the
    table would silently mix runs that trained for different lengths.
    """
    return f"{condition}:seed{seed}:g{generations}"


def load_done(path: Path) -> set[str]:
    """Keys already recorded, so a re-run continues rather than repeats."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.add(run_key(row["condition"], row["seed"], row["config"]["generations"]))
    return done


def build_model(dry_run: bool, model: str):
    """The chat model, or a scripted stand-in for a free rehearsal."""
    if not dry_run:
        from meta_evolver.llm.client import build_chat_model

        return build_chat_model(model)

    import sys

    sys.path.insert(0, str(REPO / "examples"))
    from offline_demo import OfflineModel  # noqa: PLC0415

    return OfflineModel()


def run_one(
    condition: Condition,
    seed: int,
    generations: int,
    benchmark: str,
    model: Any,
    max_steps: int,
    rollouts: int,
    out_dir: Path,
) -> dict[str, Any]:
    """One independent run: fresh bank, fresh prompt, its own run id."""
    from meta_evolver.core.evolver import MetaEvolver
    from meta_evolver.graphs.evolution import EvolutionConfig
    from meta_evolver.harness.curriculum import Curriculum
    from meta_evolver.memory.bank import ReasoningMemoryBank

    config = EvolutionConfig(
        generations=generations,
        max_steps=max_steps,
        rollouts_per_task=rollouts,
        seed=seed,
        # Early stopping would truncate arms at different generations and make
        # the curves incomparable, so every arm runs the full budget.
        patience=10**6,
        **condition.overrides,
    )
    evolver = MetaEvolver(
        benchmark=benchmark,
        chat_model=model,
        bank=ReasoningMemoryBank(),
        config=config,
        curriculum=Curriculum(enabled=config.curriculum),
        run_id=f"sweep-{condition.name}-s{seed}",
        use_memory=condition.use_memory,
        telemetry=False,
    )

    started = time.time()
    try:
        reports = evolver.evolve()
    finally:
        evolver.close()

    # Held out twice. At level 0 a competent agent solves nearly everything and
    # every arm ties, which says nothing about the scaffold. The perturbed level
    # is where a missing channel actually costs something -- injected faults, a
    # verification gate, distractor observations -- so that is where an ablation
    # becomes visible. Both are measured on the same held-out tasks for every
    # arm, so an arm that escalated its own curriculum gets no advantage here.
    held_clean = evolver.evaluate(split="eval", curriculum_level=0.0)
    held_hard = evolver.evaluate(split="eval", curriculum_level=HARD_LEVEL)

    return {
        "condition": condition.name,
        "label": condition.label,
        "seed": seed,
        "benchmark": benchmark,
        "generations": [
            {
                "generation": r.generation,
                "pass_rate": r.pass_rate,
                "avg_steps": r.avg_steps,
                "avg_score": r.avg_score,
                "n_errors": r.n_errors,
                "regressions": r.regressions,
                "recoveries": r.recoveries,
                "tokens": r.tokens,
                "curriculum_level": r.curriculum_level,
                "prompt_version": r.prompt_version,
                "memories": r.memories_before + r.memories_added - r.memories_pruned,
            }
            for r in reports
        ],
        "held_out_pass_rate": held_clean["pass_rate"],
        "held_out_errors": held_clean["n_errors"],
        "held_out_hard_pass_rate": held_hard["pass_rate"],
        "held_out_hard_errors": held_hard["n_errors"],
        "hard_level": HARD_LEVEL,
        "final_memories": len(evolver.bank),
        "total_tokens": sum(r.tokens for r in reports),
        "duration_s": round(time.time() - started, 1),
        "config": {
            "generations": generations,
            "max_steps": max_steps,
            "rollouts_per_task": rollouts,
            "use_memory": condition.use_memory,
            **condition.overrides,
        },
    }


def bootstrap_ci(values: list[float], n: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval.

    Preferred over a normal approximation because a pass rate over a handful
    of seeds is bounded, discrete and usually skewed, and a symmetric interval
    on that can run past 1.0 and look absurd.
    """
    import random

    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(12345)
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n)
    )
    lo = means[int((alpha / 2) * n)]
    hi = means[int((1 - alpha / 2) * n) - 1]
    return (lo, hi)


def summarise(path: Path) -> None:
    """Aggregate the JSONL into a table with intervals."""
    if not path.exists():
        print(f"no results at {path}")
        return

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        print("no results yet")
        return

    by_condition: dict[str, list[dict]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)

    order = [c.name for c in CONDITIONS]
    print(f"\n{len(rows)} runs from {path}\n")
    header = (
        f"{'condition':<20} {'n':>2}  {'held-out clean':>18}  "
        f"{'held-out perturbed':>18}  {'mem':>5}  {'tokens':>9}"
    )
    print(header)
    print("-" * len(header))

    for name in sorted(by_condition, key=lambda n: order.index(n) if n in order else 99):
        runs = by_condition[name]
        clean = [r["held_out_pass_rate"] for r in runs]
        hard = [r.get("held_out_hard_pass_rate", float("nan")) for r in runs]
        hard = [h for h in hard if h == h]  # drop NaN from older rows lacking this field
        mems = [r["final_memories"] for r in runs]
        toks = [r["total_tokens"] for r in runs]

        c_lo, c_hi = bootstrap_ci(clean)
        label = runs[0].get("label", name)
        clean_cell = f"{statistics.fmean(clean) * 100:>6.1f}% [{c_lo * 100:4.0f},{c_hi * 100:4.0f}]"
        if hard:
            h_lo, h_hi = bootstrap_ci(hard)
            hard_cell = f"{statistics.fmean(hard) * 100:>6.1f}% [{h_lo * 100:4.0f},{h_hi * 100:4.0f}]"
        else:
            hard_cell = f"{'not measured':>18}"
        print(
            f"{label:<20} {len(runs):>2}  {clean_cell}  {hard_cell}  "
            f"{statistics.fmean(mems):>5.1f}  {statistics.fmean(toks):>9.0f}"
        )

    print(
        f"\nIntervals are 95% percentile bootstrap over seeds. Both columns use "
        f"the same held-out tasks for every arm.\n'clean' is curriculum level 0; "
        f"'perturbed' is level {HARD_LEVEL} (faults + verification gate + distractor "
        f"observations),\nwhich is where a missing channel is expected to cost "
        f"something. If the two columns agree, the benchmark is not\n"
        f"discriminating and the ablation says nothing."
    )

    n_seeds = max(len(v) for v in by_condition.values())
    if n_seeds < 5:
        print(
            f"\nCAUTION: {n_seeds} seed(s) per condition. Intervals this wide do not "
            "support a claim of difference between arms;\nrun more seeds before "
            "putting these in a table."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="devops")
    parser.add_argument("--seeds", type=int, default=5, help="seeds per condition")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--rollouts", type=int, default=1, help="attempts per task (MaTTS)")
    parser.add_argument("--model", default=None, help="chat model id")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help=f"subset of: {', '.join(c.name for c in CONDITIONS)}")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--name", default="ablation", help="sweep name")
    parser.add_argument("--dry-run", action="store_true",
                        help="scripted model, no API calls: proves the grid runs")
    parser.add_argument("--summarise", action="store_true", help="aggregate and exit")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else DEFAULT_OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / ("results_dry.jsonl" if args.dry_run else "results.jsonl")

    if args.summarise:
        summarise(results)
        return 0

    chosen = [c for c in CONDITIONS if not args.conditions or c.name in args.conditions]
    if not chosen:
        print(f"no matching conditions; choose from {[c.name for c in CONDITIONS]}")
        return 1

    done = load_done(results)
    todo = [
        (c, s) for c in chosen for s in range(args.seeds)
        if run_key(c.name, s, args.generations) not in done
    ]

    print(f"sweep '{args.name}' -> {results}")
    print(f"  {len(chosen)} conditions x {args.seeds} seeds x {args.generations} generations")
    print(f"  {len(done)} already done, {len(todo)} to run"
          f"{'  [DRY RUN, no API calls]' if args.dry_run else ''}")
    if not todo:
        summarise(results)
        return 0

    model = build_model(args.dry_run, args.model or "google_genai:gemini-3.8-flash")
    started = time.time()

    for i, (condition, seed) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {condition.name} seed={seed}  ({condition.why})")
        try:
            row = run_one(
                condition, seed, args.generations, args.benchmark, model,
                args.max_steps, args.rollouts, out_dir,
            )
        except Exception as exc:
            # One arm failing must not lose the arms already paid for.
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue

        with results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        gens = row["generations"]
        print(
            f"  final gen {gens[-1]['pass_rate'] * 100:.0f}%  "
            f"held-out {row['held_out_pass_rate'] * 100:.0f}% clean / "
            f"{row['held_out_hard_pass_rate'] * 100:.0f}% perturbed  "
            f"memories {row['final_memories']}  "
            f"{row['total_tokens']} tok  {row['duration_s']}s"
        )

    print(f"\nsweep finished in {(time.time() - started) / 60:.1f} min")
    summarise(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
