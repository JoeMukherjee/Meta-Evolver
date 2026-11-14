"""Measure the three arms on one benchmark and write a results file.

    python scripts/baseline.py --benchmark locomo \
        --data-path ../locomo/data/locomo10.json --limit 50 --seeds 0,1

Any benchmark works; the ones that need a dataset take ``--data-path``, and
``--bench-arg k=v`` reaches any other constructor argument. Nothing here is
specific to a benchmark on purpose -- the table it writes is the thing every
later phase of this project is compared against, so it has to be re-runnable
against anything the registry knows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_evolver.cli.main import _bench_kwargs  # noqa: E402
from meta_evolver.core.registry import get_benchmark  # noqa: E402
from meta_evolver.eval.baseline import ARMS, run_baseline, summarize  # noqa: E402
from meta_evolver.llm.client import DEFAULT_MODEL, build_chat_model  # noqa: E402

console = Console()


def main(
    benchmark: str = typer.Option(..., help="Registered benchmark name."),
    model: str = typer.Option(DEFAULT_MODEL, help="Any LangChain model id."),
    data_path: Path | None = typer.Option(None, "--data-path"),
    limit: int | None = typer.Option(50, "--limit", help="Stratified task subset size."),
    bench_arg: list[str] = typer.Option([], "--bench-arg", help="key=value. Repeatable."),
    seeds: str = typer.Option("0,1", help="Comma-separated run seeds."),
    arms: str = typer.Option(",".join(ARMS), help="Comma-separated arms to run."),
    generations: int = typer.Option(2, help="Training generations for the learning arms."),
    max_steps: int = typer.Option(15),
    token_ceiling: int | None = typer.Option(None, help="Stop starting new arms once this many tokens are spent in total."),
    tasks_per_generation: int = typer.Option(0, help="Cap training tasks per generation (0 = all). The main lever on training cost."),
    max_total_tokens: int | None = typer.Option(None, help="Per-arm training token budget."),
    requests_per_second: float | None = typer.Option(None, help="Throttle model calls."),
    out_dir: Path = typer.Option(Path("results")),
    no_trace: bool = typer.Option(False, "--no-trace", help="Skip local span traces."),
) -> None:
    kwargs = _bench_kwargs(data_path, limit, bench_arg)
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]

    # Fail before spending anything if the benchmark cannot even be built.
    probe = get_benchmark(benchmark, **kwargs)
    console.print(
        f"[bold]{benchmark}[/bold] via [cyan]{model}[/cyan] -- "
        f"{len(probe.task_ids('train'))} train / {len(probe.task_ids('eval'))} eval tasks, "
        f"arms={arm_list}, seeds={seed_list}"
    )

    def show(row: dict) -> None:
        console.print(
            f"  seed {row['seed']} {row['arm']:<12} "
            f"pass {row['pass_rate'] * 100:5.1f}%  score {row['avg_score']:.3f}  "
            f"{(row['train_tokens'] + row['eval_tokens']) / 1000:.0f}k tok  "
            f"{row['duration_s']:.0f}s"
        )

    target = run_baseline(
        benchmark_name=benchmark,
        benchmark_factory=lambda: get_benchmark(benchmark, **kwargs),
        model_factory=lambda: build_chat_model(model, requests_per_second=requests_per_second),
        model_name=model,
        seeds=seed_list,
        arms=arm_list,
        generations=generations,
        max_steps=max_steps,
        max_total_tokens=max_total_tokens,
        out_dir=out_dir,
        trace=not no_trace,
        on_row=show,
        tasks_per_generation=tasks_per_generation,
        token_ceiling=token_ceiling,
    )

    import json

    summary = summarize(json.loads(target.read_text(encoding="utf-8"))["rows"])
    table = Table(title=f"{benchmark} baseline")
    table.add_column("arm", style="bold cyan")
    table.add_column("pass rate", justify="right")
    table.add_column("+/- stdev", justify="right")
    table.add_column("avg score", justify="right")
    table.add_column("tokens", justify="right")
    for arm, s in summary.items():
        table.add_row(
            arm,
            f"{s['pass_rate_mean'] * 100:.1f}%",
            f"{s['pass_rate_stdev'] * 100:.1f}",
            f"{s['avg_score_mean']:.3f}",
            f"{s['tokens_mean'] / 1000:.0f}k",
        )
    console.print(table)
    console.print(f"\nwrote [bold]{target}[/bold]")


if __name__ == "__main__":
    typer.run(main)
