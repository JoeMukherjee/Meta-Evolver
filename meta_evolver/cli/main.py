"""``meta-evolver`` command line.

Four verbs, matching the four things you actually do with this system:

    benchmarks   what can I run against
    run          one episode, printed step by step
    evolve       the outer loop, streaming a generation table
    ablate       does the memory bank actually help

``evolve`` streams each generation's line as it completes rather than printing
a table at the end. A run is minutes to hours of model calls, and a progress
display that only appears on success is not a progress display.
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from meta_evolver.adaptive.controller import AdaptiveControllerConfig
from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.core.registry import get_benchmark, list_benchmarks
from meta_evolver.core.types import GenerationReport
from meta_evolver.graphs.evolution import EvolutionConfig

app = typer.Typer(
    help="Meta-Evolver: a LangGraph engine for agents that improve on any benchmark.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

DEFAULT_MODEL = "gemini/gemini-3-flash"


@app.command()
def benchmarks() -> None:
    """List registered benchmarks and their task splits."""
    table = Table(title="Registered benchmarks")
    table.add_column("name", style="bold cyan")
    table.add_column("train", justify="right")
    table.add_column("eval", justify="right")
    table.add_column("description")

    for name in list_benchmarks():
        try:
            bench = get_benchmark(name)
            table.add_row(
                name,
                str(len(bench.task_ids("train"))),
                str(len(bench.task_ids("eval"))),
                bench.description,
            )
        except Exception as exc:  # an optional dependency is missing
            table.add_row(name, "-", "-", f"[red]unavailable: {exc}[/red]")

    console.print(table)


@app.command()
def run(
    benchmark: str = typer.Option("devops", help="Registered benchmark name."),
    task: str | None = typer.Option(None, help="Task id. Defaults to the first train task."),
    model: str = typer.Option(DEFAULT_MODEL, help="Any litellm model id."),
    memory: Path | None = typer.Option(None, help="Memory bank JSONL to retrieve from."),
    max_steps: int = typer.Option(15),
    curriculum_level: float = typer.Option(0.0, help="0 = clean env, 1 = fully adversarial."),
    no_memory: bool = typer.Option(False, "--no-memory", help="Disable retrieval."),
) -> None:
    """Run a single episode and print the trajectory."""
    bench = get_benchmark(benchmark)
    task_id = task or (bench.task_ids("train") or bench.task_ids("all"))[0]

    evolver = MetaEvolver(
        benchmark=bench,
        model=model,
        memory_path=memory,
        config=EvolutionConfig(max_steps=max_steps),
        use_memory=not no_memory,
        telemetry=False,
    )
    evolver.curriculum_level = curriculum_level
    result = evolver.evaluate(task_ids=[task_id], curriculum_level=curriculum_level,
                              use_memory=not no_memory)
    trajectory = result["trajectories"][0]

    console.print(trajectory.render(max_steps=max_steps))
    if trajectory.error:
        console.print(f"[red]error:[/red] {trajectory.error}")
        raise typer.Exit(code=1)
    console.print(
        f"\n[bold]{'PASS' if trajectory.success else 'FAIL'}[/bold] "
        f"score={trajectory.score:.2f} steps={trajectory.n_steps}"
    )


@app.command()
def evolve(
    benchmark: str = typer.Option("devops", help="Registered benchmark name."),
    model: str = typer.Option(DEFAULT_MODEL, help="Any litellm model id."),
    generations: int = typer.Option(5),
    max_steps: int = typer.Option(15),
    memory: Path | None = typer.Option(None, help="Memory bank JSONL to load and update."),
    run_dir: Path | None = typer.Option(None, help="Where telemetry is written."),
    patience: int = typer.Option(6, help="Steps without progress before memory is evicted."),
    retrieval_k: int = typer.Option(4),
    no_curriculum: bool = typer.Option(False, "--no-curriculum"),
    no_prompt_evolution: bool = typer.Option(False, "--no-prompt-evolution"),
    no_validation: bool = typer.Option(
        False,
        "--no-validation",
        help="Adopt prompt candidates without a held-out check. Faster, and weaker.",
    ),
) -> None:
    """Run the evolution loop, streaming one line per generation."""
    config = EvolutionConfig(
        generations=generations,
        max_steps=max_steps,
        retrieval_k=retrieval_k,
        curriculum=not no_curriculum,
        optimize_prompt=not no_prompt_evolution,
        validate_prompt=not no_validation,
    )
    evolver = MetaEvolver(
        benchmark=benchmark,
        model=model,
        memory_path=memory,
        run_dir=run_dir,
        config=config,
        controller_config=AdaptiveControllerConfig(patience=patience),
    )

    console.print(
        f"[bold]{evolver.benchmark.name}[/bold] via [cyan]{evolver.client.model}[/cyan] "
        f"-- {generations} generations, {len(evolver.benchmark.task_ids('train'))} train tasks"
    )
    if no_validation:
        console.print("[yellow]validation disabled: prompt changes are unmeasured[/yellow]")

    def show(report: GenerationReport) -> None:
        console.print(report.render())

    try:
        reports = evolver.evolve(on_report=show)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted; saving what was learned so far[/yellow]")
        reports = evolver.reports

    console.print()
    console.print(evolver.render_progress())

    saved = evolver.save(memory_path=memory)
    console.print(
        f"\nmemory  -> {saved['memory']}  ({evolver.bank.stats()})\n"
        f"prompt  -> {saved['prompt']}  (version {evolver.prompt_version})\n"
        f"run     -> {evolver.telemetry.run_dir}"
    )
    if not reports:
        raise typer.Exit(code=1)


@app.command()
def ablate(
    benchmark: str = typer.Option("devops"),
    model: str = typer.Option(DEFAULT_MODEL),
    memory: Path = typer.Option(..., help="Memory bank JSONL to evaluate."),
    split: str = typer.Option("eval"),
    max_steps: int = typer.Option(15),
) -> None:
    """Measure what the memory bank is actually worth.

    Same prompt, same tasks, retrieval on versus off. The gap is the bank's
    contribution -- the number worth quoting, and the one a run that only
    reports "with memory" cannot support.
    """
    evolver = MetaEvolver(
        benchmark=benchmark,
        model=model,
        memory_path=memory,
        config=EvolutionConfig(max_steps=max_steps),
        telemetry=False,
    )
    console.print(f"bank: {evolver.bank.stats()}\n")

    with_memory = evolver.evaluate(split=split, use_memory=True)
    without = evolver.evaluate(split=split, use_memory=False)

    table = Table(title=f"Memory ablation on {benchmark}/{split}")
    table.add_column("condition")
    table.add_column("pass rate", justify="right")
    table.add_column("avg steps", justify="right")
    table.add_column("errors", justify="right")
    for label, result in (("with memory", with_memory), ("no memory", without)):
        table.add_row(
            label,
            f"{result['pass_rate'] * 100:.1f}%",
            f"{result['avg_steps']:.2f}",
            str(result["n_errors"]),
        )
    console.print(table)

    delta = (with_memory["pass_rate"] - without["pass_rate"]) * 100
    console.print(f"[bold]memory delta: {delta:+.1f} points[/bold]")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)


if __name__ == "__main__":
    main()
