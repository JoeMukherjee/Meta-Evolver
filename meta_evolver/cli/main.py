"""``meta-evolver`` command line.

Four verbs, matching the four things you actually do with this system:

    benchmarks   what can I run against
    run          one episode, printed step by step
    evolve       the outer loop, streaming a generation table
    ablate       does the memory bank actually help
    graph        query the causal graph of a run, or print the Cypher to

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
from meta_evolver.graph_view import SAVED_QUERIES, CausalGraphRecorder
from meta_evolver.graph_view.recorder import GRAPH_URL_ENV
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import DEFAULT_EMBED_MODEL, DEFAULT_MODEL
from meta_evolver.storage.base import DB_URL_ENV

app = typer.Typer(
    help="Meta-Evolver: a LangGraph engine for agents that improve on any benchmark.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


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
    model: str = typer.Option(DEFAULT_MODEL, help="Any LangChain model id, e.g. openai:gpt-4.1."),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, help="Embedding model."),
    embed_dimensions: int = typer.Option(768, help="Embedding width (128-3072)."),
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
        embed_model=embed_model,
        embed_dimensions=embed_dimensions,
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
    model: str = typer.Option(DEFAULT_MODEL, help="Any LangChain model id, e.g. openai:gpt-4.1."),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, help="Embedding model."),
    embed_dimensions: int = typer.Option(
        768, help="Embedding width (128-3072). Lower is smaller and faster."
    ),
    generations: int = typer.Option(5),
    max_steps: int = typer.Option(15),
    memory: Path | None = typer.Option(None, help="Memory bank JSONL to load and update."),
    db_url: str | None = typer.Option(
        None,
        "--db-url",
        help=(
            "Postgres or MongoDB connection string for the memory bank. "
            f"Defaults to ${DB_URL_ENV}; without either, a JSONL file is used."
        ),
    ),
    graph_url: str | None = typer.Option(
        None,
        "--graph-url",
        help=(
            "Neo4j bolt URL to record the run's causal graph to. "
            f"Defaults to ${GRAPH_URL_ENV}."
        ),
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help=(
            "Name this run. Reusing a name RESUMES that run from its last "
            "checkpoint; omit it for a fresh run."
        ),
    ),
    rollouts_per_task: int = typer.Option(
        1, help="Attempts per task (MaTTS). Above 1, disagreeing attempts feed induction."
    ),
    requests_per_second: float | None = typer.Option(
        None, help="Throttle model calls. Worth setting when rollouts run concurrently."
    ),
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
        rollouts_per_task=rollouts_per_task,
        curriculum=not no_curriculum,
        optimize_prompt=not no_prompt_evolution,
        validate_prompt=not no_validation,
    )
    evolver = MetaEvolver(
        benchmark=benchmark,
        model=model,
        embed_model=embed_model,
        embed_dimensions=embed_dimensions,
        memory_path=memory,
        db_url=db_url,
        graph_url=graph_url,
        run_id=run_id,
        requests_per_second=requests_per_second,
        run_dir=run_dir,
        config=config,
        controller_config=AdaptiveControllerConfig(patience=patience),
    )

    console.print(
        f"[bold]{evolver.benchmark.name}[/bold] via [cyan]{evolver.model_name}[/cyan] "
        f"-- {generations} generations, {len(evolver.benchmark.task_ids('train'))} train tasks"
    )
    console.print(f"run id:     [cyan]{evolver.run_id}[/cyan]"
                  + ("  [yellow](resuming)[/yellow]" if evolver.resuming else ""))
    console.print(f"memory:     [cyan]{evolver.bank.backend}[/cyan]")
    if evolver.recorder.enabled:
        console.print(
            f"graph:      [cyan]{evolver.recorder.describe}[/cyan]  "
            f"watch it at [bold]{evolver.recorder.browser_url()}[/bold]"
        )
    elif graph_url or GRAPH_URL_ENV in __import__("os").environ:
        console.print(f"[yellow]causal graph {evolver.recorder.describe}[/yellow]")
    console.print(
        f"embeddings: [cyan]{embed_model}[/cyan] at {embed_dimensions} dims"
        f"{'' if evolver.embedder.remote_available else ' [yellow](unavailable; using local encoder)[/yellow]'}"
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
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, help="Embedding model."),
    embed_dimensions: int = typer.Option(768, help="Embedding width (128-3072)."),
    memory: Path | None = typer.Option(None, help="Memory bank JSONL to evaluate."),
    db_url: str | None = typer.Option(None, "--db-url", help="Database bank to evaluate."),
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
        embed_model=embed_model,
        embed_dimensions=embed_dimensions,
        memory_path=memory,
        db_url=db_url,
        config=EvolutionConfig(max_steps=max_steps),
        telemetry=False,
    )
    console.print(f"bank: {evolver.bank.backend}")
    console.print(f"      {evolver.bank.stats()}\n")

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


@app.command()
def graph(
    query: str = typer.Argument(
        "overview", help=f"One of: {', '.join(SAVED_QUERIES)}. Use 'list' to see them all."
    ),
    run_id: str = typer.Option(
        "", "--run-id", help="Which run to query. Not needed for 'list'."
    ),
    graph_url: str | None = typer.Option(None, "--graph-url", help="Neo4j bolt URL."),
    generation: int = typer.Option(0, help="For queries that take one."),
    cypher: bool = typer.Option(False, "--cypher", help="Print the Cypher instead of running it."),
) -> None:
    """Query a run's causal graph -- provenance, regressions, prompt lineage.

    Each saved query answers something the flat telemetry makes you write a
    join for. ``--cypher`` prints it instead of running it, so it can be
    pasted into the Neo4j browser and explored as a picture.
    """
    if query == "list":
        table = Table(title="Saved queries")
        table.add_column("name", style="bold cyan")
        table.add_column("answers")
        for name, (description, _) in SAVED_QUERIES.items():
            table.add_row(name, description)
        console.print(table)
        return

    if not run_id:
        console.print("[red]--run-id is required[/red] (the run's name, printed when it started)")
        raise typer.Exit(code=1)

    if query not in SAVED_QUERIES:
        console.print(f"[red]unknown query {query!r}[/red]; try: {', '.join(SAVED_QUERIES)}")
        raise typer.Exit(code=1)

    description, statement = SAVED_QUERIES[query]
    console.print(f"[dim]{description}[/dim]")

    if cypher:
        console.print(f"\n:param run_id => '{run_id}';")
        if "$generation" in statement:
            console.print(f":param generation => {generation};")
        console.print(f"\n{statement}")
        return

    recorder = CausalGraphRecorder(url=graph_url, run_id=run_id)
    if not recorder.enabled:
        console.print(f"[red]causal graph {recorder.describe}[/red]")
        raise typer.Exit(code=1)

    rows = recorder.query(statement, generation=generation)
    recorder.close()

    if not rows:
        console.print("[yellow]no rows -- is the run id right?[/yellow]")
        return

    table = Table()
    for column in rows[0]:
        table.add_column(column)
    for row in rows[:50]:
        table.add_row(*[str(row.get(c, ""))[:70] for c in rows[0]])
    console.print(table)
    if len(rows) > 50:
        console.print(f"[dim]... {len(rows) - 50} more rows[/dim]")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)


if __name__ == "__main__":
    main()
