"""CLI Entry point for Meta-Evolver."""
import typer
from rich.console import Console

app = typer.Typer(help="Meta-Evolver: Environment-Harness Evolutionary Meta-Agent")
console = Console()

@app.command()
def version():
    """Show Meta-Evolver version."""
    console.print("[bold green]Meta-Evolver[/bold green] version 0.1.0")

@app.command()
def eval(
    bank: str = typer.Option(..., help="Path to reasoning bank JSONL"),
    adaptive: bool = typer.Option(True, help="Enable Adaptive Exploration Controller"),
    patience: int = typer.Option(6, help="Stagnation eviction patience"),
):
    """Run evaluation on benchmark environment."""
    console.print(f"[bold blue]Evaluating[/bold blue] bank: {bank} (adaptive={adaptive}, patience={patience})")

if __name__ == "__main__":
    app()
