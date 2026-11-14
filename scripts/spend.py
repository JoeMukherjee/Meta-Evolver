"""Show or adjust the persistent spend ledger.

    python scripts/spend.py                      # where the budget stands
    python scripts/spend.py --add-tokens 500000 --note "manual correction"
    python scripts/spend.py --set-spent-usd 18.40   # reconcile with your billing console
    python scripts/spend.py --cap 40 --rate 1.31

The ledger (``~/.meta_evolver/spend.json``, or ``$META_EVOLVER_SPEND_LEDGER``)
guards every model built by ``build_chat_model``; see ``meta_evolver/llm/budget.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_evolver.llm.budget import DEFAULT_LEDGER, LEDGER_ENV, SpendLedger  # noqa: E402


def main(
    add_tokens: int = typer.Option(0, help="Record extra tokens spent."),
    note: str = typer.Option("", help="Why (stored with the entry)."),
    set_spent_usd: float | None = typer.Option(None, help="Reset the total to this dollar figure, e.g. from your billing console."),
    cap: float | None = typer.Option(None, help="New total cap in USD."),
    rate: float | None = typer.Option(None, help="New blended USD per million tokens."),
) -> None:
    import os

    path = Path(os.environ.get(LEDGER_ENV) or DEFAULT_LEDGER)
    led = SpendLedger(path)
    if cap is not None:
        led.cap_usd = cap
    if rate is not None:
        led.usd_per_mtok = rate
    if set_spent_usd is not None:
        led.tokens = int(set_spent_usd / led.usd_per_mtok * 1e6)
        led.entries.append({"t": 0, "tokens": 0, "note": f"reconciled to ${set_spent_usd:.2f}: {note}".strip()})
    if add_tokens:
        led.add(add_tokens, note or "manual", flush=False)
    led.save()
    typer.echo(f"{path}\n{led.summary()}")


if __name__ == "__main__":
    typer.run(main)
