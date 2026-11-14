"""A persistent dollar budget that stops model calls -- not just a report of them.

Why a ledger and not a per-run flag
-----------------------------------
A spending limit set for "this experiment" is gone the moment the process
exits, and a development cycle is dozens of experiments across sessions. The
ledger is a small JSON file outside the repository that accumulates every
token any run spends, so the cap applies to the *whole cycle*.

Why it raises a ``BaseException``
---------------------------------
The engine deliberately turns provider failures into data: an episode that
hits an API error is recorded as an infrastructure error and the run carries
on. That is right for a rate limit and exactly wrong for a budget -- a guard
that raised an ordinary exception would be swallowed episode by episode while
the run kept looping. :class:`BudgetExceeded` derives from ``BaseException``
(like ``KeyboardInterrupt``) so the ``except Exception`` handlers in between
cannot absorb it, and it propagates to the top and ends the process.

Accounting, stated plainly
--------------------------
Usage is metered as *total tokens*, converted at one blended ``usd_per_mtok``
rate. Providers bill input and output at different rates, so the blended rate
is an assumption the owner of the budget sets, with a safety margin -- the
ledger is a conservative estimate, not an invoice. Embedding calls are not
counted. A ``reserve_usd`` is held back because concurrent in-flight calls
cannot be recalled once the cap is reached.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

LEDGER_ENV = "META_EVOLVER_SPEND_LEDGER"
DEFAULT_LEDGER = Path.home() / ".meta_evolver" / "spend.json"


class BudgetExceeded(BaseException):
    """Raised when the next model call would exceed the budget.

    A ``BaseException`` on purpose -- see the module docstring."""


class SpendLedger:
    def __init__(
        self,
        path: str | Path,
        cap_usd: float = 40.0,
        usd_per_mtok: float = 1.31,
        reserve_usd: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.cap_usd, self.usd_per_mtok, self.reserve_usd = cap_usd, usd_per_mtok, reserve_usd
        self.tokens = 0
        self.entries: list[dict[str, Any]] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.cap_usd = float(data.get("cap_usd", cap_usd))
            self.usd_per_mtok = float(data.get("usd_per_mtok", usd_per_mtok))
            self.reserve_usd = float(data.get("reserve_usd", reserve_usd))
            self.tokens = int(data.get("tokens", 0))
            self.entries = list(data.get("entries", []))

    # -- accounting ------------------------------------------------------------

    @property
    def spent_usd(self) -> float:
        return self.tokens / 1e6 * self.usd_per_mtok

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.spent_usd

    @property
    def remaining_tokens(self) -> int:
        return max(0, int(self.remaining_usd / self.usd_per_mtok * 1e6))

    def exhausted(self) -> bool:
        """True once the next call must be refused (cap minus the reserve)."""
        return self.spent_usd >= self.cap_usd - self.reserve_usd

    def add(self, tokens: int, note: str = "", flush: bool = True) -> None:
        with self._lock:
            self.tokens += int(tokens)
            if note:
                self.entries.append({"t": round(time.time()), "tokens": int(tokens), "note": note})
            if flush:
                self.save()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "cap_usd": self.cap_usd, "usd_per_mtok": self.usd_per_mtok,
                        "reserve_usd": self.reserve_usd, "tokens": self.tokens,
                        "entries": self.entries[-200:],
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
            tmp.replace(self.path)  # atomic: a crash never leaves half a ledger

    def summary(self) -> str:
        return (
            f"spent ${self.spent_usd:.2f} of ${self.cap_usd:.2f} "
            f"({self.tokens / 1e6:.2f}M tokens at ${self.usd_per_mtok}/M); "
            f"${max(0.0, self.remaining_usd):.2f} left (~{self.remaining_tokens / 1e6:.1f}M tokens)"
        )


class BudgetGuard(BaseCallbackHandler):
    """LangChain callback: refuses calls once the ledger is exhausted and
    records the tokens of every call that completes."""

    raise_error = True  # LangChain swallows callback errors unless told not to
    run_inline = True

    def __init__(self, ledger: SpendLedger) -> None:
        self.ledger = ledger

    def _check(self) -> None:
        if self.ledger.exhausted():
            raise BudgetExceeded(f"budget exhausted -- {self.ledger.summary()}")

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self._check()

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self._check()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        total = 0
        for generations in getattr(response, "generations", []) or []:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                total += int(usage.get("total_tokens") or 0)
        if total:
            self.ledger.add(total)


def default_ledger() -> SpendLedger | None:
    """The ledger every model built by :func:`~meta_evolver.llm.client.
    build_chat_model` is guarded by -- if one exists.

    Looked up from ``$META_EVOLVER_SPEND_LEDGER``, else ``~/.meta_evolver/
    spend.json``. No file, no guard: a fresh checkout is not silently capped,
    but once a ledger has been created every later run is covered
    automatically, in any session."""
    path = os.environ.get(LEDGER_ENV)
    candidate = Path(path) if path else DEFAULT_LEDGER
    return SpendLedger(candidate) if candidate.exists() else None
