"""Event-loop plumbing for the synchronous entry points.

The graphs are async; most callers are not. Bridging the two naively --
``asyncio.run`` per public method -- breaks in two ways that only show up once
a durable checkpointer is involved.

**A pool outlives the call that opened it.** ``asyncio.run`` closes its loop on
return, so a connection pool opened inside ``evolve()`` is bound to a dead loop
by the time ``close()`` wants to release it. Every sync entry point on one
object therefore shares one :class:`asyncio.Runner`, kept for the object's
lifetime.

**Windows defaults to the wrong loop.** ``ProactorEventLoop`` is the default
there and psycopg refuses to run async on it. The selector loop is the
supported one, so that is what gets built.

Async callers who want durable checkpointing on Windows need the same loop, and
cannot get it from a plain ``asyncio.run``. :func:`selector_loop_factory` is
exported for exactly that::

    asyncio.run(main(), loop_factory=selector_loop_factory())
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def selector_loop_factory() -> Callable[[], asyncio.AbstractEventLoop]:
    """A loop factory that psycopg can use on every platform.

    Windows only: elsewhere the default loop is already selector-based, and
    forcing one would discard whatever policy the host application chose.
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop
    return asyncio.new_event_loop


class LoopRunner:
    """One persistent event loop, shared by an object's sync entry points."""

    def __init__(self) -> None:
        self._runner: asyncio.Runner | None = None

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` on this runner's loop.

        Raises if a loop is already running: silently nesting one, or blocking
        the caller's, are both worse than saying so. Inside async code, await
        the ``a``-prefixed method directly.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            raise RuntimeError(
                "this synchronous method was called from a running event loop; "
                "await the async form instead (evolve -> aevolve, and so on)"
            )

        if self._runner is None:
            self._runner = asyncio.Runner(loop_factory=selector_loop_factory())
        return self._runner.run(coro)

    def close(self) -> None:
        if self._runner is not None:
            try:
                self._runner.close()
            except Exception:
                pass
            self._runner = None
