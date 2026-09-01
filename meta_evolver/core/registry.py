"""Benchmark registry.

Adding a benchmark to Meta-Evolver is one decorator. The CLI, the evolution
graph, and the telemetry layer all discover benchmarks through this registry,
so nothing else has to be edited when a new one lands.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from meta_evolver.benchmarks.base import BenchmarkAdapter

_REGISTRY: dict[str, type] = {}

#: Benchmarks shipped in-tree, imported lazily on first lookup so that an
#: optional dependency (a real ALFWorld install, say) never breaks `import
#: meta_evolver` for someone using a different benchmark.
_BUILTIN_MODULES = (
    "meta_evolver.benchmarks.devops",
    "meta_evolver.benchmarks.textworld",
)

T = TypeVar("T", bound=type)


def register_benchmark(name: str) -> Callable[[T], T]:
    """Class decorator: make an adapter reachable as ``name``."""

    def deco(cls: T) -> T:
        cls.name = name  # type: ignore[attr-defined]
        _REGISTRY[name] = cls
        return cls

    return deco


def _load_builtins() -> None:
    for mod in _BUILTIN_MODULES:
        try:
            importlib.import_module(mod)
        except Exception:  # pragma: no cover - optional deps
            continue


def get_benchmark(name: str, **kwargs) -> BenchmarkAdapter:
    """Instantiate a registered benchmark adapter."""
    if name not in _REGISTRY:
        _load_builtins()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown benchmark {name!r}. Registered: {sorted(_REGISTRY) or '(none)'}. "
            "Register your own with @register_benchmark('my-bench')."
        )
    return _REGISTRY[name](**kwargs)  # type: ignore[return-value]


def list_benchmarks() -> list[str]:
    _load_builtins()
    return sorted(_REGISTRY)
