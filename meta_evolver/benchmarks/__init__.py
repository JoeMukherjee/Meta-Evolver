"""Benchmark adapters -- the seam where your own tasks plug in."""
from meta_evolver.benchmarks.base import BenchmarkAdapter, tool_schema
from meta_evolver.benchmarks.custom import (
    FunctionBenchmark,
    FunctionEnv,
    Task,
    ToolCallRecord,
)
from meta_evolver.benchmarks.external import (
    ExternalBenchmark,
    ExternalEnvAdapter,
    TextEnvAdapter,
)

__all__ = [
    "BenchmarkAdapter",
    "ExternalBenchmark",
    "ExternalEnvAdapter",
    "FunctionBenchmark",
    "FunctionEnv",
    "Task",
    "TextEnvAdapter",
    "ToolCallRecord",
    "tool_schema",
]
