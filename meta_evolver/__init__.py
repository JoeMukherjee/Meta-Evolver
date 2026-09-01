"""Meta-Evolver: a LangGraph engine for agents that improve on any benchmark.

Three loops, composed:

* the **episode graph** runs one rollout with retrieval and adaptive control;
* the **evolution graph** runs generations that curate memory, evolve the
  system prompt against a held-out split, and escalate environment difficulty;
* the **benchmark adapter** is the seam where your own tasks plug in.

Quick start::

    from meta_evolver import MetaEvolver

    evolver = MetaEvolver(benchmark="devops", model="gemini/gemini-3-flash")
    evolver.evolve(generations=4)
    print(evolver.render_progress())
"""
from meta_evolver.adaptive.controller import (
    AdaptiveControllerConfig,
    AdaptiveExplorationController,
)
from meta_evolver.benchmarks.base import BenchmarkAdapter, tool_schema
from meta_evolver.core.aio import selector_loop_factory
from meta_evolver.core.env import ActionableEnv, EnvHarness
from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.core.registry import (
    get_benchmark,
    list_benchmarks,
    register_benchmark,
)
from meta_evolver.core.rules import Rules
from meta_evolver.core.types import (
    Action,
    EvaluationResult,
    GenerationReport,
    MemoryItem,
    Observation,
    StepRecord,
    TaskSpec,
    Trajectory,
)
from meta_evolver.graphs.episode import build_episode_graph, run_episode
from meta_evolver.graphs.evolution import EvolutionConfig, build_evolution_graph
from meta_evolver.harness.curriculum import Curriculum
from meta_evolver.llm.client import ScriptedChatModel, build_chat_model, tool_call_message
from meta_evolver.memory.bank import ReasoningMemoryBank

__version__ = "0.2.0"

__all__ = [
    "ActionableEnv",
    "Action",
    "AdaptiveControllerConfig",
    "AdaptiveExplorationController",
    "BenchmarkAdapter",
    "Curriculum",
    "EnvHarness",
    "EvaluationResult",
    "EvolutionConfig",
    "GenerationReport",
    "MemoryItem",
    "MetaEvolver",
    "Observation",
    "ReasoningMemoryBank",
    "Rules",
    "ScriptedChatModel",
    "StepRecord",
    "TaskSpec",
    "Trajectory",
    "build_chat_model",
    "build_episode_graph",
    "build_evolution_graph",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
    "run_episode",
    "selector_loop_factory",
    "tool_call_message",
    "tool_schema",
    "__version__",
]
