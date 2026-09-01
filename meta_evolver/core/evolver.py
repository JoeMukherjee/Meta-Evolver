"""MetaEvolver -- the assembled system.

Everything else in this package is a component; this is the object that wires
them into a working loop and is the only thing most callers need::

    evolver = MetaEvolver(benchmark="devops")
    reports = evolver.evolve(generations=5)
    print(evolver.render_progress())

It owns construction order (client -> embedder -> bank -> graphs), which is
fiddly enough to be worth centralising: the embedder needs the client, the bank
needs the embedder, the episode graph needs the bank, and the evolution graph
needs the episode graph.

``evaluate`` runs the current agent without learning from it, which is what a
held-out number should be measured with.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel

from meta_evolver.adaptive.controller import AdaptiveControllerConfig
from meta_evolver.core.aio import LoopRunner
from meta_evolver.core.registry import get_benchmark
from meta_evolver.core.types import GenerationReport, Trajectory
from meta_evolver.graph_view.recorder import GRAPH_URL_ENV, CausalGraphRecorder
from meta_evolver.graphs.episode import arun_episode, build_episode_graph
from meta_evolver.graphs.evolution import EvolutionConfig, build_evolution_graph
from meta_evolver.harness.curriculum import Curriculum
from meta_evolver.llm.client import DEFAULT_MODEL, build_chat_model, model_name
from meta_evolver.llm.embeddings import Embedder
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.storage.base import DB_URL_ENV
from meta_evolver.storage.checkpoint import (
    describe_checkpointer,
    is_postgres,
    open_checkpointer,
    thread_id,
)
from meta_evolver.telemetry.engine import TelemetryEngine


def _memory_checkpointer() -> Any:
    """The in-process saver, built eagerly because it needs no loop."""
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except ImportError:  # pragma: no cover - older langgraph
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


class MetaEvolver:
    """A benchmark, an agent, and the loop that improves the agent on it."""

    def __init__(
        self,
        benchmark: str | Any = "devops",
        model: str = DEFAULT_MODEL,
        chat_model: BaseChatModel | None = None,
        bank: ReasoningMemoryBank | None = None,
        memory_path: str | Path | None = None,
        db_url: str | None = None,
        checkpoint_url: str | None = None,
        graph_url: str | None = None,
        run_id: str | None = None,
        requests_per_second: float | None = None,
        embed_model: str | None = None,
        embed_dimensions: int | None = None,
        config: EvolutionConfig | None = None,
        controller_config: AdaptiveControllerConfig | None = None,
        curriculum: Curriculum | None = None,
        run_dir: str | Path | None = None,
        telemetry: bool = True,
        use_memory: bool = True,
    ) -> None:
        self.benchmark = get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
        self.model = chat_model or build_chat_model(
            model, requests_per_second=requests_per_second
        )
        # Unique per run unless one is named. With durable checkpointing, a
        # stable id makes `evolve()` *resume* the run that used it -- which is
        # the point, but a surprising default: re-running the same script would
        # silently replay a finished run and report its last state as new.
        # Passing an explicit run_id is how you ask to resume.
        self.run_id = run_id or f"run-{uuid4().hex[:8]}"
        self.resuming = run_id is not None
        # Defaults to the memory database when one is configured: if Postgres
        # is already there, resumable runs are free, and a run that dies in
        # generation four should not have to start from the base prompt.
        self.checkpoint_url = (
            checkpoint_url
            if checkpoint_url is not None
            else (db_url or os.environ.get(DB_URL_ENV) or None)
        )
        self.model_name = model_name(self.model)
        self.config = config or EvolutionConfig()
        self.curriculum = curriculum if curriculum is not None else Curriculum(
            enabled=self.config.curriculum
        )

        # Embeddings are their own model, not the chat model. A scripted or
        # local chat model must not disable retrieval, and a chat-provider
        # switch must not silently re-embed an existing bank into a different
        # vector space.
        self.embedder = Embedder(model=embed_model, dimensions=embed_dimensions)
        if bank is not None:
            self.bank = bank
            # A caller-supplied bank adopts this embedder. Otherwise
            # `MetaEvolver(bank=..., embed_model=...)` would construct the
            # requested embedder, hand it to nothing, and quietly keep the
            # bank's default -- the configuration would appear to apply while
            # every vector came from a different model.
            self.bank.embedder = self.embedder
        elif db_url or os.environ.get(DB_URL_ENV):
            # Namespaced by benchmark so one database serves several without
            # their lessons retrieving each other.
            self.bank = ReasoningMemoryBank.connect(
                db_url,
                namespace=self.benchmark.name,
                embedder=self.embedder,
                dim=embed_dimensions,
            )
        elif memory_path is not None:
            self.bank = ReasoningMemoryBank.load(memory_path, embedder=self.embedder)
        else:
            self.bank = ReasoningMemoryBank(embedder=self.embedder)

        # Observability, not machinery: a Neo4j that is down costs the picture
        # and nothing else. Disabled unless a URL is given or configured.
        self.recorder = CausalGraphRecorder(
            url=graph_url or os.environ.get(GRAPH_URL_ENV) or None,
            run_id=self.run_id,
            benchmark=self.benchmark.name,
        )

        self.telemetry = TelemetryEngine(
            run_dir or Path("runs") / self.benchmark.name, enabled=telemetry
        )
        self.prompt_template = self.benchmark.system_prompt()
        self.prompt_version = "base"
        self.curriculum_level = 0.0
        self.reports: list[GenerationReport] = []

        self._use_memory = use_memory
        self._controller_config = controller_config
        self._checkpoint_cm: Any = None
        # Every sync entry point shares one loop, so a connection pool opened
        # by evolve() is still on a live loop when close() releases it.
        self._loop = LoopRunner()

        # A durable checkpointer owns an async connection pool, so it can only
        # be opened inside a running loop -- which __init__ is not. The
        # in-memory saver has no such constraint, so it is built now and the
        # object is fully usable straight away; a Postgres one is swapped in on
        # the first async call, and the episode graph is rebuilt around it.
        self.checkpointer = None if is_postgres(self.checkpoint_url) else _memory_checkpointer()
        self.episode_graph = self._build_episode_graph()

    def _build_episode_graph(self) -> Any:
        return build_episode_graph(
            model=self.model,
            bank=self.bank if self._use_memory else None,
            retrieval_k=self.config.retrieval_k,
            retrieval_mode=self.config.retrieval_mode,
            controller_config=self._controller_config,
            checkpointer=self.checkpointer,
        )

    async def _ensure_checkpointer(self) -> None:
        """Open the durable checkpointer, once, inside the running loop."""
        if self.checkpointer is not None:
            return
        self._checkpoint_cm = open_checkpointer(self.checkpoint_url)
        self.checkpointer = await self._checkpoint_cm.__aenter__()
        self.episode_graph = self._build_episode_graph()

    # -- the loop ----------------------------------------------------------

    def evolve(
        self,
        generations: int | None = None,
        on_report: Callable[[GenerationReport], None] | None = None,
    ) -> list[GenerationReport]:
        """Synchronous :meth:`aevolve`, for callers with no event loop."""
        return self._loop.run(self.aevolve(generations, on_report))

    async def aevolve(
        self,
        generations: int | None = None,
        on_report: Callable[[GenerationReport], None] | None = None,
    ) -> list[GenerationReport]:
        """Run the outer loop and return one report per generation.

        Resumable in the ordinary sense: the prompt, curriculum level and bank
        persist on this object, so calling this again continues from where the
        last call stopped rather than restarting from the base prompt. With a
        checkpoint URL configured it is resumable in the stronger sense too --
        the graph's own state survives the process.
        """
        await self._ensure_checkpointer()
        total = generations or self.config.generations

        def _record(report: GenerationReport) -> None:
            self.reports.append(report)
            self.telemetry.log_generation(report)
            if on_report is not None:
                on_report(report)

        graph = build_evolution_graph(
            benchmark=self.benchmark,
            episode_graph=self.episode_graph,
            model=self.model,
            bank=self.bank,
            config=self.config,
            curriculum=self.curriculum,
            on_report=_record,
            run_id=self.run_id,
            recorder=self.recorder if self.recorder.enabled else None,
        )

        initial = {
            "benchmark": self.benchmark.name,
            "generation": 0,
            "max_generations": total,
            "prompt_template": self.prompt_template,
            "prompt_version": self.prompt_version,
            "curriculum_level": self.curriculum_level,
            "trajectories": [],
            "reports": [],
            "best_pass_rate": 0.0,
            "generations_without_gain": 0,
        }
        # Each generation costs ~10 supersteps plus one per concurrent rollout.
        # LangGraph's default of 25 would abort a three-generation run midway.
        n_tasks = max(1, len(self.benchmark.task_ids("train")))
        limit = total * (12 + n_tasks * self.config.rollouts_per_task) + 20

        config: dict[str, Any] = {
            "recursion_limit": limit,
            "configurable": {"thread_id": thread_id(self.run_id, "evolution")},
        }
        final = await graph.ainvoke(initial, config=config)

        self.prompt_template = final.get("prompt_template", self.prompt_template)
        self.prompt_version = final.get("prompt_version", self.prompt_version)
        self.curriculum_level = float(final.get("curriculum_level", self.curriculum_level))

        for trajectory in final.get("trajectories", []):
            self.telemetry.log_episode(trajectory, curriculum_level=self.curriculum_level)
        self.telemetry.save_summary(
            {
                "benchmark": self.benchmark.name,
                "model": self.model_name,
                "memory": self.bank.stats(),
                "memory_backend": self.bank.backend,
                "checkpointer": describe_checkpointer(self.checkpointer),
                "causal_graph": self.recorder.describe,
                "rollouts_per_task": self.config.rollouts_per_task,
                "tokens": sum(r.tokens for r in self.reports),
                "prompt_version": self.prompt_version,
                "curriculum": self.curriculum.describe(self.curriculum_level),
            }
        )
        return list(self.reports)

    # -- measurement -------------------------------------------------------

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Synchronous :meth:`aevaluate`."""
        return self._loop.run(self.aevaluate(*args, **kwargs))

    async def aevaluate(
        self,
        split: str = "eval",
        task_ids: Sequence[str] | None = None,
        curriculum_level: float | None = None,
        use_memory: bool = True,
    ) -> dict[str, Any]:
        """Run the current agent on a split without learning from it.

        ``use_memory=False`` gives the honest ablation: same prompt, same
        tasks, no retrieval. The difference between the two is the memory
        bank's actual contribution, which is a claim worth being able to make
        with a number.
        """
        await self._ensure_checkpointer()
        ids = list(task_ids or self.benchmark.task_ids(split))
        level = self.curriculum_level if curriculum_level is None else curriculum_level

        graph = self.episode_graph
        if not use_memory:
            # The ablation needs a bank-free graph but the same checkpointer,
            # so its episodes are resumable and inspectable like any other.
            graph = build_episode_graph(
                model=self.model, bank=None, checkpointer=self.checkpointer
            )

        async def one(task_id: str) -> Trajectory:
            env = self.benchmark.make_env(
                task_id, curriculum_level=level, seed=self.config.seed
            )
            env = self.curriculum.wrap(env, level=level, seed=self.config.seed)
            try:
                return await arun_episode(
                    graph,
                    env=env,
                    task_id=task_id,
                    benchmark=self.benchmark.name,
                    instruction=self.benchmark.instruction_for(task_id),
                    prompt_template=self.prompt_template,
                    prompt_version=self.prompt_version,
                    max_steps=self.config.max_steps,
                    thread_id=thread_id(
                        self.run_id, "evaluate", split, use_memory, task_id
                    ),
                )
            finally:
                env.close()

        # Concurrent: held-out tasks are independent, and a serial evaluation
        # is the slowest thing in a run that is otherwise fully overlapped.
        trajectories: list[Trajectory] = list(await asyncio.gather(*[one(t) for t in ids]))
        for trajectory in trajectories:
            self.telemetry.log_episode(trajectory, curriculum_level=level)

        usable = [t for t in trajectories if t.usable]
        return {
            "split": split,
            "n_tasks": len(trajectories),
            "n_errors": len(trajectories) - len(usable),
            "pass_rate": (
                sum(t.success for t in usable) / len(usable) if usable else 0.0
            ),
            "avg_steps": (
                sum(t.n_steps for t in usable) / len(usable) if usable else 0.0
            ),
            "avg_score": sum(t.score for t in usable) / len(usable) if usable else 0.0,
            "use_memory": use_memory,
            "curriculum_level": level,
            "tokens": sum(t.tokens for t in trajectories),
            "trajectories": trajectories,
        }

    # -- convenience -------------------------------------------------------

    async def aclose(self) -> None:
        """Release the checkpointer's and the bank's connections."""
        cm = self._checkpoint_cm
        if cm is not None:
            self._checkpoint_cm = None
            self.checkpointer = None
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
        self.recorder.close()
        self.bank.close()

    def close(self) -> None:
        """Synchronous :meth:`aclose`. Also shuts the shared loop down."""
        try:
            self._loop.run(self.aclose())
        finally:
            self._loop.close()

    def __enter__(self) -> MetaEvolver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> MetaEvolver:
        await self._ensure_checkpointer()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def render_progress(self) -> str:
        return TelemetryEngine.render_progress(self.reports)

    def save(self, memory_path: str | Path | None = None, prompt_path: str | Path | None = None):
        """Persist what was learned: the bank and the evolved prompt.

        A database-backed bank has already been written through on every add,
        prune and credit pass, so this only flushes the current items and
        reports the backend. Passing ``memory_path`` explicitly exports to a
        file regardless, which is how you snapshot a shared bank.
        """
        out: dict[str, Path | str] = {}
        if self.bank.store is not None and memory_path is None:
            out["memory"] = self.bank.save()
        else:
            target = memory_path or self.bank.path or (self.telemetry.run_dir / "memories.jsonl")
            out["memory"] = self.bank.save(target)
        prompt_target = Path(prompt_path or (self.telemetry.run_dir / "prompt.txt"))
        prompt_target.parent.mkdir(parents=True, exist_ok=True)
        prompt_target.write_text(self.prompt_template, encoding="utf-8")
        out["prompt"] = prompt_target
        return out
