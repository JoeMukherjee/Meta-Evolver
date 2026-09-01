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

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from meta_evolver.adaptive.controller import AdaptiveControllerConfig
from meta_evolver.core.registry import get_benchmark
from meta_evolver.core.types import GenerationReport, Trajectory
from meta_evolver.graphs.episode import build_episode_graph, run_episode
from meta_evolver.graphs.evolution import EvolutionConfig, build_evolution_graph
from meta_evolver.harness.curriculum import Curriculum
from meta_evolver.llm.client import DEFAULT_MODEL, build_chat_model, model_name
from meta_evolver.llm.embeddings import Embedder
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.telemetry.engine import TelemetryEngine


class MetaEvolver:
    """A benchmark, an agent, and the loop that improves the agent on it."""

    def __init__(
        self,
        benchmark: str | Any = "devops",
        model: str = DEFAULT_MODEL,
        chat_model: BaseChatModel | None = None,
        bank: ReasoningMemoryBank | None = None,
        memory_path: str | Path | None = None,
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
        self.model = chat_model or build_chat_model(model)
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
        elif memory_path is not None:
            self.bank = ReasoningMemoryBank.load(memory_path, embedder=self.embedder)
        else:
            self.bank = ReasoningMemoryBank(embedder=self.embedder)

        self.telemetry = TelemetryEngine(
            run_dir or Path("runs") / self.benchmark.name, enabled=telemetry
        )
        self.prompt_template = self.benchmark.system_prompt()
        self.prompt_version = "base"
        self.curriculum_level = 0.0
        self.reports: list[GenerationReport] = []

        self.episode_graph = build_episode_graph(
            model=self.model,
            bank=self.bank if use_memory else None,
            retrieval_k=self.config.retrieval_k,
            retrieval_mode=self.config.retrieval_mode,
            controller_config=controller_config,
        )

    # -- the loop ----------------------------------------------------------

    def evolve(
        self,
        generations: int | None = None,
        on_report: Callable[[GenerationReport], None] | None = None,
    ) -> list[GenerationReport]:
        """Run the outer loop and return one report per generation.

        Resumable in the ordinary sense: the prompt, curriculum level and bank
        persist on this object, so calling ``evolve`` again continues from
        where the last call stopped rather than restarting from the base
        prompt.
        """
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
        # Each generation costs ~10 supersteps plus one per concurrent task.
        # LangGraph's default of 25 would abort a three-generation run midway.
        n_tasks = max(1, len(self.benchmark.task_ids("train")))
        limit = total * (12 + n_tasks) + 20

        final = graph.invoke(initial, config={"recursion_limit": limit})

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
                "prompt_version": self.prompt_version,
                "curriculum": self.curriculum.describe(self.curriculum_level),
            }
        )
        return list(self.reports)

    # -- measurement -------------------------------------------------------

    def evaluate(
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
        ids = list(task_ids or self.benchmark.task_ids(split))
        level = self.curriculum_level if curriculum_level is None else curriculum_level

        graph = self.episode_graph
        if not use_memory:
            graph = build_episode_graph(model=self.model, bank=None)

        trajectories: list[Trajectory] = []
        for task_id in ids:
            env = self.benchmark.make_env(
                task_id, curriculum_level=level, seed=self.config.seed
            )
            env = self.curriculum.wrap(env, level=level, seed=self.config.seed)
            try:
                trajectory = run_episode(
                    graph,
                    env=env,
                    task_id=task_id,
                    benchmark=self.benchmark.name,
                    instruction=self.benchmark.instruction_for(task_id),
                    prompt_template=self.prompt_template,
                    prompt_version=self.prompt_version,
                    max_steps=self.config.max_steps,
                )
            finally:
                env.close()
            trajectories.append(trajectory)
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
            "trajectories": trajectories,
        }

    # -- convenience -------------------------------------------------------

    def render_progress(self) -> str:
        return TelemetryEngine.render_progress(self.reports)

    def save(self, memory_path: str | Path | None = None, prompt_path: str | Path | None = None):
        """Persist what was learned: the bank and the evolved prompt."""
        out: dict[str, Path] = {}
        target = memory_path or self.bank.path or (self.telemetry.run_dir / "memories.jsonl")
        out["memory"] = self.bank.save(target)
        prompt_target = Path(prompt_path or (self.telemetry.run_dir / "prompt.txt"))
        prompt_target.parent.mkdir(parents=True, exist_ok=True)
        prompt_target.write_text(self.prompt_template, encoding="utf-8")
        out["prompt"] = prompt_target
        return out
