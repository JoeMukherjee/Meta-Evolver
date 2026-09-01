"""The evolution graph -- the outer loop that makes the agent better.

::

    START -> sample_tasks -> rollout -> score -> induce -> credit -> prune
                  ^                                                    |
                  |                                                    v
                  |                                            optimize_prompt
                  |                                                    |
                  |                                                    v
                  +--(continue)-- checkpoint <-- curriculum <----------+
                                       |
                                  (converged / budget) --> END

Each generation improves the agent along three independent channels, which is
the reason the loop keeps paying off after the first one:

**Memory.** Failures and successes are distilled into strategies, deduplicated
into the bank, then *credited* -- every memory that appeared in a prompt is
scored by whether that episode succeeded. Memories that keep losing get pruned.
A bank that only grows plateaus; a bank that is curated does not.

**Prompt.** Failure traces drive OPRO proposals. Crucially, a proposal is only
adopted after it beats the incumbent on a held-out validation split. An
unvalidated rewrite each generation is drift, not improvement.

**Curriculum.** When the agent clears the current difficulty, the harness
stack gets harsher -- injected faults, distractor observations, tighter
budgets. Static benchmarks stop teaching once they are solved; an environment
that escalates keeps producing the failures the other two channels feed on.

The three are coupled: a harder curriculum produces richer failures, which
produce better memories and sharper prompts, which clear the next difficulty.
That coupling is the "meta" in meta-evolver.

Rollouts fan out with :class:`~langgraph.types.Send`, so tasks execute
concurrently and the ``score`` barrier waits for all of them.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from meta_evolver.core.types import GenerationReport, Trajectory
from meta_evolver.graphs.episode import run_episode
from meta_evolver.graphs.state import EvolutionState, RolloutInput
from meta_evolver.harness.curriculum import Curriculum
from meta_evolver.llm.client import BaseLLMClient
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.memory.induction import MemoryInducer
from meta_evolver.prompts.optimizer import PromptOptimizer


class EvolutionConfig:
    """Knobs for the outer loop.

    Defaults are chosen to be honest rather than flattering: validation is on
    by default, the adoption margin is non-zero, and early stopping fires
    after two flat generations rather than running out the budget to produce a
    longer-looking curve.
    """

    def __init__(
        self,
        generations: int = 5,
        tasks_per_generation: int = 0,
        validation_fraction: float = 0.34,
        max_steps: int = 15,
        retrieval_k: int = 4,
        retrieval_mode: str = "mmr",
        induce_memories: bool = True,
        optimize_prompt: bool = True,
        validate_prompt: bool = True,
        prompt_adoption_margin: float = 0.02,
        prune_min_uses: int = 4,
        prune_min_utility: float = 0.34,
        curriculum: bool = True,
        curriculum_promote_at: float = 0.7,
        curriculum_demote_at: float = 0.3,
        patience: int = 2,
        seed: int = 0,
    ) -> None:
        self.generations = int(generations)
        self.tasks_per_generation = int(tasks_per_generation)
        self.validation_fraction = float(validation_fraction)
        self.max_steps = int(max_steps)
        self.retrieval_k = int(retrieval_k)
        self.retrieval_mode = retrieval_mode
        self.induce_memories = bool(induce_memories)
        self.optimize_prompt = bool(optimize_prompt)
        self.validate_prompt = bool(validate_prompt)
        self.prompt_adoption_margin = float(prompt_adoption_margin)
        self.prune_min_uses = int(prune_min_uses)
        self.prune_min_utility = float(prune_min_utility)
        self.curriculum = bool(curriculum)
        self.curriculum_promote_at = float(curriculum_promote_at)
        self.curriculum_demote_at = float(curriculum_demote_at)
        self.patience = int(patience)
        self.seed = int(seed)


def build_evolution_graph(
    benchmark: Any,
    episode_graph: Any,
    client: BaseLLMClient,
    bank: ReasoningMemoryBank,
    config: EvolutionConfig | None = None,
    curriculum: Curriculum | None = None,
    on_report: Any = None,
):
    """Compile the evolution graph.

    ``on_report`` is called with each :class:`GenerationReport` as it is
    produced, so a CLI can stream progress without waiting for the whole run.
    """
    cfg = config or EvolutionConfig()
    curr = curriculum or Curriculum()
    inducer = MemoryInducer(client)
    optimizer = PromptOptimizer(client)
    clock: dict[int, float] = {}

    # -- nodes -------------------------------------------------------------

    def sample_tasks(state: EvolutionState) -> dict[str, Any]:
        """Pick this generation's train and validation tasks.

        The split is by *task*, not by episode, and validation tasks never
        enter training. Selecting a prompt on tasks it was written from would
        make every candidate look like an improvement.
        """
        generation = state.get("generation", 0)
        clock[generation] = time.time()
        train, val = benchmark.sample(
            generation=generation,
            n=cfg.tasks_per_generation or None,
            validation_fraction=cfg.validation_fraction if cfg.validate_prompt else 0.0,
            seed=cfg.seed,
        )
        return {"task_ids": list(train), "validation_task_ids": list(val)}

    def fan_out(state: EvolutionState):
        """Dispatch one ``rollout`` per task, concurrently.

        Each branch gets only the fields it reads: ``Send`` copies its payload
        into every branch, so forwarding the whole state would duplicate every
        trajectory collected so far once per concurrent task.
        """
        task_ids = state.get("task_ids", [])
        if not task_ids:
            return "score"
        payload: RolloutInput = {
            "generation": state.get("generation", 0),
            "curriculum_level": state.get("curriculum_level", 0.0),
            "prompt_template": state.get("prompt_template", ""),
            "prompt_version": state.get("prompt_version", "base"),
        }
        return [Send("rollout", {**payload, "task_id": task_id}) for task_id in task_ids]

    def rollout(state: RolloutInput) -> dict[str, Any]:
        """Run a single task. Returns exactly one trajectory."""
        task_id = state["task_id"]
        level = state.get("curriculum_level", 0.0)
        env = benchmark.make_env(task_id, curriculum_level=level, seed=cfg.seed)
        env = curr.wrap(env, level=level, seed=cfg.seed)
        try:
            trajectory = run_episode(
                episode_graph,
                env=env,
                task_id=task_id,
                benchmark=benchmark.name,
                instruction=benchmark.instruction_for(task_id),
                prompt_template=state.get("prompt_template", ""),
                prompt_version=state.get("prompt_version", "base"),
                max_steps=cfg.max_steps,
                generation=state.get("generation", 0),
            )
        finally:
            env.close()
        return {"trajectories": [trajectory]}

    def score(state: EvolutionState) -> dict[str, Any]:
        """Barrier: all rollouts are in. Nothing to compute yet -- the report
        is assembled at ``checkpoint``, after the learners have run and can
        report what they changed."""
        return {}

    def induce(state: EvolutionState) -> dict[str, Any]:
        """Distil this generation's episodes into candidate memories."""
        if not cfg.induce_memories:
            return {"induced": []}
        current = [t for t in state.get("trajectories", []) if t.generation == state.get("generation", 0)]
        return {"induced": inducer.induce(current, benchmark=benchmark.name)}

    def credit(state: EvolutionState) -> dict[str, Any]:
        """Charge each retrieved memory for the episode it took part in.

        Episodes that errored are excluded. Charging a memory for a rate-limit
        would, over enough generations, prune the bank's best strategies for
        reasons that have nothing to do with their quality.
        """
        generation = state.get("generation", 0)
        pairs = [
            (t.retrieved_memory_ids, t.success)
            for t in state.get("trajectories", [])
            if t.generation == generation and t.usable
        ]
        bank.credit_assign(pairs, generation=generation)
        return {}

    def prune(state: EvolutionState) -> dict[str, Any]:
        """Add the new memories, then evict the proven-bad ones.

        Order matters: adding first lets a new item merge into an existing
        near-duplicate and inherit its record, so a strategy that keeps being
        rediscovered is not repeatedly re-tried from a blank slate.
        """
        generation = state.get("generation", 0)
        before = len(bank)
        added = bank.extend(state.get("induced", []), generation=generation)
        dropped = bank.prune(
            min_uses=cfg.prune_min_uses, min_utility=cfg.prune_min_utility
        )
        if bank.path is not None:
            bank.save()
        return {
            "induced": [],
            "memories_before": before,
            "memories_added": added,
            "memories_pruned": len(dropped),
        }

    def optimize_prompt(state: EvolutionState) -> dict[str, Any]:
        """Propose prompt candidates and adopt one only if it validates."""
        if not cfg.optimize_prompt:
            return {}
        generation = state.get("generation", 0)
        current = [t for t in state.get("trajectories", []) if t.generation == generation]
        incumbent = state.get("prompt_template", "")

        candidates = optimizer.propose(
            incumbent, current, generation=generation, benchmark=benchmark.name
        )
        if not candidates:
            return {}

        val_ids = state.get("validation_task_ids") or []
        if not cfg.validate_prompt or not val_ids:
            # No held-out split available: adopt the first proposal, and say so
            # in the report. This is the weaker mode and the caller should know
            # it is running.
            best = candidates[0]
            return {
                "prompt_template": best.text,
                "prompt_version": best.version,
                "prompt_note": "adopted unvalidated (no validation split)",
            }

        baseline = _validate(state, incumbent, state.get("prompt_version", "base"), val_ids)
        best_text, best_version, best_score = incumbent, state.get("prompt_version", "base"), baseline
        for cand in candidates:
            cand.score = _validate(state, cand.text, cand.version, val_ids)
            if cand.score > best_score + cfg.prompt_adoption_margin:
                best_text, best_version, best_score = cand.text, cand.version, cand.score

        changed = best_version != state.get("prompt_version", "base")
        challenger = max((c.score or 0.0) for c in candidates)
        return {
            "prompt_template": best_text,
            "prompt_version": best_version,
            "validation_pass_rate": best_score,
            "prompt_note": (
                f"adopted {best_version} ({best_score:.2f} beat {baseline:.2f})"
                if changed
                else f"kept incumbent ({baseline:.2f}); best challenger {challenger:.2f}"
            ),
        }

    def _validate(
        state: EvolutionState, prompt: str, version: str, task_ids: Sequence[str]
    ) -> float:
        """Pass rate of ``prompt`` on the held-out tasks."""
        wins = usable = 0
        for task_id in task_ids:
            env = benchmark.make_env(
                task_id, curriculum_level=state.get("curriculum_level", 0.0), seed=cfg.seed
            )
            env = curr.wrap(env, level=state.get("curriculum_level", 0.0), seed=cfg.seed)
            try:
                t = run_episode(
                    episode_graph,
                    env=env,
                    task_id=task_id,
                    benchmark=benchmark.name,
                    instruction=benchmark.instruction_for(task_id),
                    prompt_template=prompt,
                    prompt_version=version,
                    max_steps=cfg.max_steps,
                    generation=state.get("generation", 0),
                )
            finally:
                env.close()
            if t.usable:
                usable += 1
                wins += int(t.success)
        return wins / usable if usable else 0.0

    def adapt_curriculum(state: EvolutionState) -> dict[str, Any]:
        """Raise or lower environment difficulty based on this generation."""
        if not cfg.curriculum:
            return {}
        generation = state.get("generation", 0)
        stats = _stats(state.get("trajectories", []), generation)
        level = curr.adjust(
            state.get("curriculum_level", 0.0),
            pass_rate=stats["pass_rate"],
            promote_at=cfg.curriculum_promote_at,
            demote_at=cfg.curriculum_demote_at,
        )
        return {"curriculum_level": level}

    def checkpoint(state: EvolutionState) -> dict[str, Any]:
        """Emit this generation's report and decide whether to continue."""
        generation = state.get("generation", 0)
        trajectories = state.get("trajectories", [])
        stats = _stats(trajectories, generation)
        started = clock.get(generation, time.time())

        outcomes = {
            t.task_id: t.success
            for t in trajectories
            if t.generation == generation and t.usable
        }
        previous = state.get("last_outcomes") or {}
        regressions = sum(
            1 for tid, ok in outcomes.items() if previous.get(tid) is True and not ok
        )
        recoveries = sum(
            1 for tid, ok in outcomes.items() if previous.get(tid) is False and ok
        )

        notes: list[str] = []
        if state.get("prompt_note"):
            notes.append(str(state["prompt_note"]))
        if inducer.last_error:
            notes.append(f"induction: {inducer.last_error}")
        if optimizer.last_error:
            notes.append(f"optimizer: {optimizer.last_error}")

        report = GenerationReport(
            generation=generation,
            benchmark=benchmark.name,
            n_tasks=stats["n"],
            n_errors=stats["n_errors"],
            regressions=regressions,
            recoveries=recoveries,
            pass_rate=stats["pass_rate"],
            avg_steps=stats["avg_steps"],
            avg_score=stats["avg_score"],
            memories_before=int(state.get("memories_before", len(bank))),
            memories_added=int(state.get("memories_added", 0)),
            memories_pruned=int(state.get("memories_pruned", 0)),
            prompt_changed=bool(state.get("prompt_version", "base") != _prev_version(state)),
            prompt_version=state.get("prompt_version", "base"),
            curriculum_level=float(state.get("curriculum_level", 0.0)),
            validation_pass_rate=state.get("validation_pass_rate"),
            duration_s=time.time() - started,
            notes=notes,
        )
        if on_report is not None:
            on_report(report)

        best = state.get("best_pass_rate", 0.0)
        stalled = state.get("generations_without_gain", 0)
        # Curriculum level is part of the comparison: a flat pass rate at a
        # higher difficulty is progress, and treating it as a stall would stop
        # the run exactly when it started working.
        improved = stats["pass_rate"] > best + 1e-9 or (
            cfg.curriculum and report.curriculum_level > _prev_level(state)
        )
        if regressions:
            report.notes.append(
                f"{regressions} task(s) regressed since last generation"
            )

        return {
            "reports": [report],
            "last_outcomes": outcomes,
            "best_pass_rate": max(best, stats["pass_rate"]),
            "generations_without_gain": 0 if improved else stalled + 1,
            "generation": generation + 1,
            # Clear per-generation scratch so the next report cannot inherit
            # this generation numbers when a node declines to run.
            "memories_added": 0,
            "memories_pruned": 0,
            "validation_pass_rate": None,
            "prompt_note": "",
        }

    def route_after_checkpoint(state: EvolutionState) -> Literal["sample_tasks", "__end__"]:
        if state.get("generation", 0) >= state.get("max_generations", cfg.generations):
            return "__end__"
        if state.get("generations_without_gain", 0) >= cfg.patience:
            return "__end__"
        return "sample_tasks"

    # -- assembly ----------------------------------------------------------

    graph = StateGraph(EvolutionState)
    graph.add_node("sample_tasks", sample_tasks)
    graph.add_node("rollout", rollout)
    graph.add_node("score", score)
    graph.add_node("induce", induce)
    graph.add_node("credit", credit)
    graph.add_node("prune", prune)
    graph.add_node("optimize_prompt", optimize_prompt)
    graph.add_node("adapt_curriculum", adapt_curriculum)
    graph.add_node("checkpoint", checkpoint)

    graph.add_edge(START, "sample_tasks")
    graph.add_conditional_edges("sample_tasks", fan_out, ["rollout", "score"])
    graph.add_edge("rollout", "score")
    graph.add_edge("score", "induce")
    graph.add_edge("induce", "credit")
    graph.add_edge("credit", "prune")
    graph.add_edge("prune", "optimize_prompt")
    graph.add_edge("optimize_prompt", "adapt_curriculum")
    graph.add_edge("adapt_curriculum", "checkpoint")
    graph.add_conditional_edges(
        "checkpoint", route_after_checkpoint, ["sample_tasks", END]
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _stats(trajectories: Sequence[Trajectory], generation: int) -> dict[str, Any]:
    """Aggregate one generation's episodes.

    Errored episodes are counted and reported but excluded from the rates.
    A pass rate diluted by API failures is not a measurement of the agent.
    """
    current = [t for t in trajectories if t.generation == generation]
    usable = [t for t in current if t.usable]
    n_errors = len(current) - len(usable)
    if not usable:
        return {
            "n": len(current),
            "n_errors": n_errors,
            "pass_rate": 0.0,
            "avg_steps": 0.0,
            "avg_score": 0.0,
        }
    return {
        "n": len(current),
        "n_errors": n_errors,
        "pass_rate": sum(t.success for t in usable) / len(usable),
        "avg_steps": sum(t.n_steps for t in usable) / len(usable),
        "avg_score": sum(t.score for t in usable) / len(usable),
    }


def _prev_version(state: EvolutionState) -> str:
    reports = state.get("reports") or []
    return reports[-1].prompt_version if reports else "base"


def _prev_level(state: EvolutionState) -> float:
    reports = state.get("reports") or []
    return reports[-1].curriculum_level if reports else -1.0
