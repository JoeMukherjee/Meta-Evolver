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

Rollouts fan out with :class:`~langgraph.types.Send` onto async nodes, so a
generation's episodes genuinely overlap on one event loop rather than queuing
on a thread each. Wall-clock per generation is roughly the slowest episode,
not the sum. The ``score`` barrier waits for all of them.

``rollouts_per_task`` is the scaling half of ReasoningBank's MaTTS: attempt
each task more than once and the attempts that disagree become contrastive
evidence for induction. It only pays off because the inducer exploits the
disagreement -- see :mod:`meta_evolver.memory.induction`. Note the honest
accounting that follows from it: with K rollouts, a task counts as passed if
*any* attempt passed (pass@K), and the report says which K it used, because a
pass@3 number is not comparable with a pass@1 one.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from meta_evolver.benchmarks.custom import derive_schema
from meta_evolver.core.types import GenerationReport, Trajectory
from meta_evolver.graphs.episode import arun_episode
from meta_evolver.graphs.state import EvolutionState, RolloutInput
from meta_evolver.harness.curriculum import Curriculum
from meta_evolver.llm.client import TokenMeter
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.memory.consolidate import consolidate
from meta_evolver.telemetry.tracer import TelemetryTracer
from meta_evolver.memory.induction import MemoryInducer
from meta_evolver.prompts.optimizer import PromptOptimizer
from meta_evolver.storage.checkpoint import thread_id
from meta_evolver.tools.proposer import ToolProposal, ToolProposer
from meta_evolver.tools.routing import tool_name as _schema_tool_name
from meta_evolver.tools.sandbox import run_sandboxed


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
        rollouts_per_task: int = 1,
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
        max_total_tokens: int | None = None,
        utility_decay: float = 1.0,
        domain_conditioned_utility: bool = False,
        prune_mode: str = "delete",
        rehearsal_size: int = 0,
        rehearsal_tolerance: float = 0.05,
        replay_screen: bool = False,
        replay_top_k: int | None = None,
        consolidate_every: int = 0,
        consolidate_similarity: float = 0.80,
        context_reducer: bool = False,
        context_reducer_min_chars: int = 2000,
        context_reducer_max_chars: int = 800,
        retrieval_min_similarity: float = 0.0,
        enable_tool_proposals: bool = False,
        tool_sandbox_timeout: float = 5.0,
    ) -> None:
        self.generations = int(generations)
        self.tasks_per_generation = int(tasks_per_generation)
        self.rollouts_per_task = max(1, int(rollouts_per_task))
        """Attempts per task per generation (MaTTS). Above 1, a task counts as
        passed if any attempt passed, and disagreeing attempts become
        contrastive evidence for memory induction. Costs K times as much."""
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
        self.max_total_tokens = None if max_total_tokens is None else int(max_total_tokens)
        """Hard budget for a run. No new generation starts once the finished
        ones have spent this many tokens; ``None`` leaves a run unbounded."""
        self.utility_decay = float(utility_decay)
        """Gamma for domain-conditioned credit assignment -- see
        :meth:`~meta_evolver.core.types.MemoryItem.record_domain_usage`.
        ``1.0`` (the default) never discounts. Only takes effect when a
        trajectory carries a derivable domain; see :func:`_domain_of`."""
        self.domain_conditioned_utility = bool(domain_conditioned_utility)
        """Whether :meth:`~meta_evolver.memory.bank.ReasoningMemoryBank.prune`
        spares an item that is strong in *some* domain even though its pooled
        utility fell below the cutoff -- the out-of-domain eviction fix.
        ``False`` (the default) is today's pooled-only behaviour."""
        if prune_mode not in ("delete", "archive"):
            raise ValueError(f"prune_mode must be 'delete' or 'archive', got {prune_mode!r}")
        self.prune_mode = prune_mode
        """``"delete"`` (default, today's behaviour) or ``"archive"``, which
        moves a pruned item to the bank's ``archived`` list instead -- see
        :meth:`~meta_evolver.memory.bank.ReasoningMemoryBank.revive`."""
        self.rehearsal_size = int(rehearsal_size)
        """How many previously-passed task ids a prompt challenger must also
        still pass before it can be adopted. ``0`` (the default) runs no
        rehearsal check at all -- today's behaviour, where the held-out
        margin is the only gate."""
        self.rehearsal_tolerance = float(rehearsal_tolerance)
        """Fraction of the rehearsal set a challenger may fail and still be
        adopted -- ``0.05`` allows one miss in twenty. Irrelevant when
        ``rehearsal_size`` is ``0``."""
        self.replay_screen = bool(replay_screen)
        """Whether ``_select_prompt`` pre-trims candidates with a
        :class:`~meta_evolver.eval.replay.ReplayScreener` before paying for
        real validation rollouts. ``False`` (the default) validates every
        candidate, exactly as before this existed -- turning it on with no
        ``replay_top_k`` set still costs nothing (screening only trims when
        there are more candidates than the budget)."""
        self.replay_top_k = None if replay_top_k is None else int(replay_top_k)
        """How many candidates screening keeps for real validation once it
        has enough history to predict from. ``None`` means "no trimming" --
        the screener still records every real result either way, in case a
        later generation wants to raise the budget down."""
        self.consolidate_every = int(consolidate_every)
        """Run :func:`~meta_evolver.memory.consolidate.consolidate` after
        every Nth completed generation's ``prune`` step. ``0`` (the default)
        never runs it -- today's behaviour, where only write-time dedup and
        prune shape the bank."""
        self.consolidate_similarity = float(consolidate_similarity)
        """Cosine threshold for :func:`consolidate`'s clustering. Looser
        than the bank's own write-time ``dedup_similarity`` on purpose --
        see that function's docstring for why the two thresholds differ."""
        self.context_reducer = bool(context_reducer)
        """Whether the episode graph compresses a large tool observation
        through a :class:`~meta_evolver.tools.reducer.EvidenceReducer`
        instead of flat-truncating it. ``False`` (the default) is today's
        behaviour -- cut to ``observation_chars`` and keep whichever prefix
        that lands on. Built by :class:`~meta_evolver.core.evolver.
        MetaEvolver` from this flag; it reuses the run's own chat model, so
        turning it on costs one extra model call per large observation."""
        self.context_reducer_min_chars = int(context_reducer_min_chars)
        """Below this many characters, the reducer returns the observation
        unchanged with no model call -- see ``EvidenceReducer.min_chars``."""
        self.context_reducer_max_chars = int(context_reducer_max_chars)
        """Cap on the reducer's own output, mirroring ``EvidenceReducer.
        max_output_chars`` -- a fallback truncation still applies if the
        model call fails or the compressed reply itself runs long."""
        self.retrieval_min_similarity = float(retrieval_min_similarity)
        """Floor passed through to ``bank.retrieve`` in the episode graph's
        ``prepare`` node, making ``retrieval_k`` query-adaptive: a candidate
        below this similarity is dropped rather than padding out the top-k,
        so a query with few relevant memories gets fewer than
        ``retrieval_k`` of them. ``0.0`` (the default) never drops anything
        on similarity alone -- today's behaviour, exactly ``retrieval_k``
        items whenever the bank has that many."""
        self.enable_tool_proposals = bool(enable_tool_proposals)
        """Research-only, off by default. When true *and* the benchmark
        exposes a mutable tool registry (a ``tools: dict`` and ``schemas:
        list`` -- ``FunctionBenchmark``'s shape; most benchmarks do not and
        the node is then a documented no-op), each generation's failures may
        produce one proposed pure-function tool. It only ever reaches the
        live registry after clearing both gates: :mod:`meta_evolver.tools.
        sandbox` (AST allowlist, then a real subprocess run of its own
        generated test cases, with a timeout) and the same held-out-margin-
        plus-rehearsal standard a prompt candidate has to clear. ``False``
        (the default) adds no tool-proposal node work at all."""
        self.tool_sandbox_timeout = float(tool_sandbox_timeout)
        """Wall-clock seconds the sandbox subprocess gets to run a proposed
        tool's own test cases before it is killed and rejected."""


def build_evolution_graph(
    benchmark: Any,
    episode_graph: Any,
    model: BaseChatModel,
    bank: ReasoningMemoryBank,
    config: EvolutionConfig | None = None,
    curriculum: Curriculum | None = None,
    on_report: Any = None,
    run_id: str = "run",

    recorder: Any = None,
    trace_dir: str | Path | None = None,
    prompt_optimizer: Any = None,
    replay_screener: Any = None,
    tool_proposer: Any = None,
    learner: Any = None,
):
    """Compile the evolution graph.

    ``on_report`` is called with each :class:`GenerationReport` as it is
    produced, so a CLI can stream progress without waiting for the whole run.

    ``recorder`` is an optional
    :class:`~meta_evolver.graph_view.recorder.CausalGraphRecorder`. It is
    written to as each generation completes rather than at the end, so the
    graph is watchable while the run happens -- and it can never fail a run:
    every write there is guarded.

    ``learner`` is an optional :class:`~meta_evolver.learning.GenerationLearner`
    -- the knowledge graph and skill library. It is called once after credit
    and once after prune, and can never fail a run.

    ``trace_dir`` opts every episode this graph runs -- training rollouts and
    validation rollouts alike, since both funnel through the same internal
    ``_episode`` helper -- into a fresh per-episode
    :class:`~meta_evolver.telemetry.tracer.TelemetryTracer`, saved to
    ``trace_dir`` as one JSON file per episode on completion. ``None`` (the
    default) skips tracing entirely, at zero cost.

    ``prompt_optimizer`` defaults to :class:`~meta_evolver.prompts.optimizer.
    PromptOptimizer` -- today's monolithic-prompt rewriter -- when ``None``.
    Pass a :class:`~meta_evolver.prompts.gepa.GEPAPromptOptimizer` for
    component-targeted mutation instead: it already freezes ``core_role`` and
    only ever mutates ``planning``/``tool_policy``/``error_recovery``, and
    implements the same ``propose(current_prompt, trajectories, generation,
    benchmark) -> list[PromptCandidate]`` contract, so nothing downstream of
    this line -- ``_select_prompt``, the rehearsal gate, the causal-graph
    recorder -- has to know which one it got.

    ``replay_screener`` is an optional
    :class:`~meta_evolver.eval.replay.ReplayScreener`. When ``cfg.
    replay_screen`` is also true, ``_select_prompt`` trims candidates through
    it before validating (its own ``top_k`` decides how many survive), then
    feeds every real validation result back in -- growing its predictor and
    checking, every generation, whether its predictions still agree with
    what real validation found. ``None`` (the default) skips this entirely.

    ``tool_proposer`` defaults to :class:`~meta_evolver.tools.proposer.
    ToolProposer` when ``None``, mirroring ``prompt_optimizer`` -- it is
    always constructed, but only ever invoked when ``cfg.
    enable_tool_proposals`` is true, so building it costs nothing by
    default.
    """
    cfg = config or EvolutionConfig()
    curr = curriculum or Curriculum()
    inducer = MemoryInducer(model)
    optimizer = prompt_optimizer if prompt_optimizer is not None else PromptOptimizer(model)
    proposer = tool_proposer if tool_proposer is not None else ToolProposer(model)
    clock: dict[int, float] = {}

    # -- nodes -------------------------------------------------------------

    async def sample_tasks(state: EvolutionState) -> dict[str, Any]:
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
        # One branch per (task, attempt). Each is an independent Send, so K
        # rollouts of one task overlap with every other task's rather than
        # running as an inner loop inside a single branch.
        return [
            Send("rollout", {**payload, "task_id": task_id, "rollout_index": attempt})
            for task_id in task_ids
            for attempt in range(cfg.rollouts_per_task)
        ]

    async def rollout(state: RolloutInput) -> dict[str, Any]:
        """Run a single attempt at a single task. Returns one trajectory."""
        generation = state.get("generation", 0)
        trajectory = await _episode(
            task_id=state["task_id"],
            generation=generation,
            level=state.get("curriculum_level", 0.0),
            prompt=state.get("prompt_template", ""),
            version=state.get("prompt_version", "base"),
            rollout_index=int(state.get("rollout_index", 0)),
            phase="train",
        )
        return {"trajectories": [trajectory]}

    async def _episode(
        task_id: str,
        generation: int,
        level: float,
        prompt: str,
        version: str,
        rollout_index: int = 0,
        phase: str = "train",
    ) -> Trajectory:
        """Build the env stack, run one episode, and always release the env.

        Seeded by attempt so K rollouts of one task meet different perturbation
        draws -- identical harness seeds would make repeated attempts
        near-duplicates, and MaTTS needs them to be able to diverge.
        """
        seed = cfg.seed + rollout_index * 7919
        env = benchmark.make_env(task_id, curriculum_level=level, seed=seed)
        env = curr.wrap(env, level=level, seed=seed)
        tracer = (
            TelemetryTracer(task_id=task_id, run_id=run_id) if trace_dir is not None else None
        )
        try:
            trajectory = await arun_episode(
                episode_graph,
                env=env,
                task_id=task_id,
                benchmark=benchmark.name,
                instruction=benchmark.instruction_for(task_id),
                prompt_template=prompt,
                prompt_version=version,
                max_steps=cfg.max_steps,
                generation=generation,
                rollout_index=rollout_index,
                thread_id=thread_id(run_id, phase, generation, version, task_id, rollout_index),
                tracer=tracer,
            )
            if tracer is not None:
                path = (
                    Path(trace_dir)
                    / f"gen{generation}_{phase}_{task_id}_r{rollout_index}_{tracer.trace_id}.json"
                )
                tracer.save_to_file(path)
            return trajectory
        finally:
            env.close()

    async def score(state: EvolutionState) -> dict[str, Any]:
        """Barrier: all rollouts are in. Nothing to compute yet -- the report
        is assembled at ``checkpoint``, after the learners have run and can
        report what they changed."""
        return {}

    async def induce(state: EvolutionState) -> dict[str, Any]:
        """Distil this generation's episodes into candidate memories."""
        if not cfg.induce_memories:
            return {"induced": [], "induction_tokens": 0}
        current = [
            t for t in state.get("trajectories", []) if t.generation == state.get("generation", 0)
        ]
        with TokenMeter() as meter:
            induced = inducer.induce(current, benchmark=benchmark.name)
        if recorder is not None and induced:
            recorder.record_memories(induced, generation=state.get("generation", 0))
        return {"induced": induced, "induction_tokens": meter.total}

    async def credit(state: EvolutionState) -> dict[str, Any]:
        """Charge each retrieved memory for the episode it took part in.

        Episodes that errored are excluded. Charging a memory for a rate-limit
        would, over enough generations, prune the bank's best strategies for
        reasons that have nothing to do with their quality.
        """
        generation = state.get("generation", 0)
        triples = [
            (t.retrieved_memory_ids, t.success, _domain_of(t))
            for t in state.get("trajectories", [])
            if t.generation == generation and t.usable
        ]
        bank.credit_assign(triples, generation=generation, gamma=cfg.utility_decay)
        if learner is not None:
            learner.credit(state.get("trajectories", []), generation, gamma=cfg.utility_decay)
        return {}

    async def prune(state: EvolutionState) -> dict[str, Any]:
        """Add the new memories, then evict the proven-bad ones.

        Order matters: adding first lets a new item merge into an existing
        near-duplicate and inherit its record, so a strategy that keeps being
        rediscovered is not repeatedly re-tried from a blank slate.
        """
        generation = state.get("generation", 0)
        before = len(bank)
        added = bank.extend(state.get("induced", []), generation=generation)
        dropped = bank.prune(
            min_uses=cfg.prune_min_uses,
            min_utility=cfg.prune_min_utility,
            domain_conditioned=cfg.domain_conditioned_utility,
            archive=(cfg.prune_mode == "archive"),
        )
        if recorder is not None and dropped:
            recorder.record_pruned(dropped, generation=generation)

        consolidated = 0
        if cfg.consolidate_every > 0 and (generation + 1) % cfg.consolidate_every == 0:
            # After prune, not before: consolidating items about to be
            # evicted anyway wastes the pass. Fires on the Nth, 2Nth, ...
            # completed generation, not generation 0 -- nothing has
            # accumulated yet to usefully cluster on the very first one.
            merged = consolidate(bank, similarity_threshold=cfg.consolidate_similarity)
            consolidated = len(merged)
            if recorder is not None and merged:
                recorder.record_pruned(merged, generation=generation)

        if learner is not None:
            learner.learn(state.get("trajectories", []), generation)

        if bank.path is not None:
            bank.save()
        return {
            "induced": [],
            "memories_before": before,
            "memories_added": added,
            "memories_pruned": len(dropped) + consolidated,
        }

    async def optimize_prompt(state: EvolutionState) -> dict[str, Any]:
        """Propose prompt candidates and adopt one only if it validates."""
        if not cfg.optimize_prompt:
            return {"optimization_tokens": 0}
        generation = state.get("generation", 0)
        current = [t for t in state.get("trajectories", []) if t.generation == generation]
        incumbent = state.get("prompt_template", "")

        # Metered around proposals and validation together: validation
        # rollouts are real spend attributable to prompt optimization, and
        # attributing them to the generation's rollouts would understate what
        # searching for a better prompt actually costs.
        meter = TokenMeter().__enter__()
        try:
            candidates = optimizer.propose(
                incumbent, current, generation=generation, benchmark=benchmark.name
            )
            if not candidates:
                return {"optimization_tokens": 0}
            result = await _select_prompt(state, incumbent, candidates)
        finally:
            meter.__exit__(None, None, None)
        result["optimization_tokens"] = meter.total
        return result

    async def _select_prompt(
        state: EvolutionState, incumbent: str, candidates: list
    ) -> dict[str, Any]:
        """Measure each candidate against the incumbent and pick a winner."""
        val_ids = state.get("validation_task_ids") or []
        if not cfg.validate_prompt or not val_ids:
            # No held-out split available: adopt the first proposal, and say so
            # in the report. This is the weaker mode and the caller should know
            # it is running -- rehearsal has nothing to check a prompt against
            # here either, since it is built from validated passes.
            best = candidates[0]
            return {
                "prompt_template": best.text,
                "prompt_version": best.version,
                "prompt_note": "adopted unvalidated (no validation split)",
            }

        screen_note = ""
        if cfg.replay_screen and replay_screener is not None:
            before = len(candidates)
            candidates = replay_screener.screen(candidates)
            if len(candidates) < before:
                screen_note = f"; replay-screened {before} candidates to {len(candidates)}"

        incumbent_version = state.get("prompt_version", "base")
        # The incumbent and every candidate are measured on the same held-out
        # tasks, concurrently: they are independent, and running them in
        # sequence would make validation the slowest part of a generation.
        scores = await asyncio.gather(
            _validate(state, incumbent, incumbent_version, val_ids),
            *[_validate(state, c.text, c.version, val_ids) for c in candidates],
        )
        (baseline, baseline_passed), candidate_results = scores[0], scores[1:]

        if cfg.replay_screen and replay_screener is not None:
            # Every real result feeds the predictor -- the incumbent's own
            # baseline included, since it is just as real a data point as any
            # candidate's. Then check whether this generation's predictions
            # actually agreed with what validation found.
            replay_screener.record(incumbent, baseline)
            for cand, (score, _passed) in zip(candidates, candidate_results, strict=True):
                replay_screener.record(cand.text, score)
            agreement = replay_screener.update_agreement(
                [(cand, score) for cand, (score, _p) in zip(candidates, candidate_results, strict=True)]
            )
            if agreement is not None:
                screen_note += f"; replay agreement {agreement:.2f}"
                if replay_screener.disabled:
                    screen_note += " (screening disabled)"

        best_text, best_version, best_score = incumbent, incumbent_version, baseline
        best_passed = baseline_passed
        for cand, (score, passed) in zip(candidates, candidate_results, strict=True):
            cand.score = score
            if score > best_score + cfg.prompt_adoption_margin:
                best_text, best_version, best_score = cand.text, cand.version, score
                best_passed = passed

        rehearsal_note = ""
        rehearsal_ids = list(state.get("rehearsal_ids") or [])
        if cfg.rehearsal_size > 0 and rehearsal_ids and best_version != incumbent_version:
            # The margin alone only proves the challenger is better *here*;
            # rehearsal proves it did not quietly break something that used
            # to work. Checked only for the prompt about to change, not the
            # incumbent -- the incumbent produced this rehearsal set, so it
            # passes it by construction.
            rehearsal_rate, _ = await _validate(state, best_text, best_version, rehearsal_ids)
            if rehearsal_rate < 1.0 - cfg.rehearsal_tolerance:
                rehearsal_note = (
                    f"; rejected {best_version} on rehearsal ({rehearsal_rate:.2f} of "
                    f"{len(rehearsal_ids)} previously-passed tasks)"
                )
                best_text, best_version, best_score = incumbent, incumbent_version, baseline
                best_passed = baseline_passed

        if recorder is not None:
            # Rejected candidates are recorded too: they are the evidence that
            # alternatives were tried and measured, which a graph showing only
            # the adopted lineage would lose.
            recorder.record_prompt_candidates(
                candidates, generation=state.get("generation", 0), adopted_version=best_version
            )

        # The buffer tracks the winning prompt's held-out passes *and* this
        # generation's own train-task passes (always run under the
        # incumbent, since optimize_prompt happens after rollout/score). Both
        # matter: the held-out set alone never changes within one run
        # (BenchmarkAdapter.sample seeds it once, not per generation), so
        # without the train side rehearsal could only ever re-check the same
        # task the margin check just ran -- never catching a regression on
        # something solved earlier and not touched this generation.
        train_passed = [
            t.task_id
            for t in state.get("trajectories", [])
            if t.generation == state.get("generation", 0) and t.usable and t.success
        ]
        new_ids = [
            tid for tid in (best_passed + train_passed) if tid not in rehearsal_ids
        ]
        new_rehearsal_ids = rehearsal_ids + new_ids
        if cfg.rehearsal_size > 0:
            new_rehearsal_ids = new_rehearsal_ids[-cfg.rehearsal_size :]
        else:
            new_rehearsal_ids = []

        changed = best_version != state.get("prompt_version", "base")
        # Empty only if screening trimmed every candidate this generation
        # (a real possibility with a strict top_k) -- the incumbent's own
        # baseline is the only honest "best challenger" figure to show then.
        challenger = max((c.score or 0.0) for c in candidates) if candidates else baseline
        return {
            "prompt_template": best_text,
            "prompt_version": best_version,
            "validation_pass_rate": best_score,
            "rehearsal_ids": new_rehearsal_ids,
            "prompt_note": (
                (
                    f"adopted {best_version} ({best_score:.2f} beat {baseline:.2f})"
                    if changed
                    else f"kept incumbent ({baseline:.2f}); best challenger {challenger:.2f}"
                )
                + rehearsal_note
                + screen_note
            ),
        }

    async def _validate(
        state: EvolutionState, prompt: str, version: str, task_ids: Sequence[str]
    ) -> tuple[float, list[str]]:
        """``(pass rate, passed task ids)`` of ``prompt`` on ``task_ids``.

        Validation is single-rollout regardless of ``rollouts_per_task``: it
        compares prompts, and pass@K would blur the comparison it exists to
        make while multiplying its cost.
        """
        results = await asyncio.gather(
            *[
                _episode(
                    task_id=task_id,
                    generation=state.get("generation", 0),
                    level=state.get("curriculum_level", 0.0),
                    prompt=prompt,
                    version=version,
                    phase=f"validate-{version}",
                )
                for task_id in task_ids
            ]
        )
        usable = [t for t in results if t.usable]
        rate = sum(t.success for t in usable) / len(usable) if usable else 0.0
        passed = [t.task_id for t in usable if t.success]
        return rate, passed

    def _tool_registry_available() -> bool:
        """Whether ``benchmark`` exposes the ``FunctionBenchmark``-shaped
        mutable tool registry (``tools: dict``, ``schemas: list``) tool
        adoption needs. Most benchmarks -- devops, textworld, the memory
        ones -- build their action space from fixed environment code and do
        not; the node is a documented no-op for them, not an error."""
        return isinstance(getattr(benchmark, "tools", None), dict) and isinstance(
            getattr(benchmark, "schemas", None), list
        )

    async def propose_tool(state: EvolutionState) -> dict[str, Any]:
        """Propose one pure-function tool and adopt it only if it clears
        both self-modification gates."""
        if not cfg.enable_tool_proposals:
            return {"tool_tokens": 0}
        if not _tool_registry_available():
            return {
                "tool_tokens": 0,
                "tool_note": "tool proposals skipped: benchmark has no mutable tool registry",
            }
        generation = state.get("generation", 0)
        current = [t for t in state.get("trajectories", []) if t.generation == generation]

        meter = TokenMeter().__enter__()
        try:
            proposals = proposer.propose(current, generation=generation, benchmark=benchmark.name)
            if not proposals:
                return {"tool_tokens": 0}
            result = await _select_tool(state, proposals[0])
        finally:
            meter.__exit__(None, None, None)
        result["tool_tokens"] = meter.total
        return result

    async def _select_tool(state: EvolutionState, proposal: ToolProposal) -> dict[str, Any]:
        """Gate 1 (sandbox), then Gate 2 (held-out margin plus rehearsal) --
        the same standard :func:`_select_prompt` holds a prompt candidate
        to, applied to a tool instead. Adoption mutates ``benchmark.tools``/
        ``schemas`` permanently; rejection leaves them untouched."""
        if proposal.name in benchmark.tools:
            return {"tool_note": f"tool {proposal.name} rejected: name already in use"}

        sandbox = run_sandboxed(
            proposal.code, proposal.name, proposal.test_cases, timeout=cfg.tool_sandbox_timeout
        )
        if not sandbox.passed:
            return {"tool_note": f"tool {proposal.name} rejected at sandbox gate: {sandbox.render()}"}

        val_ids = state.get("validation_task_ids") or []
        if not val_ids:
            return {
                "tool_note": f"tool {proposal.name} passed sandbox but skipped adoption (no validation split)"
            }

        fn = _compile_tool(proposal)
        schema = derive_schema(proposal.name, fn)

        baseline_score, _baseline_passed = await _validate(
            state, state.get("prompt_template", ""), state.get("prompt_version", "base"), val_ids
        )
        candidate_score, candidate_passed = await _validate_with_tool(
            state, proposal.name, fn, schema, val_ids
        )

        if candidate_score <= baseline_score + cfg.prompt_adoption_margin:
            return {
                "tool_note": (
                    f"tool {proposal.name} rejected on held-out validation "
                    f"({candidate_score:.2f} did not beat baseline {baseline_score:.2f})"
                )
            }

        rehearsal_note = ""
        rehearsal_ids = list(state.get("rehearsal_ids") or [])
        if cfg.rehearsal_size > 0 and rehearsal_ids:
            rehearsal_rate, _ = await _validate_with_tool(
                state, proposal.name, fn, schema, rehearsal_ids
            )
            if rehearsal_rate < 1.0 - cfg.rehearsal_tolerance:
                return {
                    "tool_note": (
                        f"tool {proposal.name} rejected on rehearsal ({rehearsal_rate:.2f} of "
                        f"{len(rehearsal_ids)} previously-passed tasks)"
                    )
                }
            rehearsal_note = f"; rehearsal {rehearsal_rate:.2f}"

        # Adopted for real -- every episode from here on can see this tool.
        benchmark.tools[proposal.name] = fn
        benchmark.schemas.append(schema)

        return {
            "tool_note": (
                f"adopted tool {proposal.name} "
                f"({candidate_score:.2f} beat {baseline_score:.2f}){rehearsal_note}"
            ),
            "adopted_tools": [
                {
                    "name": proposal.name,
                    "code": proposal.code,
                    "description": proposal.description,
                    "generation": state.get("generation", 0),
                    "score": candidate_score,
                    "baseline_score": baseline_score,
                }
            ],
        }

    async def _validate_with_tool(
        state: EvolutionState, name: str, fn: Any, schema: dict[str, Any], task_ids: Sequence[str]
    ) -> tuple[float, list[str]]:
        """:func:`_validate`, with ``name`` temporarily added to the
        benchmark's live tool registry for the duration of the call.

        Safe to call for a name the registry already owns (adoption checks
        that before ever reaching here) -- this only ever adds and reverts
        an addition it made itself.
        """
        benchmark.tools[name] = fn
        benchmark.schemas.append(schema)
        try:
            return await _validate(
                state, state.get("prompt_template", ""), state.get("prompt_version", "base"), task_ids
            )
        finally:
            del benchmark.tools[name]
            benchmark.schemas[:] = [s for s in benchmark.schemas if _schema_tool_name(s) != name]

    async def adapt_curriculum(state: EvolutionState) -> dict[str, Any]:
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

    async def checkpoint(state: EvolutionState) -> dict[str, Any]:
        """Emit this generation's report and decide whether to continue."""
        generation = state.get("generation", 0)
        trajectories = state.get("trajectories", [])
        stats = _stats(trajectories, generation)
        started = clock.get(generation, time.time())

        outcomes: dict[str, bool] = {}
        for t in trajectories:
            if t.generation == generation and t.usable:
                # pass@K: a task regressed only if every attempt failed.
                outcomes[t.task_id] = outcomes.get(t.task_id, False) or t.success
        previous = state.get("last_outcomes") or {}
        regressions = sum(
            1 for tid, ok in outcomes.items() if previous.get(tid) is True and not ok
        )
        recoveries = sum(
            1 for tid, ok in outcomes.items() if previous.get(tid) is False and ok
        )

        # Rollouts report their own usage on the trajectory; the two spending
        # nodes report theirs directly. Together that is every model call the
        # generation made.
        tokens = (
            sum(t.tokens for t in trajectories if t.generation == generation)
            + int(state.get("induction_tokens", 0) or 0)
            + int(state.get("optimization_tokens", 0) or 0)
            + int(state.get("tool_tokens", 0) or 0)
        )

        notes: list[str] = []
        if state.get("prompt_note"):
            notes.append(str(state["prompt_note"]))
        if state.get("tool_note"):
            notes.append(str(state["tool_note"]))
        if inducer.last_error:
            notes.append(f"induction: {inducer.last_error}")
        if optimizer.last_error:
            notes.append(f"optimizer: {optimizer.last_error}")
        if proposer.last_error:
            notes.append(f"tool proposer: {proposer.last_error}")

        report = GenerationReport(
            generation=generation,
            benchmark=benchmark.name,
            n_tasks=stats["n"],
            n_errors=stats["n_errors"],
            regressions=regressions,
            recoveries=recoveries,
            tokens=tokens,
            rollouts_per_task=cfg.rollouts_per_task,
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
        if recorder is not None:
            recorder.record_generation(
                report,
                trajectories,
                prompt_text=state.get("prompt_template", ""),
                curriculum_name=curr.describe(report.curriculum_level),
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
            "tool_note": "",
            "induction_tokens": 0,
            "optimization_tokens": 0,
            "tool_tokens": 0,
        }

    def route_after_checkpoint(state: EvolutionState) -> Literal["sample_tasks", "__end__"]:
        if state.get("generation", 0) >= state.get("max_generations", cfg.generations):
            return "__end__"
        if state.get("generations_without_gain", 0) >= cfg.patience:
            return "__end__"
        if cfg.max_total_tokens is not None:
            # Checked between generations, not mid-generation: stopping a
            # generation halfway would report a pass rate over a partial task
            # set, which is not comparable with any other generation's.
            spent = sum(r.tokens for r in state.get("reports") or [])
            if spent >= cfg.max_total_tokens:
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
    graph.add_node("propose_tool", propose_tool)
    graph.add_node("adapt_curriculum", adapt_curriculum)
    graph.add_node("checkpoint", checkpoint)

    graph.add_edge(START, "sample_tasks")
    graph.add_conditional_edges("sample_tasks", fan_out, ["rollout", "score"])
    graph.add_edge("rollout", "score")
    graph.add_edge("score", "induce")
    graph.add_edge("induce", "credit")
    graph.add_edge("credit", "prune")
    graph.add_edge("prune", "optimize_prompt")
    graph.add_edge("optimize_prompt", "propose_tool")
    graph.add_edge("propose_tool", "adapt_curriculum")
    graph.add_edge("adapt_curriculum", "checkpoint")
    graph.add_conditional_edges(
        "checkpoint", route_after_checkpoint, ["sample_tasks", END]
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _compile_tool(proposal: ToolProposal) -> Any:
    """Turn a proposal's source into a live callable.

    Only ever called after :func:`~meta_evolver.tools.sandbox.run_sandboxed`
    has already passed it -- the AST allowlist ruled out file, network,
    process, and reflection access, and a real subprocess run of the
    proposal's own test cases confirmed it behaves as claimed. Compiling it
    in-process at this point runs the same vetted source again, not
    unvetted code; re-sandboxing every call to an adopted tool would repeat
    a check that vetting already established the code cannot fail.
    """
    ns: dict[str, Any] = {}
    exec(compile(proposal.code, f"<tool:{proposal.name}>", "exec"), ns)
    return ns[proposal.name]


def _domain_of(trajectory: Trajectory) -> str | None:
    """The finest-grained domain label available for a trajectory.

    Prefers a benchmark-supplied category (LongMemEval's ``question_type``,
    LoCoMo's ``category_name``) over the benchmark name itself, because that
    is what actually separates "this rule works" from "this rule doesn't"
    within one benchmark's run -- ``trajectory.benchmark`` is constant for
    the whole run, so as a domain key alone it would only start doing
    anything once a bank is reused across benchmarks (a later phase's
    MetaAdapter reattachment). Falling back to it costs nothing today and
    is correct once that happens.
    """
    metrics = trajectory.metrics or {}
    category = metrics.get("category_name") or metrics.get("question_type")
    if category:
        return str(category)
    return trajectory.benchmark or None


def _stats(trajectories: Sequence[Trajectory], generation: int) -> dict[str, Any]:
    """Aggregate one generation's episodes, per task rather than per episode.

    Two things this gets right that a naive mean does not:

    * Errored episodes are counted and reported but excluded from the rates.
      A pass rate diluted by API failures is not a measurement of the agent.
    * With K rollouts per task, a task passes if *any* attempt passed (pass@K)
      and is counted once. Averaging over episodes instead would let a task
      attempted three times outvote one attempted once, so the headline number
      would drift with the rollout budget rather than with the agent.
    """
    current = [t for t in trajectories if t.generation == generation]
    usable = [t for t in current if t.usable]
    n_errors = len(current) - len(usable)
    if not usable:
        return {
            "n": len({t.task_id for t in current}),
            "n_errors": n_errors,
            "pass_rate": 0.0,
            "avg_steps": 0.0,
            "avg_score": 0.0,
        }

    by_task: dict[str, list[Trajectory]] = {}
    for trajectory in usable:
        by_task.setdefault(trajectory.task_id, []).append(trajectory)

    passed = sum(1 for attempts in by_task.values() if any(t.success for t in attempts))
    # Steps and score describe the attempt that counted: the best one.
    best = [max(attempts, key=lambda t: (t.success, t.score, -t.n_steps)) for attempts in by_task.values()]
    return {
        "n": len(by_task),
        "n_errors": n_errors,
        "pass_rate": passed / len(by_task),
        "avg_steps": sum(t.n_steps for t in best) / len(best),
        "avg_score": sum(t.score for t in best) / len(best),
    }


def _prev_version(state: EvolutionState) -> str:
    reports = state.get("reports") or []
    return reports[-1].prompt_version if reports else "base"


def _prev_level(state: EvolutionState) -> float:
    reports = state.get("reports") or []
    return reports[-1].curriculum_level if reports else -1.0
