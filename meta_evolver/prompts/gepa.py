"""Feedback-driven multi-component prompt optimizer with Pareto frontier selection.

Inspired by DSPy's GEPA (Genetic Evolutionary Prompt Architecture), this
optimizer models the system prompt as a composite of distinct semantic
components (e.g., core role, planning strategy, tool governance, error recovery).

Key capabilities:
1. **Component-Targeted Mutation**: Reflects on failure traces and feedback to
   mutate specific sub-components while leaving well-performing components intact.
2. **Pareto Frontier Population Tracking**: Maintains a population of candidate
   prompts scored across distinct tasks; samples from the multi-objective Pareto
   frontier rather than a single collapsed scalar score.
3. **Crossover & Merging**: Combines winning components from distinct Pareto
   parents (e.g., Parent A's superior planning with Parent B's superior tool policy).
4. **Rich Natural-Language Feedback**: Feeds detailed failure diagnoses directly
   into the reflection model's prompt.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from meta_evolver.core.types import Trajectory
from meta_evolver.llm.client import invoke_text
from meta_evolver.prompts.optimizer import PromptCandidate
from meta_evolver.prompts.templates import BASE_SYSTEM_PROMPT, ensure_placeholders


@dataclass
class ModularPrompt:
    """A prompt decomposed into modular semantic components."""

    core_role: str = (
        "You are an expert autonomous software and systems agent solving complex benchmark tasks."
    )
    planning: str = (
        "Formulate a clear hypothesis before taking action. Break multi-step problems into "
        "testable sub-tasks."
    )
    tool_policy: str = (
        "Choose tools deliberately. Verify required arguments before invoking tools. "
        "Treat unexpected results as evidence to update your hypothesis."
    )
    error_recovery: str = (
        "If an action fails, is blocked, or produces no progress, do not repeat it verbatim. "
        "Switch approaches or inspect system state to identify root causes."
    )

    def render(self) -> str:
        """Render all components into a coherent system prompt."""
        body = (
            f"{self.core_role.strip()}\n\n"
            f"### Planning & Strategy\n{self.planning.strip()}\n\n"
            f"### Tool Governance\n{self.tool_policy.strip()}\n\n"
            f"### Error Recovery\n{self.error_recovery.strip()}\n\n"
            f"{{memory_section}}\n\n"
            f"{{guidance_section}}\n\n"
            f"### Verification\nBefore submitting a final resolution, verify your fix with ground-truth checks."
        )
        return ensure_placeholders(body)

    @classmethod
    def from_text(cls, text: str) -> ModularPrompt:
        """Construct a modular prompt from existing prompt text."""
        cleaned = text.replace("{memory_section}", "").replace("{guidance_section}", "").strip()
        lines = [line.strip() for line in cleaned.split("\n\n") if line.strip()]
        if len(lines) >= 4:
            return cls(
                core_role=lines[0],
                planning=lines[1],
                tool_policy=lines[2],
                error_recovery=lines[3],
            )
        return cls(core_role=cleaned or cls().core_role)

    def copy(self) -> ModularPrompt:
        return ModularPrompt(
            core_role=self.core_role,
            planning=self.planning,
            tool_policy=self.tool_policy,
            error_recovery=self.error_recovery,
        )


@dataclass
class ParetoCandidate:
    """A candidate in the GEPA evolutionary population."""

    id: str
    modular_prompt: ModularPrompt
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0
    task_scores: dict[str, float] = field(default_factory=dict)
    feedbacks: list[str] = field(default_factory=list)
    validation_score: float | None = None

    @property
    def mean_score(self) -> float:
        if not self.task_scores:
            return 0.0
        return sum(self.task_scores.values()) / len(self.task_scores)


class ParetoFrontier:
    """Calculates and maintains the non-dominated Pareto frontier over task scores."""

    @staticmethod
    def dominates(cand_a: ParetoCandidate, cand_b: ParetoCandidate, all_tasks: list[str]) -> bool:
        """Returns True if Candidate A Pareto-dominates Candidate B."""
        if not all_tasks:
            return cand_a.mean_score > cand_b.mean_score

        a_better_on_any = False
        for t in all_tasks:
            sa = cand_a.task_scores.get(t, 0.0)
            sb = cand_b.task_scores.get(t, 0.0)
            if sa < sb:
                return False
            if sa > sb:
                a_better_on_any = True
        return a_better_on_any

    @classmethod
    def get_frontier(cls, population: list[ParetoCandidate]) -> list[ParetoCandidate]:
        """Compute the subset of candidates not dominated by any other in the population."""
        if not population:
            return []

        all_tasks = sorted(list({t for c in population for t in c.task_scores.keys()}))
        frontier = []
        for cand in population:
            dominated = False
            for other in population:
                if other.id != cand.id and cls.dominates(other, cand, all_tasks):
                    dominated = True
                    break
            if not dominated:
                frontier.append(cand)
        return frontier or population


GEPA_REFLECTION_SYSTEM = """\
You are an expert prompt compiler and evolutionary agent architect.

You are given:
1. The current prompt component being mutated.
2. Concrete failure traces and natural language error feedback from agent runs.
3. Diagnostic signatures of what went wrong.

Rewrite ONLY the specified component to directly prevent these failure modes.
- Preserve domain-generality (do not hardcode specific task names or answers).
- Provide concrete, operational behavioral rules rather than vague encouragement.
- Keep the component concise (1-3 paragraphs, under 150 words).
- Output ONLY the updated component text with no extra commentary, preambles, or markdown quotes."""

GEPA_REFLECTION_USER = """\
## Component Name: {component_name}

## Current Component Text:
{current_component}

## Overall Task Performance:
Pass rate: {pass_rate:.1f}% ({n_success}/{n_total}) on {benchmark}

## Diagnostic Feedback & Traces:
{feedback_block}

Rewrite the {component_name} component now:"""


class GEPAPromptOptimizer:
    """Multi-component evolutionary prompt optimizer using Pareto selection and reflection."""

    def __init__(
        self,
        model: BaseChatModel,
        n_candidates: int = 3,
        max_population_size: int = 20,
        reflection_lm: BaseChatModel | None = None,
        seed: int = 42,
    ):
        self.model = model
        self.reflection_lm = reflection_lm or model
        self.n_candidates = int(n_candidates)
        self.max_population_size = max_population_size
        self.rng = random.Random(seed)
        self.population: list[ParetoCandidate] = []
        self.candidate_counter = 0

    def seed_population(self, base_prompt_text: str = BASE_SYSTEM_PROMPT) -> ParetoCandidate:
        """Seed the population with a base prompt."""
        mod = ModularPrompt.from_text(base_prompt_text)
        cand = ParetoCandidate(
            id="cand_0",
            modular_prompt=mod,
            parent_ids=[],
            generation=0,
        )
        self.population = [cand]
        self.candidate_counter = 1
        return cand

    def record_evaluations(
        self,
        candidate_id: str,
        trajectories: Sequence[Trajectory],
        validation_score: float | None = None,
    ) -> None:
        """Record task scores and failure feedback for a candidate."""
        cand = next((c for c in self.population if c.id == candidate_id), None)
        if not cand:
            return

        usable = [t for t in trajectories if t.usable]
        for t in usable:
            cand.task_scores[t.task_id] = float(t.score)
            if not t.success:
                feedback = (
                    f"Task {t.task_id} failed (score {t.score:.2f}, error='{t.error}'). "
                    f"Steps: {len(t.steps)}."
                )
                if t.steps:
                    last_step = t.steps[-1]
                    feedback += f" Last action: {last_step.action.render()} -> {last_step.observation[:120]}"
                cand.feedbacks.append(feedback)

        if validation_score is not None:
            cand.validation_score = validation_score

    def propose(
        self,
        current_prompt: str,
        trajectories: Sequence[Trajectory],
        generation: int = 0,
        benchmark: str = "",
    ) -> list[PromptCandidate]:
        """Propose candidate prompt variants via component-level reflection and crossover."""
        usable = [t for t in trajectories if t.usable]
        failures = [t for t in usable if not t.success]
        if not usable or not failures:
            return []

        # Ensure base candidate exists in population
        if not self.population:
            self.seed_population(current_prompt)

        # Update candidate 0 scores
        active_cand = self.population[-1]
        self.record_evaluations(active_cand.id, usable)

        # Compute Pareto frontier
        frontier = ParetoFrontier.get_frontier(self.population)
        candidates_to_evaluate: list[PromptCandidate] = []

        # Build diagnostic feedback
        n_total = len(usable)
        n_success = sum(1 for t in usable if t.success)
        pass_rate = (n_success / n_total) * 100.0 if n_total > 0 else 0.0

        feedback_snippets = []
        for f in failures[:4]:
            step_summary = []
            for s in f.steps[-3:]:
                step_summary.append(f"  - {s.action.render()} -> {s.observation[:150]}")
            step_text = "\n".join(step_summary)
            feedback_snippets.append(
                f"* Task {f.task_id} (error: {f.error or 'no error'}):\n{step_text}"
            )
        feedback_block = "\n".join(feedback_snippets)

        components = ["planning", "tool_policy", "error_recovery"]

        for _ in range(self.n_candidates):
            parent = self.rng.choice(frontier) if frontier else active_cand
            target_comp = self.rng.choice(components)

            # Mutate component via reflection
            mutated_prompt = parent.modular_prompt.copy()
            current_val = getattr(mutated_prompt, target_comp, "")

            user_msg = GEPA_REFLECTION_USER.format(
                component_name=target_comp,
                current_component=current_val,
                pass_rate=pass_rate,
                n_success=n_success,
                n_total=n_total,
                benchmark=benchmark or "benchmark",
                feedback_block=feedback_block,
            )

            try:
                reply = invoke_text(
                    self.reflection_lm,
                    [
                        SystemMessage(content=GEPA_REFLECTION_SYSTEM),
                        HumanMessage(content=user_msg),
                    ],
                ).strip()
                if reply and len(reply) > 20:
                    setattr(mutated_prompt, target_comp, reply)
            except Exception:
                # Fallback to local heuristic mutation if reflection call fails
                heuristic_patch = f"{current_val} Note: When stuck or blocked, inspect state before retrying."
                setattr(mutated_prompt, target_comp, heuristic_patch)

            cand_id = f"gen{generation}_cand{self.candidate_counter}"
            self.candidate_counter += 1

            pareto_cand = ParetoCandidate(
                id=cand_id,
                modular_prompt=mutated_prompt,
                parent_ids=[parent.id],
                generation=generation,
            )
            self.population.append(pareto_cand)

            rendered_text = mutated_prompt.render()
            candidates_to_evaluate.append(
                PromptCandidate(
                    text=rendered_text,
                    version=cand_id,
                    parent=parent.id,
                )
            )

        # If frontier has at least 2 parents, also propose a Crossover / Merge candidate
        if len(frontier) >= 2:
            p1, p2 = self.rng.sample(frontier, 2)
            merged_prompt = self.merge(p1.modular_prompt, p2.modular_prompt)
            cand_id = f"gen{generation}_merge{self.candidate_counter}"
            self.candidate_counter += 1

            merged_cand = ParetoCandidate(
                id=cand_id,
                modular_prompt=merged_prompt,
                parent_ids=[p1.id, p2.id],
                generation=generation,
            )
            self.population.append(merged_cand)
            candidates_to_evaluate.append(
                PromptCandidate(
                    text=merged_prompt.render(),
                    version=cand_id,
                    parent=f"{p1.id}+{p2.id}",
                )
            )

        # Cap population size
        if len(self.population) > self.max_population_size:
            # Keep top candidates by mean score and frontier membership
            frontier_ids = {c.id for c in frontier}
            non_frontier = [c for c in self.population if c.id not in frontier_ids]
            non_frontier.sort(key=lambda c: c.mean_score, reverse=True)
            self.population = frontier + non_frontier[: self.max_population_size - len(frontier)]

        return candidates_to_evaluate

    def merge(self, parent_a: ModularPrompt, parent_b: ModularPrompt) -> ModularPrompt:
        """Crossover: combine best sub-components from two distinct parents."""
        return ModularPrompt(
            core_role=self.rng.choice([parent_a.core_role, parent_b.core_role]),
            planning=parent_a.planning,
            tool_policy=parent_b.tool_policy,
            error_recovery=self.rng.choice([parent_a.error_recovery, parent_b.error_recovery]),
        )
