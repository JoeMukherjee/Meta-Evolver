"""OPRO-style prompt evolution with empirical selection.

Optimization by PROmpting: show a meta-model the current instruction, the
measured score, and the traces of what went wrong; ask for a better
instruction. The mechanism works, with one caveat that decides whether it
helps or hurts over many generations -- **a proposed prompt is a hypothesis,
not an improvement.**

So this optimizer proposes several candidates and returns them for
*measurement*. The evolution graph runs each on a held-out validation split and
keeps the winner only if it beats the incumbent by a margin; otherwise the
incumbent survives. Without that guard, prompt drift compounds: each
generation rewrites a prompt that was never shown to be worse, and by
generation five the agent is running an instruction nobody ever validated.

Two further guards, both learned from watching this fail:

* Candidates that drop ``{memory_section}`` or ``{guidance_section}`` are
  repaired, not discarded -- a good prompt that forgot a placeholder is worth
  keeping.
* Failure traces are sampled across *distinct tasks*. Five traces of one
  pathological task produce a prompt overfitted to that task.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from meta_evolver.core.types import Trajectory
from meta_evolver.llm.client import LLMError, invoke_text
from meta_evolver.prompts.templates import ensure_placeholders

OPTIMIZER_SYSTEM = """\
You improve the system instructions of autonomous tool-using agents.

You are given the current instruction, its measured performance, and traces of
episodes where the agent failed. Diagnose the *behavioural* cause of the
failures -- what the agent did, or failed to do, that a better instruction
would have changed -- and rewrite the instruction to prevent it.

Constraints:
- Keep the placeholders {memory_section} and {guidance_section} exactly as
  written, on their own lines. They are filled at runtime.
- Stay domain-general. Do not hardcode task-specific values, service names, or
  answers from the traces; those change every episode and would mislead.
- Prefer a decision rule the agent can apply mid-episode ("before X, confirm
  Y") over exhortation ("be careful", "think step by step").
- Keep it under 400 words. A longer instruction is not a better one.
- Reply with the instruction text only: no preamble, no commentary, no fences."""

OPTIMIZER_USER = """\
## Current instruction
{current_prompt}

## Measured performance
Pass rate: {pass_rate:.1f}% ({n_success}/{n_total}) over {benchmark}
Average steps: {avg_steps:.1f}   Average score: {avg_score:.2f}

## Recurring failure signatures
{signatures}

## Failure traces
{failures}

{diversity_hint}
Write the improved instruction now."""


@dataclass
class PromptCandidate:
    text: str
    version: str
    parent: str = "base"
    score: float | None = None
    """Validation pass rate; ``None`` until measured."""


class PromptOptimizer:
    """Proposes candidate system prompts from failure evidence."""

    def __init__(
        self,
        model: BaseChatModel,
        n_candidates: int = 2,
        max_failures_in_prompt: int = 4,
        max_steps_per_trace: int = 10,
    ) -> None:
        self.model = model
        self.n_candidates = int(n_candidates)
        self.max_failures_in_prompt = int(max_failures_in_prompt)
        self.max_steps_per_trace = int(max_steps_per_trace)
        self.last_error: str = ""

    # -- proposal ----------------------------------------------------------

    def propose(
        self,
        current_prompt: str,
        trajectories: Sequence[Trajectory],
        generation: int = 0,
        benchmark: str = "",
    ) -> list[PromptCandidate]:
        """Generate candidate replacements for ``current_prompt``.

        Returns an empty list when there is nothing to learn from -- no usable
        episodes, or no failures. Rewriting a prompt that is winning is a
        coin-flip that can only cost.
        """
        usable = [t for t in trajectories if t.usable]
        failures = [t for t in usable if not t.success]
        if not usable or not failures:
            return []

        n_total = len(usable)
        n_success = sum(1 for t in usable if t.success)
        common = {
            "current_prompt": current_prompt,
            "pass_rate": 100.0 * n_success / n_total,
            "n_success": n_success,
            "n_total": n_total,
            "benchmark": benchmark or "this benchmark",
            "avg_steps": sum(t.n_steps for t in usable) / n_total,
            "avg_score": sum(t.score for t in usable) / n_total,
            "signatures": _render_signatures(failures),
            "failures": self._render_failures(failures),
        }

        candidates: list[PromptCandidate] = []
        for i in range(self.n_candidates):
            # Candidate 0 is the straight rewrite. Later ones are pushed toward
            # a different fix, so validation compares real alternatives instead
            # of paraphrases of one idea.
            hint = (
                ""
                if i == 0
                else (
                    f"This is attempt {i + 1}. Earlier attempts addressed the most "
                    "obvious cause; target a DIFFERENT failure mode from the traces "
                    "above.\n\n"
                )
            )
            body = OPTIMIZER_USER.format(**common, diversity_hint=hint)
            try:
                content = invoke_text(
                    self.model,
                    [SystemMessage(content=OPTIMIZER_SYSTEM), HumanMessage(content=body)],
                )
            except LLMError as exc:
                self.last_error = str(exc)
                break

            text = _clean(content)
            if len(text) < 80:
                continue
            candidates.append(
                PromptCandidate(
                    text=ensure_placeholders(text),
                    version=f"g{generation}c{i}",
                    parent=f"g{max(0, generation - 1)}",
                )
            )
        return candidates

    def _render_failures(self, failures: Sequence[Trajectory]) -> str:
        # One trace per distinct task, so the sample spans failure modes rather
        # than repeating the single worst task.
        by_task: dict[str, Trajectory] = {}
        for t in failures:
            by_task.setdefault(t.task_id, t)
        chosen = list(by_task.values())[: self.max_failures_in_prompt]
        return "\n\n".join(t.render(max_steps=self.max_steps_per_trace) for t in chosen)


# ---------------------------------------------------------------------------
# Failure-signature mining (no model call -- cheap, and it grounds the prompt)
# ---------------------------------------------------------------------------


def _render_signatures(failures: Sequence[Trajectory]) -> str:
    """Aggregate structural failure patterns across the failed episodes.

    Computed rather than asked for: counts of repeated actions, blocked
    submissions and budget exhaustion are facts the meta-model would otherwise
    have to infer from truncated traces, and it infers them unreliably.
    """
    if not failures:
        return "(none)"

    n = len(failures)
    truncated = sum(1 for t in failures if t.steps and t.steps[-1].truncated)
    blocked = sum(1 for t in failures if any(s.blocked for s in t.steps))
    perturbed = sum(1 for t in failures if any(s.perturbed for s in t.steps))

    loops = 0
    action_counts: dict[str, int] = {}
    for t in failures:
        rendered = [s.action.render() for s in t.steps]
        if len(rendered) != len(set(rendered)):
            loops += 1
        for name in {s.action.name for s in t.steps}:
            action_counts[name] = action_counts.get(name, 0) + 1

    never_used = sorted(
        name
        for name in {"verify", "run_healthcheck", "evaluate", "check"}
        if action_counts.get(name, 0) == 0
    )

    lines = [
        f"- {n} failed episodes analysed.",
        f"- {loops}/{n} repeated an identical action (search loop).",
        f"- {truncated}/{n} ran out of step budget without finishing.",
    ]
    if blocked:
        lines.append(f"- {blocked}/{n} had an action rejected by a validation guard.")
    if perturbed:
        lines.append(f"- {perturbed}/{n} hit an injected fault; check whether it was retried.")
    if never_used:
        lines.append(f"- Verification-style actions never invoked: {', '.join(never_used)}.")
    return "\n".join(lines)


def _clean(text: str) -> str:
    """Strip fences and conversational preamble from a model reply."""
    out = (text or "").strip()
    fenced = re.match(r"^```(?:\w+)?\s*\n(.*?)\n?```$", out, re.DOTALL)
    if fenced:
        out = fenced.group(1).strip()
    out = re.sub(
        r"^(?:here(?:'s| is)[^\n]*|sure[^\n]*|improved instruction:?)\n+",
        "",
        out,
        flags=re.IGNORECASE,
    )
    return out.strip()
