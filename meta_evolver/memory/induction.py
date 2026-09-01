"""Memory induction: turning trajectories into reusable strategies.

Distils each episode into at most one durable lesson. Two design choices are
worth stating because both were wrong in the version this replaces.

**Failures are induced, not skipped.** A failed episode carries the sharper
signal -- it names a specific dead end. Those become ``polarity="failure"``
anti-patterns and render differently in the prompt, so the agent reads them as
"do not do this" rather than as a procedure to follow.

**Induction is batched over an episode set, not per episode.** Reflecting on
one trajectory produces lessons that restate that task ("restart db-proxy
after raising max_connections"). Reflecting on several at once forces the
model to find what they share, which is the part that transfers. When only one
trajectory is available it falls back to single-episode reflection.

**Contrastive pairs get priority.** When several rollouts of the *same* task
disagree -- one succeeded, another failed -- that pair localises the decision
that mattered far better than any number of independent episodes can. This is
the synthesis half of memory-aware test-time scaling (ReasoningBank, MaTTS):
the extra rollouts are worth their cost only if induction actually exploits
the disagreement, so those traces are surfaced first and labelled as a
contrast.

Output uses the provider's own structured-output enforcement where available,
falling back to lenient JSON parsing where it is not -- a scripted or local
model may not support it, and induction failing entirely would silently stop
the bank from growing.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from meta_evolver.core.types import MemoryItem, Trajectory
from meta_evolver.llm.client import LLMError, invoke_text


class InducedMemory(BaseModel):
    """One distilled strategy, as the model is asked to return it."""

    title: str = Field(description="Short imperative name for the strategy.")
    scenario: str = Field(
        description="lowercase_snake category, e.g. resource_limit, stale_config, search_order"
    )
    polarity: Literal["success", "failure"] = Field(
        description="'failure' for an anti-pattern -- a mistake to avoid."
    )
    lesson: str = Field(
        description="One or two sentences: the generalizable insight, or the mistake to avoid."
    )
    procedure: str = Field(description="Numbered steps a future agent can execute.")
    triggers: list[str] = Field(
        default_factory=list, description="Observable symptoms or keywords that signal this case."
    )


class InducedMemories(BaseModel):
    """The batch a single induction call returns."""

    memories: list[InducedMemory] = Field(default_factory=list)

INDUCTION_SYSTEM = """\
You distil agent execution traces into reusable strategy memories for future agents.

A good memory is TRANSFERABLE: it names the class of situation and the ordering
constraint or checkable invariant that governs it. A bad memory restates one
episode's specifics.

  Bad:  "Set max_connections to 20 on db-proxy."
  Good: "When a service times out acquiring pooled connections while the pool
         reports a full queue, the limit -- not the caller -- is the constraint.
         Raise it, restart the owner, then re-verify before concluding."

Reply with JSON only, no prose and no code fences."""

INDUCTION_USER = """\
Benchmark: {benchmark}
Outcome mix: {n_success} succeeded, {n_failure} failed out of {n_total}.

{contrast_note}
{traces}

Induce at most {max_items} memories that would raise the success rate of a
future agent on this benchmark. Prefer few, general, and specific over many
and vague. Cover the failures: each recurring failure mode deserves one
anti-pattern memory.

JSON schema:
{{
  "memories": [
    {{
      "title": "short imperative name",
      "scenario": "lowercase_snake category, e.g. resource_limit, stale_config, search_order",
      "polarity": "success" | "failure",
      "lesson": "one or two sentences: the generalizable insight or the mistake to avoid",
      "procedure": "numbered steps a future agent can execute",
      "triggers": ["observable symptom or keyword", "..."]
    }}
  ]
}}"""


class MemoryInducer:
    """Reflects over trajectories and emits :class:`MemoryItem` objects."""

    def __init__(
        self,
        model: BaseChatModel,
        max_items_per_batch: int = 3,
        max_traces_in_prompt: int = 6,
        max_steps_per_trace: int = 14,
    ) -> None:
        self.model = model
        self.max_items_per_batch = int(max_items_per_batch)
        self.max_traces_in_prompt = int(max_traces_in_prompt)
        self.max_steps_per_trace = int(max_steps_per_trace)
        self.last_error: str = ""
        #: None until the first call decides whether this model supports
        #: provider-enforced structured output. Probed once, not per call.
        self._structured: bool | None = None

    def induce(
        self, trajectories: Sequence[Trajectory], benchmark: str = ""
    ) -> list[MemoryItem]:
        """Distil a batch of episodes into memories.

        Episodes that ended in an infrastructure error are dropped: their
        traces describe a broken API call, not a reasoning mistake, and
        letting them through pollutes the bank with lessons about retrying
        HTTP requests.
        """
        usable = [t for t in trajectories if t.usable]
        if not usable:
            return []

        failures = [t for t in usable if not t.success]
        successes = [t for t in usable if t.success]

        contrasts = _contrastive_pairs(usable)
        chosen, traces = self._render_traces(usable, failures, successes, contrasts)
        prompt = INDUCTION_USER.format(
            benchmark=benchmark or (chosen[0].benchmark if chosen else ""),
            n_success=len(successes),
            n_failure=len(failures),
            n_total=len(usable),
            traces=traces,
            max_items=self.max_items_per_batch,
            contrast_note=(
                "\nSome traces below are paired: the SAME task, attempted more than once, "
                "with different outcomes. A pair localises the decision that actually "
                "mattered far better than unrelated episodes can -- start there.\n"
                if contrasts
                else ""
            ),
        )

        messages = [SystemMessage(content=INDUCTION_SYSTEM), HumanMessage(content=prompt)]
        raw_items = self._invoke(messages)
        if raw_items is None:
            return []

        source_ids = sorted({t.task_id for t in chosen})
        out: list[MemoryItem] = []
        for raw in raw_items[: self.max_items_per_batch]:
            lesson = str(raw.get("lesson", "")).strip()
            if not lesson:
                continue
            item = MemoryItem(
                    title=str(raw.get("title", ""))[:80] or lesson[:70],
                    scenario=_slug(raw.get("scenario", "general")),
                    polarity="failure" if raw.get("polarity") == "failure" else "success",
                    lesson=lesson,
                    procedure=str(raw.get("procedure", "")).strip(),
                    triggers=[str(t) for t in (raw.get("triggers") or []) if str(t).strip()][:8],
                    source_task_ids=source_ids,
                    benchmark=benchmark,
            )
            # The id is a pure function of scenario + lesson, so it is settled
            # here rather than at insertion. Anything that observes a memory
            # before it reaches the bank -- the causal graph, a log line --
            # then names it the same way the bank will.
            item.id = f"mem-{item.key()}"
            out.append(item)
        return out

    def _invoke(self, messages: list) -> list[dict] | None:
        """Run one induction call, preferring provider-enforced schemas.

        Structured output removes a whole class of failure -- fenced JSON,
        prose preambles, trailing commas -- by making the provider validate
        against the schema. Not every model supports it, so the first
        NotImplementedError switches this inducer to text parsing for the rest
        of the run rather than retrying a capability that will not appear.
        """
        if self._structured is not False:
            try:
                structured = self.model.with_structured_output(InducedMemories)
                result = structured.invoke(messages)
                self._structured = True
                if isinstance(result, InducedMemories):
                    return [m.model_dump() for m in result.memories]
                if isinstance(result, dict):
                    return [m for m in result.get("memories", []) if isinstance(m, dict)]
            except NotImplementedError:
                self._structured = False
            except Exception as exc:
                # A provider that advertises structured output can still fail
                # a single call on it. Fall through to text for this call, but
                # do not give up on the capability.
                self.last_error = f"structured output failed, retrying as text: {exc}"

        try:
            content = invoke_text(self.model, messages)
        except LLMError as exc:
            self.last_error = str(exc)
            return None

        payload = _parse_json_object(content)
        if not payload:
            self.last_error = "induction returned unparseable JSON"
            return None
        return [m for m in payload.get("memories", []) if isinstance(m, dict)]

    def _render_traces(
        self,
        usable: Sequence[Trajectory],
        failures: Sequence[Trajectory],
        successes: Sequence[Trajectory],
        contrasts: Sequence[tuple[Trajectory, Trajectory]],
    ) -> tuple[list[Trajectory], str]:
        """Pick the traces to reflect over, contrastive pairs first."""
        chosen: list[Trajectory] = []
        blocks: list[str] = []

        for won, lost in contrasts:
            if len(chosen) + 2 > self.max_traces_in_prompt:
                break
            chosen.extend([won, lost])
            blocks.append(
                f"--- contrastive pair on task {won.task_id} ---\n"
                f"[ATTEMPT THAT SUCCEEDED]\n{won.render(max_steps=self.max_steps_per_trace)}\n\n"
                f"[ATTEMPT THAT FAILED]\n{lost.render(max_steps=self.max_steps_per_trace)}"
            )

        # Fill the remaining budget with a failure/success mix: failures teach
        # most, successes confirm what works.
        seen = {id(t) for t in chosen}
        remaining = self.max_traces_in_prompt - len(chosen)
        if remaining > 0:
            half = max(1, remaining // 2)
            extra = [t for t in failures if id(t) not in seen][:half]
            extra += [t for t in successes if id(t) not in seen][: remaining - len(extra)]
            if not extra and not chosen:
                extra = list(usable[:remaining])
            for i, trace in enumerate(extra, len(blocks) + 1):
                chosen.append(trace)
                blocks.append(f"--- trace {i} ---\n{trace.render(max_steps=self.max_steps_per_trace)}")

        return chosen, "\n\n".join(blocks)


def _contrastive_pairs(
    trajectories: Sequence[Trajectory],
) -> list[tuple[Trajectory, Trajectory]]:
    """Same task, different outcome -- one pair per task, best against worst.

    Only produced when a generation ran more than one rollout per task. The
    pair is what makes extra rollouts pay for themselves: two attempts at the
    same problem that diverged isolate the decision responsible, which no
    number of unrelated episodes can.
    """
    by_task: dict[str, list[Trajectory]] = {}
    for trajectory in trajectories:
        by_task.setdefault(trajectory.task_id, []).append(trajectory)

    pairs: list[tuple[Trajectory, Trajectory]] = []
    for attempts in by_task.values():
        won = [t for t in attempts if t.success]
        lost = [t for t in attempts if not t.success]
        if won and lost:
            # Shortest success against longest failure: the widest gap in
            # behaviour, so the contrast is easiest to attribute.
            pairs.append((min(won, key=lambda t: t.n_steps), max(lost, key=lambda t: t.n_steps)))
    return pairs


def _slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value).lower()).strip("_")
    return text or "general"


def _parse_json_object(text: str) -> dict | None:
    """Parse a JSON object out of a model reply.

    Models wrap JSON in prose or fences often enough that a bare
    ``json.loads`` throws away good responses, so fall back to the outermost
    brace-balanced span.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
