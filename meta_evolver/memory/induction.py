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
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from meta_evolver.core.types import MemoryItem, Trajectory
from meta_evolver.llm.client import LLMError, invoke_text

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

        # Prefer a mix: failures teach most, successes confirm what works.
        failures = [t for t in usable if not t.success]
        successes = [t for t in usable if t.success]
        half = max(1, self.max_traces_in_prompt // 2)
        chosen = (failures[:half] + successes[:half])[: self.max_traces_in_prompt]
        if not chosen:
            chosen = usable[: self.max_traces_in_prompt]

        traces = "\n\n".join(
            f"--- trace {i} ---\n{t.render(max_steps=self.max_steps_per_trace)}"
            for i, t in enumerate(chosen, 1)
        )
        prompt = INDUCTION_USER.format(
            benchmark=benchmark or (chosen[0].benchmark if chosen else ""),
            n_success=len(successes),
            n_failure=len(failures),
            n_total=len(usable),
            traces=traces,
            max_items=self.max_items_per_batch,
        )

        try:
            content = invoke_text(
                self.model,
                [SystemMessage(content=INDUCTION_SYSTEM), HumanMessage(content=prompt)],
            )
        except LLMError as exc:
            self.last_error = str(exc)
            return []

        payload = _parse_json_object(content)
        if not payload:
            self.last_error = "induction returned unparseable JSON"
            return []

        source_ids = [t.task_id for t in chosen]
        out: list[MemoryItem] = []
        for raw in payload.get("memories", [])[: self.max_items_per_batch]:
            if not isinstance(raw, dict):
                continue
            lesson = str(raw.get("lesson", "")).strip()
            if not lesson:
                continue
            polarity = raw.get("polarity")
            out.append(
                MemoryItem(
                    title=str(raw.get("title", ""))[:80] or lesson[:70],
                    scenario=_slug(raw.get("scenario", "general")),
                    polarity="failure" if polarity == "failure" else "success",
                    lesson=lesson,
                    procedure=str(raw.get("procedure", "")).strip(),
                    triggers=[str(t) for t in (raw.get("triggers") or []) if str(t).strip()][:8],
                    source_task_ids=source_ids,
                    benchmark=benchmark,
                )
            )
        return out


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
