"""Auto-creation: mine trajectories for action patterns worth keeping.

The question is not "what did a successful episode do" -- every successful
episode did *something* -- but "what do successes do that failures do not".
So a pattern (a contiguous run of abstract actions) becomes a skill candidate
only if it

* appears in successful episodes of at least ``min_support`` *distinct tasks*
  (one task repeated is memorisation, not a skill), and
* has precision >= ``min_precision``: of the distinct tasks it appeared in, at
  least that fraction succeeded. A step every episode does -- including the
  failures -- has precision ~ the base rate and is filtered out.

Maximal patterns win: ``go>take>heat`` subsumes ``go>take`` when it holds at
least the same support. No model call; naming and wording are mechanical, and
an optional ``refiner`` (an LLM that rewrites the description from examples)
plugs in at the library level.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from meta_evolver.core.types import Action, Trajectory
from meta_evolver.knowledge.graph import slug
from meta_evolver.knowledge.ingest import action_verb, extract_concepts
from meta_evolver.skills.models import Skill

_NUM = re.compile(r"\b\d+\b")
_ERRORISH = re.compile(
    r"\b(error|exception|traceback|failed|failure|cannot|can't|unable|not found|invalid|denied|timed? ?out|nothing happens)\b",
    re.I,
)


def error_fraction(t: Trajectory) -> float:
    """Share of an episode's steps whose observation reads like an error."""
    if not t.steps:
        return 1.0
    return sum(bool(_ERRORISH.search(s.observation or "")) for s in t.steps) / len(t.steps)


def is_clean_success(t: Trajectory, max_error_fraction: float = 0.34) -> bool:
    """Is this episode a procedure worth learning from?

    The 'do not capture' rules Hermes's skill reviewer applies, made
    mechanical: a success that flailed through errors, or whose *last* step
    was an error, is a narrative of one lucky recovery -- not a workflow --
    and an environment failure that resolved is not a skill either.
    """
    if not (t.success and t.usable):
        return False
    last = t.steps[-1].observation or ""
    return not _ERRORISH.search(last) and error_fraction(t) <= max_error_fraction


def verb_sequence(t: Trajectory) -> list[str]:
    """Abstract actions with immediate repeats collapsed (``go go go`` -> ``go``)."""
    seq: list[str] = []
    for s in t.steps:
        v = action_verb(s.action.name, s.action.kwargs)
        if not seq or seq[-1] != v:
            seq.append(v)
    return seq


def step_template(action: Action) -> str:
    """A step with its episode-specific arguments abstracted to placeholders."""
    if action.name == "do" and isinstance(action.kwargs.get("text"), str):
        return _NUM.sub("<n>", action.kwargs["text"])
    args = ", ".join(f"{k}=<{k}>" for k in sorted(action.kwargs))
    return f"{action.name}({args})"


def _ngrams(seq: list[str], lo: int, hi: int) -> set[tuple[str, ...]]:
    return {tuple(seq[i : i + n]) for n in range(lo, hi + 1) for i in range(len(seq) - n + 1)}


def _contains(big: tuple[str, ...], small: tuple[str, ...]) -> bool:
    n = len(small)
    return any(big[i : i + n] == small for i in range(len(big) - n + 1))


def mine_skills(
    trajectories: Iterable[Trajectory],
    generation: int = 0,
    min_support: int = 2,
    min_precision: float = 0.6,
    min_len: int = 2,
    max_len: int = 6,
    max_skills: int = 5,
    max_error_fraction: float = 0.34,
) -> list[Skill]:
    # Failures stay in (they define precision); only *successes* are held to
    # the clean-procedure bar before they can vouch for a pattern.
    usable = [
        t for t in trajectories
        if t.usable and (not t.success or is_clean_success(t, max_error_fraction))
    ]
    succ_tasks: dict[tuple[str, ...], set[str]] = defaultdict(set)
    fail_tasks: dict[tuple[str, ...], set[str]] = defaultdict(set)
    example: dict[tuple[str, ...], Trajectory] = {}
    for t in usable:
        for g in _ngrams(verb_sequence(t), min_len, max_len):
            (succ_tasks if t.success else fail_tasks)[g].add(t.task_id)
            if t.success:
                example.setdefault(g, t)

    kept: list[tuple[str, ...]] = []
    for g, tasks in sorted(succ_tasks.items(), key=lambda kv: (-len(kv[0]), -len(kv[1]), kv[0])):
        n_s, n_f = len(tasks), len(fail_tasks.get(g, ()))
        if n_s < min_support or n_s / (n_s + n_f) < min_precision:
            continue
        # Subsumed by a longer kept pattern that is at least as well supported.
        if any(_contains(big, g) and len(succ_tasks[big]) >= n_s for big in kept):
            continue
        kept.append(g)
        if len(kept) >= max_skills:
            break

    return [_build(g, succ_tasks[g], fail_tasks.get(g, set()), example[g], usable, generation) for g in kept]


def _build(g, wins, losses, ex: Trajectory, usable: list[Trajectory], generation: int) -> Skill:
    # Concepts shared by the tasks this pattern won on: the preconditions.
    counts: Counter[str] = Counter()
    for t in usable:
        if t.success and t.task_id in wins:
            counts.update(extract_concepts(t.instruction))
    need = max(1, (len(wins) + 1) // 2)
    # Class-level triggers only: no digits (an id or a version is a one-off).
    triggers = [c for c, n in counts.most_common(8) if n >= need and not any(ch.isdigit() for ch in c)][:5]

    # Steps: the matching window of the example episode, generalised.
    actions = [s.action for s in ex.steps]
    verbs = [action_verb(a.name, a.kwargs) for a in actions]
    start = next((i for i in range(len(verbs)) if tuple(_dedupe(verbs[i:]))[: len(g)] == g), 0)
    window, prev = [], None
    for a, v in zip(actions[start:], verbs[start:]):
        if v == prev:
            continue
        window.append(a)
        prev = v
        if len(window) == len(g):
            break

    name = slug("-".join([*g[:3], triggers[0]] if triggers else g[:3]))
    return Skill(
        name=name,
        description=(
            f"Pattern '{' -> '.join(g)}' that succeeded on {len(wins)} distinct task(s)"
            + (f" and failed on {len(losses)}" if losses else "")
            + (f" in {ex.benchmark}" if ex.benchmark else "")
        ),
        when_to_use=triggers,
        steps=[step_template(a) for a in window],
        tools=list(dict.fromkeys(g)),
        signature=">".join(g),
        benchmark=ex.benchmark,
        origin_task_ids=sorted(wins)[:20],
        created_generation=generation,
    )


def _dedupe(seq: list[str]) -> list[str]:
    out: list[str] = []
    for v in seq:
        if not out or out[-1] != v:
            out.append(v)
    return out
