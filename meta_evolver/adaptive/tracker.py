"""Symbolic state tracking for exploration completeness.

Cheap, model-free bookkeeping that answers one question the LLM is
consistently bad at over a long context: *where have I already been?* An agent
that has examined eight containers and found nothing will happily examine the
ninth-that-is-really-the-third, because the earlier visit is thirty messages
back. Tracking it symbolically costs nothing and turns "search harder" into
"search where you have not looked".

Entity extraction is regex-based and covers two idioms:

  * text-game commands  -- ``go to fridge 1``, ``open drawer 3``
  * tool calls          -- ``inspect_service_logs(service_name=auth-service)``

Both reduce to "an action that targets a named thing", which is all the
exploration logic needs.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

#: Verbs that mean "I looked at / went to this thing".
_VISIT_VERBS = (
    "go to",
    "open",
    "close",
    "examine",
    "inspect",
    "search",
    "look at",
    "move to",
    "visit",
    "read",
    "query",
)

_TEXT_ACTION_RE = re.compile(
    r"^\s*(?:{verbs})\s+(?:the\s+)?([a-z0-9][a-z0-9_\-\. ]{{0,48}}?)\s*$".format(
        verbs="|".join(re.escape(v) for v in _VISIT_VERBS)
    ),
    re.IGNORECASE,
)

# `take X from Y` visits Y; `put X in/on Y` visits Y.
_PREP_ACTION_RE = re.compile(
    r"\b(?:take|put|move|place)\b.*?\b(?:from|in|on|into|onto)\s+([a-z0-9][a-z0-9_\-\. ]{0,48})$",
    re.IGNORECASE,
)

# Tool-call form: name(arg=value, ...). Any value that looks like an entity id.
_TOOL_CALL_RE = re.compile(r"^([a-z_][a-z0-9_]*)\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)
_TOOL_TARGET_KEYS = (
    "service_name",
    "service",
    "target",
    "name",
    "path",
    "file_path",
    "table",
    "url",
    "endpoint",
)


def extract_entity(action_text: str) -> str | None:
    """The thing an action targets, or ``None`` if it targets nothing."""
    text = (action_text or "").strip()
    if not text:
        return None

    match = _TEXT_ACTION_RE.match(text)
    if match:
        return _normalize(match.group(1))

    match = _PREP_ACTION_RE.search(text)
    if match:
        return _normalize(match.group(1))

    match = _TOOL_CALL_RE.match(text)
    if match:
        tool_name, arg_blob = match.group(1).lower(), match.group(2)
        if not any(v.replace(" ", "_") in tool_name for v in ("inspect", "query", "read", "search", "get", "list")):
            return None
        for key in _TOOL_TARGET_KEYS:
            found = re.search(rf'"?{key}"?\s*[:=]\s*"?([^",}}\)]+)', arg_blob, re.IGNORECASE)
            if found:
                return _normalize(f"{tool_name}:{found.group(1)}")
    return None


def _normalize(value: str) -> str:
    return " ".join(value.strip().strip("\"'.,").lower().split())


class EntityStateTracker:
    """Tracks which entities an episode has visited and which remain."""

    def __init__(self) -> None:
        self.visited: set[str] = set()
        self.known: set[str] = set()
        self.visit_counts: dict[str, int] = {}

    def record_action(self, action_text: str) -> str | None:
        entity = extract_entity(action_text)
        if entity:
            self.visited.add(entity)
            self.known.add(entity)
            self.visit_counts[entity] = self.visit_counts.get(entity, 0) + 1
        return entity

    def record_candidates(self, commands: Iterable[str]) -> None:
        """Learn the entity universe from the admissible-action list."""
        for cmd in commands:
            entity = extract_entity(cmd)
            if entity:
                self.known.add(entity)

    def unvisited(self) -> list[str]:
        return sorted(self.known - self.visited)

    def revisited(self, threshold: int = 2) -> list[str]:
        return sorted(e for e, n in self.visit_counts.items() if n >= threshold)

    def coverage(self) -> float:
        """Fraction of the known entity universe already visited."""
        if not self.known:
            return 0.0
        return len(self.visited & self.known) / len(self.known)

    def reset(self) -> None:
        self.visited.clear()
        self.known.clear()
        self.visit_counts.clear()
