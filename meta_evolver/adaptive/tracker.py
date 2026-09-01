"""Symbolic entity and state-space tracker for exploration completeness."""
from __future__ import annotations
import re
from typing import List, Set

_ENTITY_RE = re.compile(r"(?:go to|open|close|examine|take .+? from|put .+? in/on|use .+? on)\s+([a-zA-Z0-9_\s]+)", re.IGNORECASE)

class EntityStateTracker:
    """Tracks visited, unvisited, and admissible entities across an embodied trajectory."""
    def __init__(self) -> None:
        self.visited_entities: Set[str] = set()
        self.known_receptacles: Set[str] = set()

    def update_from_action(self, action: str) -> None:
        m = _ENTITY_RE.search(action.strip())
        if m:
            target = m.group(1).strip().lower()
            self.visited_entities.add(target)

    def update_from_admissible(self, admissible_commands: List[str]) -> None:
        for cmd in admissible_commands:
            m = _ENTITY_RE.search(cmd.strip())
            if m:
                target = m.group(1).strip().lower()
                self.known_receptacles.add(target)

    def get_unvisited_entities(self) -> List[str]:
        return sorted(list(self.known_receptacles - self.visited_entities))

    def reset(self) -> None:
        self.visited_entities.clear()
        self.known_receptacles.clear()
