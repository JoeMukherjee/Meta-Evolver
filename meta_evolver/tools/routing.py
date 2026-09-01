"""Tool governance: choosing which tools the agent is allowed to see.

The third scaffold channel a self-improving agent can act on, alongside its
prompt and its memory. It matters more than it looks.

An agent's action space is its prompt. Every tool schema is tokens the model
reads before every decision, and a tool it should not use is a tool it will
eventually try. The version of this project that preceded the rewrite handed
the agent one global registry of eighteen tools -- web search, code execution,
document generation -- for an incident-response task that needs seven, and
episodes were routinely lost to probing capabilities the task had no use for.
Scoping tools to the environment fixed that. Routing is the next step: with a
large registry, even the *correct* environment can offer more than fits.

The policy here is deliberately conservative, because a wrongly-pruned tool
makes a task unsolvable while a wrongly-kept one only costs tokens:

* Below ``min_tools_to_route`` the full set passes through. Ranking seven
  tools spends an embedding call to change nothing.
* ``sticky`` tools are never dropped. Terminal actions -- submit, answer,
  finish -- are rarely the closest match to a task description, and an agent
  that cannot finish has already failed.
* Ranking is over the tool's name *and* description against the task plus the
  current observation, so relevance tracks the episode rather than being fixed
  at reset.

Routing is off unless a router is supplied. It is an optimization for large
registries, not a default.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from meta_evolver.llm.embeddings import Embedder, cosine


def tool_name(schema: dict[str, Any]) -> str:
    if schema.get("type") == "function":
        return str(schema.get("function", {}).get("name", ""))
    return str(schema.get("name", ""))


def tool_text(schema: dict[str, Any]) -> str:
    """Name, description and parameter names -- what the model reads."""
    body = schema.get("function", schema) if schema.get("type") == "function" else schema
    params = (body.get("parameters") or {}).get("properties") or {}
    return " ".join(
        [str(body.get("name", "")), str(body.get("description", "")), *params.keys()]
    )


class ToolRouter:
    """Ranks a tool registry against the task at hand and keeps the top ``k``."""

    #: Names that always survive routing, whatever the ranking says.
    DEFAULT_STICKY = (
        "submit_resolution",
        "submit",
        "answer",
        "finish",
        "done",
        "do",
    )

    def __init__(
        self,
        embedder: Embedder | None = None,
        k: int = 8,
        min_tools_to_route: int = 10,
        sticky: Sequence[str] = DEFAULT_STICKY,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.k = int(k)
        self.min_tools_to_route = int(min_tools_to_route)
        self.sticky = {s.lower() for s in sticky}
        self._cache: dict[str, list[float]] = {}

    def select(
        self, tools: Sequence[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """The subset of ``tools`` worth showing the model for ``query``.

        Order is preserved from the input so a run stays reproducible: sorting
        by score would make the prompt sensitive to embedding noise, and tool
        order measurably affects which tool a model picks.
        """
        tools = list(tools)
        if len(tools) <= max(self.k, self.min_tools_to_route):
            return tools

        query_vec = self.embedder.embed_one(query[:2000])
        scored: list[tuple[float, int]] = []
        for i, schema in enumerate(tools):
            name = tool_name(schema).lower()
            if name in self.sticky:
                scored.append((float("inf"), i))
                continue
            text = tool_text(schema)
            if text not in self._cache:
                self._cache[text] = self.embedder.embed_one(text)
            scored.append((cosine(query_vec, self._cache[text]), i))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        keep = {i for _, i in scored[: self.k]}
        return [schema for i, schema in enumerate(tools) if i in keep]
