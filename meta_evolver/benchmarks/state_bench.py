"""STATE-Bench integration -- meta-evolver as a Custom Client + Agent.

Microsoft, May 2026. https://github.com/microsoft/STATE-Bench

STATE-Bench is architecturally inverted relative to every other adapter in
this package. LongMemEval and LoCoMo are ``BenchmarkAdapter``s that
meta-evolver's own episode graph drives. STATE-Bench drives itself: its own
harness runs the multi-turn loop against a protocol-locked user simulator and
a task-local sandbox database, and calls *into* a ``BaseAgent`` subclass you
provide for each turn (``docs/USE_CUSTOM_CLIENT.md`` in the official repo).
There is no ``ActionableEnv`` here to write -- this module provides the two
extension points their harness calls: a ``BaseLLMClient`` wrapping
meta-evolver's model, and a ``BaseAgent`` wrapping meta-evolver's memory bank
as a read-only retrieval tool.

That retrieval tool is the actual integration point. STATE-Bench's Agent
Learning Track scores whether *learnings extracted from past trajectories*
improve performance on held-out test tasks, via one hook:
``memory_tool_schemas()``/``memory_tool_handlers()``. This wires that hook
straight to :class:`~meta_evolver.memory.bank.ReasoningMemoryBank` -- the
same Beta(1,1)-scored bank every other benchmark in this project uses, not a
parallel implementation.

Protocol discipline the official rules require, preserved here deliberately:
learnings must come from ``datasets/train_task_trajectories/`` only, never
from held-out test tasks. :func:`build_memory_from_train_trajectories` is the
*offline* step that reads those train trajectories and populates a bank
before a scored run starts; the agent itself only ever retrieves during
evaluation (``memory_tool_handlers`` is read-only, matching the rule "memory
retrieval tools must be read-only" verbatim), never writes.

Setup
-----
This module has no hard dependency on the ``state_bench`` package -- it
imports it lazily, so ``import meta_evolver`` never breaks for someone not
using this benchmark. To actually run it, STATE-Bench's own loader requires
your agent/client classes to live as ``.py`` files under *its* repo root
(``docs/USE_CUSTOM_CLIENT.md``: "Files are imported by concrete path... under
repo-root extension folders"). Drop two one-line re-export shims there rather
than duplicating logic:

    # <state-bench-repo>/clients/meta_evolver_client.py
    from meta_evolver.benchmarks.state_bench import MetaEvolverStateBenchClient

    # <state-bench-repo>/agents/meta_evolver_agent.py
    from meta_evolver.benchmarks.state_bench import MetaEvolverStateBenchAgent

Then, from the STATE-Bench repo root::

    uv run python -m state_bench.scripts.run_batch \\
      --domain travel --agent-class MetaEvolverStateBenchAgent \\
      --agent-client-class MetaEvolverStateBenchClient \\
      --agent-model-name <model-name> --num-runs 5 --retrieve-learnings-top-k 3 \\
      --output-dir outputs/travel/

Model selection uses the same ``$META_EVOLVER_MODEL`` env var as the rest of
this project (:func:`~meta_evolver.llm.client.effective_model`), or
``META_EVOLVER_STATE_BENCH_MODEL`` specifically for this integration.

Tool schema note: STATE-Bench's domain tools use the flat Responses-API shape
(``{"type": "function", "name": ..., "parameters": ...}``), while this
project's :func:`~meta_evolver.benchmarks.base.tool_schema` and LangChain's
``bind_tools`` expect the nested Chat-Completions shape
(``{"type": "function", "function": {"name": ..., ...}}``). Every tool
crossing the boundary is converted; nothing here assumes one shape and
silently mishandles the other.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from meta_evolver.llm.client import build_chat_model, effective_model, message_text
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.core.types import MemoryItem

#: Env var this integration checks before falling back to $META_EVOLVER_MODEL.
STATE_BENCH_MODEL_ENV = "META_EVOLVER_STATE_BENCH_MODEL"

#: Env var pointing at the bank file this integration reads/writes.
STATE_BENCH_MEMORY_PATH_ENV = "META_EVOLVER_STATE_BENCH_MEMORY_PATH"


# ---------------------------------------------------------------------------
# Tool schema / conversation conversion -- no state_bench import needed
# ---------------------------------------------------------------------------


def flat_to_nested_tool_schema(flat: dict[str, Any]) -> dict[str, Any]:
    """STATE-Bench's ``{"type","name","description","parameters"}`` -> the
    nested ``{"type":"function","function":{...}}`` shape this project and
    LangChain's ``bind_tools`` use."""
    return {
        "type": "function",
        "function": {
            "name": flat["name"],
            "description": flat.get("description", ""),
            "parameters": flat.get("parameters") or {"type": "object", "properties": {}},
        },
    }


def nested_to_flat_tool_schema(nested: dict[str, Any]) -> dict[str, Any]:
    """The reverse of :func:`flat_to_nested_tool_schema`."""
    fn = nested.get("function", nested)
    return {
        "type": "function",
        "name": fn["name"],
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def conversation_to_messages(system_prompt: str, conversation: list[dict[str, Any]]) -> list[Any]:
    """STATE-Bench's canonical conversation -> LangChain messages.

    Tool calls are embedded on the assistant message that requested them
    (``docs/USE_CUSTOM_CLIENT.md``'s "Canonical Conversation") rather than
    saved as separate tool messages, so each is expanded into an
    ``AIMessage`` with ``tool_calls`` followed by one ``ToolMessage`` per
    call -- what LangChain / every provider actually expects on the wire.
    """
    messages: list[Any] = [SystemMessage(content=system_prompt)]
    for msg in conversation:
        role = msg.get("role")
        content = msg.get("content") or ""
        calls = msg.get("tool_calls") or []

        if role == "tool":
            # A temporary follow-up item STATE-Bench appends mid tool-loop;
            # its own "content" is the list of just-executed calls.
            for call in content if isinstance(content, list) else calls:
                messages.append(
                    ToolMessage(
                        content=str(call.get("result", "")),
                        tool_call_id=call.get("name", "tool"),
                        name=call.get("name", "tool"),
                    )
                )
            continue

        if role == "assistant":
            tool_calls = [
                {
                    "name": c["name"],
                    "args": c.get("arguments", {}),
                    "id": c["name"],
                    "type": "tool_call",
                }
                for c in calls
            ]
            messages.append(AIMessage(content=content, tool_calls=tool_calls))
            for c in calls:
                messages.append(
                    ToolMessage(
                        content=str(c.get("result", "")), tool_call_id=c["name"], name=c["name"]
                    )
                )
            continue

        messages.append(HumanMessage(content=content))
    return messages


def reply_to_turn_response(reply: AIMessage) -> tuple[str, list[dict[str, Any]]]:
    """One LangChain ``AIMessage`` -> ``(text, tool_call_dicts)`` for
    :class:`AgentTurnResponse`. Kept free of the ``state_bench`` import so it
    is unit-testable without the package installed."""
    text = message_text(reply)
    calls = [
        {"name": c["name"], "arguments": c.get("args") or {}} for c in (reply.tool_calls or [])
    ]
    return text, calls


# ---------------------------------------------------------------------------
# Offline learning extraction -- train trajectories only, never test tasks
# ---------------------------------------------------------------------------


def _trajectory_summary(conversation: list[dict[str, Any]]) -> str:
    """A short, heuristic lesson from one train trajectory's tool usage.

    Deliberately simple: the tool calls actually made, in order, with their
    arguments -- "what a successful run of this kind of task looked like".
    Swap in :class:`~meta_evolver.memory.induction.MemoryInducer` for
    LLM-reflective induction (contrasting successful vs. failed trajectories)
    once there is a labelled outcome to reflect against; the train
    trajectories STATE-Bench ships are not labelled pass/fail, only replayed
    conversations, so there is nothing to contrast here.
    """
    calls = []
    for msg in conversation:
        for c in msg.get("tool_calls") or []:
            calls.append(f"{c['name']}({c.get('arguments', {})})")
    return "; ".join(calls[:12]) or "(no tool calls in this trajectory)"


def build_memory_from_train_trajectories(
    domain: str,
    trajectories_dir: str | Path,
    bank: ReasoningMemoryBank,
) -> int:
    """Populate ``bank`` from ``datasets/train_task_trajectories/<domain>/*.json``.

    Run this once, offline, before a scored evaluation -- never during it.
    Mixing this into :meth:`MetaEvolverStateBenchAgent.ingest_trajectory`
    during the actual benchmark run would let the agent learn from held-out
    test trajectories mid-run, which the official rules explicitly forbid
    ("do not use held-out test task definitions or test environments as
    oracle inputs for learnings") -- this function is the only place that
    touches training data, kept deliberately separate from evaluation.

    Returns the number of memory items added (after dedup).
    """
    import json

    domain_dir = Path(trajectories_dir) / domain
    added = 0
    for path in sorted(domain_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        conversation = record.get("conversation", [])
        lesson = _trajectory_summary(conversation)
        item = MemoryItem(
            title=path.stem,
            scenario=domain,
            lesson=lesson,
            benchmark=f"state-bench-{domain}",
        )
        _, was_new = bank.add(item)
        if was_new:
            added += 1
    return added


# ---------------------------------------------------------------------------
# The two extension points -- lazy state_bench import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised only when state_bench is installed
    from state_bench.agents.base import AgentTurnResponse, BaseAgent
    from state_bench.client import BaseLLMClient

    _STATE_BENCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    BaseAgent = object  # type: ignore[assignment,misc]
    BaseLLMClient = object  # type: ignore[assignment,misc]
    AgentTurnResponse = None  # type: ignore[assignment]
    _STATE_BENCH_AVAILABLE = False


def _require_state_bench() -> None:
    if not _STATE_BENCH_AVAILABLE:
        raise ImportError(
            "the 'state_bench' package is required to use this integration -- "
            "clone https://github.com/microsoft/STATE-Bench and `uv sync` it, "
            "then run from that repo root (or add it to sys.path)."
        )


#: Lazily built, process-wide shared bank -- STATE-Bench constructs a fresh
#: agent instance per task ("no state on self" per its own docs), so
#: cross-task memory has to live above that per-task scope, same as
#: MetaEvolver.bank already lives above individual episodes.
_shared_bank: ReasoningMemoryBank | None = None


def _get_shared_bank() -> ReasoningMemoryBank:
    global _shared_bank
    if _shared_bank is None:
        path = os.environ.get(STATE_BENCH_MEMORY_PATH_ENV)
        _shared_bank = ReasoningMemoryBank.load(path) if path else ReasoningMemoryBank()
    return _shared_bank


class MetaEvolverStateBenchClient(BaseLLMClient):
    """Wraps a meta-evolver ``BaseChatModel`` as a STATE-Bench ``BaseLLMClient``."""

    def __init__(self, model_spec: str) -> None:
        _require_state_bench()
        self.model_spec = model_spec
        self.model = build_chat_model(model_spec)

    @classmethod
    def from_env(cls) -> "MetaEvolverStateBenchClient":
        spec = os.environ.get(STATE_BENCH_MODEL_ENV) or effective_model("")
        if not spec:
            raise RuntimeError(
                f"set ${STATE_BENCH_MODEL_ENV} or $META_EVOLVER_MODEL to a model spec, "
                "e.g. 'google_genai:gemini-3.8-flash' or 'ollama:hermes-4-14b'."
            )
        return cls(spec)

    @property
    def model_name(self) -> str:
        return self.model_spec


class MetaEvolverStateBenchAgent(BaseAgent):
    """Reads/writes nothing during evaluation except retrieval -- see module
    docstring for why learning is an offline, pre-run step instead."""

    def __init__(
        self,
        client: MetaEvolverStateBenchClient,
        system_prompt: str,
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Any],
        runtime_context: Any = None,
        *,
        retrieval_k: int = 3,
        retrieval_mode: str = "mmr",
    ) -> None:
        _require_state_bench()
        super().__init__(runtime_context=runtime_context)
        self.client = client
        self.system_prompt = system_prompt
        self.retrieval_k = retrieval_k
        self.retrieval_mode = retrieval_mode
        self.bank = _get_shared_bank()

    def memory_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "retrieve_memory",
                "description": (
                    "Search past task learnings for strategies relevant to the "
                    "current situation. Read-only."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for."},
                    },
                    "required": ["query"],
                },
            }
        ]

    def memory_tool_handlers(self) -> dict[str, Any]:
        return {"retrieve_memory": self._handle_retrieve_memory}

    def _handle_retrieve_memory(self, query: str) -> dict[str, Any]:
        items = self.bank.retrieve(query, k=self.retrieval_k, mode=self.retrieval_mode)
        return {"memories": [item.render() for item in items]}

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> "AgentTurnResponse":
        messages = conversation_to_messages(system_prompt, conversation)
        nested_tools = [flat_to_nested_tool_schema(t) for t in tools]
        reply = self.client.model.bind_tools(nested_tools).invoke(messages)

        usage = getattr(reply, "usage_metadata", None) or {}
        self.add_token_usage(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

        text, calls = reply_to_turn_response(reply)
        return AgentTurnResponse(
            text=text,
            tool_calls=[{"name": c["name"], "arguments": c["arguments"]} for c in calls],
        )

    def ingest_trajectory(self, trajectory: Any) -> None:
        """No-op by default -- see module docstring: learning happens
        offline from train trajectories only, via
        :func:`build_memory_from_train_trajectories`, never from what the
        agent sees during a scored run."""
        return None


__all__ = [
    "STATE_BENCH_MEMORY_PATH_ENV",
    "STATE_BENCH_MODEL_ENV",
    "MetaEvolverStateBenchAgent",
    "MetaEvolverStateBenchClient",
    "build_memory_from_train_trajectories",
    "conversation_to_messages",
    "flat_to_nested_tool_schema",
    "nested_to_flat_tool_schema",
    "reply_to_turn_response",
]
