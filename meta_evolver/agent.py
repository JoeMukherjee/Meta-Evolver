"""Agent -- an Agno-shaped facade over MetaEvolver.

Agno's ergonomics are a single flat, keyword-only constructor (``model=``,
``tools=``, ``instructions=``, ...) plus a couple of verbs (``.run()``,
``.print_response()``): assemble components, get an object you talk to.
Meta-Evolver's actual differentiator -- GEPA-style prompt evolution, the
Beta(1,1) memory bank, curriculum-gated difficulty, causal-graph recording --
lives in :class:`~meta_evolver.core.evolver.MetaEvolver`, built for a
research harness's shape, not a one-liner's. This module is the seam between
the two: the same constructor feel, wired straight onto the real engine --
no shadow implementation, no re-derived state.

Every method here is a thin call into ``MetaEvolver``/``ChatBenchmark``. If a
capability is missing, it belongs in the engine, not duplicated here.

Any Hermes-compatible endpoint already works as ``model``, unchanged:
:func:`~meta_evolver.llm.client.build_chat_model` resolves ``ollama:``,
``openai:`` (with any ``base_url`` -- including a vLLM/OpenAI-compatible
server serving a Hermes checkpoint), and every other LangChain provider
through ``init_chat_model()``. Tool calls always take the OpenAI
function-calling JSON shape Hermes models are trained on
(:func:`~meta_evolver.benchmarks.base.tool_schema`) -- nothing here
re-implements a wire format.

::

    from meta_evolver.agent import Agent

    agent = Agent(
        model="ollama:hermes-4-14b",
        tools=[web_search, calculator],
        instructions="You are a terse, technical assistant.",
    )
    agent.chat()                      # live terminal conversation
    print(agent.run("What is 2+2?"))  # one-shot ask, Agno's agent.run()
    agent.attach("devops").evolve(generations=5)  # the RSI loop, same object

Two things this is deliberately *not*: a reimplementation of Agno's Team or
Workflow layers (those compose several agents; nothing here changes when one
does) and a replacement for direct :class:`MetaEvolver` use in research code
that wants ``EvolutionConfig``, custom curricula, or a Postgres-backed bank --
:attr:`Agent.evolver` is the live object underneath for exactly that case.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel

from meta_evolver.adapter import MetaAdapter, SeededOptimizer, discount_usage
from meta_evolver.benchmarks.base import BenchmarkAdapter
from meta_evolver.benchmarks.chat import ChatBenchmark, ChatTurnSource, ScriptedTurnSource
from meta_evolver.benchmarks.custom import FunctionBenchmark, Task
from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.core.registry import get_benchmark
from meta_evolver.core.types import GenerationReport, Trajectory
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import DEFAULT_MODEL
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.prompts.optimizer import PromptOptimizer


def _tools_as_dict(tools: Sequence[Callable[..., Any]] | dict[str, Callable[..., Any]] | None) -> dict[str, Callable[..., Any]]:
    """Accept Agno's list-of-callables style or a name-keyed dict, either way."""
    if not tools:
        return {}
    if isinstance(tools, dict):
        return dict(tools)
    return {getattr(fn, "__name__", f"tool_{i}"): fn for i, fn in enumerate(tools)}


class Agent:
    """A single agent: a model, its tools, and everything it has learned.

    ``memory=True`` (the default) gives it the Beta(1,1)-scored reasoning
    bank; ``evolve_prompt=True`` opts its own prompt into GEPA-style
    proposal-and-validation when :meth:`evolve` runs. Both are on by default
    because an agent that does not accumulate anything across runs is not
    the point of this project -- pass ``memory=False`` for the honest
    ablation, same as :meth:`MetaEvolver.evaluate`'s ``use_memory``.
    """

    def __init__(
        self,
        model: str | BaseChatModel = DEFAULT_MODEL,
        *,
        name: str = "agent",
        tools: Sequence[Callable[..., Any]] | dict[str, Callable[..., Any]] | None = None,
        instructions: str | None = None,
        memory: bool = True,
        memory_path: str | Path | None = None,
        evolve_prompt: bool = True,
        max_steps: int = 200,
        embed_model: str | None = None,
        requests_per_second: float | None = None,
    ) -> None:
        self.name = name
        self.tools = _tools_as_dict(tools)
        self.instructions = instructions
        self.max_steps = int(max_steps)

        chat_model = model if isinstance(model, BaseChatModel) else None
        model_spec = model if isinstance(model, str) else DEFAULT_MODEL

        self._chat_benchmark = ChatBenchmark(
            name=name, tools=self.tools, system_prompt=instructions
        )
        self.evolver = MetaEvolver(
            benchmark=self._chat_benchmark,
            model=model_spec,
            chat_model=chat_model,
            memory_path=memory_path,
            embed_model=embed_model,
            requests_per_second=requests_per_second,
            config=EvolutionConfig(optimize_prompt=evolve_prompt, max_steps=max_steps),
            use_memory=memory,
            telemetry=False,
        )

    # -- live conversation ---------------------------------------------------

    def chat(
        self,
        turn_source: ChatTurnSource | None = None,
        on_reply: Callable[[str], None] | None = None,
    ) -> Trajectory:
        """One live, open-ended conversation. Blocks on the terminal by
        default -- see :meth:`MetaEvolver.chat` for a custom frontend."""
        return self.evolver.chat(
            turn_source=turn_source, on_reply=on_reply, max_steps=self.max_steps
        )

    def run(self, message: str) -> str:
        """Ask one thing, get one reply -- Agno's ``agent.run(message)``.

        Implemented as a one-turn chat session: the agent may call tools
        freely, then its final ``send_message`` is the return value. This is
        the same :class:`~meta_evolver.benchmarks.chat.LiveChatEnv` used by
        :meth:`chat`, not a separate code path -- a one-shot ask and a live
        conversation are the same kind of episode, one turn long.
        """
        replies: list[str] = []
        self.evolver.chat(
            turn_source=ScriptedTurnSource([message]),
            on_reply=replies.append,
            max_steps=self.max_steps,
            session_id=f"run-{uuid4().hex[:8]}",
        )
        return replies[-1] if replies else ""

    # -- attach to a benchmark and evolve -------------------------------------

    def attach(self, benchmark: str | BenchmarkAdapter) -> Agent:
        """Point this agent at a benchmark instead of live chat.

        Swaps ``self.evolver.benchmark`` in place, so the same model, bank
        and evolved prompt carry over -- this is the mechanism a detachable
        learning artifact (bank + prompt) would reattach through, whether
        the new target is a benchmark or another chat deployment.
        """
        self.evolver.benchmark = get_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
        return self

    def evolve(self, generations: int | None = None) -> list[GenerationReport]:
        """Run the outer RSI loop against whatever :meth:`attach` set."""
        return self.evolver.evolve(generations=generations)

    def evaluate(self, split: str = "eval") -> dict[str, Any]:
        return self.evolver.evaluate(split=split)

    # -- persistence: the detachable learning artifact ------------------------

    def save(self, path: str | Path) -> dict[str, Path]:
        """Persist the bank, the evolved prompt, and a manifest -- what makes
        this agent this agent, separable from the model and the tools it was
        run with, and attachable to another instance via
        :meth:`attach_adapter`. See :mod:`meta_evolver.adapter`."""
        return MetaAdapter.from_evolver(self.evolver).save(self.evolver, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> Agent:
        """Rebuild an agent from a previous :meth:`save`.

        A directory saved before ``manifest.json`` existed loads as a legacy
        artifact: the bank and prompt still round-trip (they were always
        plain files), just with no provenance or rehearsal oracle attached.
        """
        path = Path(path)
        adapter = MetaAdapter.load(path)
        agent = cls(memory_path=path / "memory.jsonl", **kwargs)
        if adapter.prompt_text:
            agent.evolver.prompt_template = adapter.prompt_text
            agent.evolver.prompt_version = adapter.prompt_version
        agent.evolver.curriculum_level = adapter.curriculum_level
        agent.evolver.rehearsal_ids = list(adapter.rehearsal_ids)
        return agent

    def attach_adapter(self, path: str | Path, *, discount: float = 0.5) -> Agent:
        """Merge another agent's learned bank into this one, and offer its
        evolved prompt as a validated candidate for the *next* :meth:`evolve`
        call -- never applied directly. See :mod:`meta_evolver.adapter` for
        why: the prompt goes through the same held-out margin and rehearsal
        gate every other candidate does, via :class:`SeededOptimizer`.

        ``discount`` scales the imported items' ``wins``/``uses`` before they
        enter this bank (``bank.extend`` still dedupes against what is
        already here) -- imported evidence has not been earned in *this*
        bank yet. ``1.0`` imports at full strength, ``0.0`` imports the
        text only, as brand-new unused items.

        The prompt seed only takes effect on a subsequent :meth:`evolve`
        call against a real benchmark (``evolve_prompt=True``, the default).
        It has no effect on :meth:`chat`/:meth:`run`: ``ChatBenchmark`` has
        exactly one task, so the train/held-out split prompt evolution
        depends on does not yet have a coherent meaning over live chat --
        a real, open gap, not something this method silently papers over.
        """
        path = Path(path)
        adapter = MetaAdapter.load(path)
        items = discount_usage(adapter.memory_items(path), discount)
        self.bank.extend(items, generation=0)

        current = self.evolver._prompt_optimizer
        if not isinstance(current, SeededOptimizer):
            current = SeededOptimizer(current or PromptOptimizer(self.evolver.model))
            self.evolver._prompt_optimizer = current
        if adapter.prompt_text:
            current.seed(adapter.prompt_text, version=f"adapter:{adapter.prompt_version}")

        return self

    @property
    def bank(self) -> ReasoningMemoryBank:
        return self.evolver.bank

    def close(self) -> None:
        self.evolver.close()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["Agent"]
