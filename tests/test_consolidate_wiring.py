"""consolidate_every wired into the live prune node -- fires on the right
generations, not before, and only when turned on."""
from __future__ import annotations

from unittest.mock import patch

from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.memory.bank import ReasoningMemoryBank


def _weak_agent_reply(messages, tools):
    return tool_call_message("inspect_service_logs", service_name="db-proxy")


def _build(**config_overrides) -> MetaEvolver:
    defaults = dict(
        generations=4,
        max_steps=2,
        induce_memories=False,  # isolate consolidation timing from induction
        optimize_prompt=False,
        validate_prompt=False,
        curriculum=False,
        patience=99,
    )
    return MetaEvolver(
        benchmark="devops",
        chat_model=ScriptedChatModel(responder=_weak_agent_reply),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(**{**defaults, **config_overrides}),
        telemetry=False,
    )


def test_consolidate_every_zero_never_calls_consolidate():
    evolver = _build(consolidate_every=0)
    with patch("meta_evolver.graphs.evolution.consolidate") as spy:
        evolver.evolve()
    spy.assert_not_called()


def test_consolidate_every_two_fires_on_generations_two_and_four():
    """0-indexed generations 0,1,2,3 -- (generation+1) % 2 == 0 on 1 and 3,
    i.e. after the completed 2nd and 4th generations."""
    evolver = _build(consolidate_every=2)
    calls = []
    real_consolidate = __import__(
        "meta_evolver.memory.consolidate", fromlist=["consolidate"]
    ).consolidate

    def spy(bank, **kwargs):
        calls.append(bank)
        return real_consolidate(bank, **kwargs)

    with patch("meta_evolver.graphs.evolution.consolidate", side_effect=spy):
        evolver.evolve()

    assert len(calls) == 2


def test_consolidate_every_one_fires_on_every_generation():
    evolver = _build(consolidate_every=1)
    with patch("meta_evolver.graphs.evolution.consolidate", wraps=lambda bank, **kw: []) as spy:
        evolver.evolve()
    assert spy.call_count == 4


def test_consolidate_similarity_is_forwarded():
    evolver = _build(consolidate_every=1, consolidate_similarity=0.42)
    with patch("meta_evolver.graphs.evolution.consolidate", wraps=lambda bank, **kw: []) as spy:
        evolver.evolve()
    for call in spy.call_args_list:
        assert call.kwargs["similarity_threshold"] == 0.42


def test_default_config_never_consolidates():
    assert EvolutionConfig().consolidate_every == 0


def test_consolidation_reduces_the_bank_when_duplicates_exist():
    """A real end-to-end check, not a spy: two near-duplicate items added
    directly to the bank before a consolidating generation runs must come
    out as one."""
    from meta_evolver.core.types import MemoryItem

    evolver = _build(generations=1, consolidate_every=1, consolidate_similarity=0.5)
    a = MemoryItem(id="a", scenario="general", title="t", lesson="restart the pool")
    b = MemoryItem(id="b", scenario="general", title="t", lesson="restart the pool")
    evolver.bank.add(a)
    # add()'s own dedup (0.93 default) would already merge two identical
    # strings -- force them in as distinct rows the way two independently
    # induced memories with slightly different embeddings would arrive.
    evolver.bank.items.append(b)
    assert len(evolver.bank) == 2

    evolver.evolve()

    assert len(evolver.bank) == 1
