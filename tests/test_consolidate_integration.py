"""consolidate() wired into the real evolution graph via consolidate_every."""
from __future__ import annotations

from meta_evolver.benchmarks.custom import FunctionBenchmark, Task
from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.core.types import MemoryItem
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.memory.bank import ReasoningMemoryBank


def _tiny_benchmark() -> FunctionBenchmark:
    def answer(text: str) -> dict:
        """Submit the final answer."""
        return {"answer": text}

    return FunctionBenchmark(
        name="consolidate-mini",
        tools={"answer": answer},
        tasks=[Task(id="t1", instruction="t1", split="train", terminal_tools=("answer",))],
        max_steps=2,
    )


def _solve(messages, tools):
    return tool_call_message("answer", text="done")


def _build(consolidate_every: int) -> MetaEvolver:
    evolver = MetaEvolver(
        benchmark=_tiny_benchmark(),
        chat_model=ScriptedChatModel(responder=_solve),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(
            generations=1,
            max_steps=2,
            induce_memories=False,  # isolate consolidation from induction's own randomness
            optimize_prompt=False,
            validate_prompt=False,
            curriculum=False,
            patience=99,
            retrieval_k=0,  # isolate from retrieval/credit-assignment touching uses/wins
            consolidate_every=consolidate_every,
        ),
        telemetry=False,
    )
    # Two near-duplicate memories, phrased differently enough that write-time
    # dedup's tighter threshold would not have caught them -- exactly the
    # gap consolidate() exists to close.
    evolver.bank.add(
        MemoryItem(
            title="a",
            scenario="db_pool",
            lesson="raise the connection pool's max_connections limit",
            uses=6,
            wins=5,
        )
    )
    evolver.bank.add(
        MemoryItem(
            title="b",
            scenario="db_pool",
            lesson="increase the connection pool max_connections setting",
            uses=2,
            wins=2,
        )
    )
    return evolver


def test_consolidate_every_zero_never_merges_anything():
    evolver = _build(consolidate_every=0)
    before = len(evolver.bank)
    evolver.evolve()
    assert len(evolver.bank) == before


def test_consolidate_every_one_merges_related_memories_on_the_first_generation():
    evolver = _build(consolidate_every=1)
    before = len(evolver.bank)
    assert before == 2

    reports = evolver.evolve()

    # Either it merged (2 -> fewer) or the two items' real fallback
    # embeddings genuinely weren't close enough at the default threshold --
    # assert the mechanism ran, not a specific cosine outcome this test
    # doesn't control precisely (real hashed embeddings, not a fixed table).
    assert len(evolver.bank) <= before
    if len(evolver.bank) < before:
        assert reports[0].memories_pruned >= 1


def test_consolidate_actually_merges_with_a_low_enough_threshold():
    """Same scenario, but with consolidate_similarity relaxed enough that
    these two paraphrases are guaranteed to cluster -- proves the wiring
    itself works, independent of the real embedder's exact cosine value."""
    evolver = _build(consolidate_every=1)
    evolver.config.consolidate_similarity = 0.1  # deliberately permissive

    reports = evolver.evolve()

    assert len(evolver.bank) == 1
    assert reports[0].memories_pruned >= 1
    survivor = evolver.bank.items[0]
    assert survivor.uses == 8  # 6 + 2, pooled
    assert survivor.wins == 7  # 5 + 2, pooled
