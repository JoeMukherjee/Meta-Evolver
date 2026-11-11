"""retrieval_min_similarity wired into the real episode graph's prepare()
node -- makes retrieval_k query-adaptive: a candidate below the floor is
dropped rather than padding out the top-k.

Uses a marker-keyed fixed embedder (deterministic cosine, no real encoder
needed) rather than the real hashed fallback, so the similarity gap between
"clearly relevant" and "clearly not" is exact rather than approximate.
"""
from __future__ import annotations

from meta_evolver.benchmarks.custom import FunctionBenchmark, Task
from meta_evolver.core.types import MemoryItem
from meta_evolver.graphs.episode import build_episode_graph, run_episode
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.llm.embeddings import Embedder
from meta_evolver.memory.bank import ReasoningMemoryBank

_NEAR = "NEARMARKER"


class _MarkerEmbedder(Embedder):
    """[1, 0] for any text containing ``_NEAR``, [0, 1] otherwise -- cosine
    similarity to a query built from ``_NEAR`` is exactly 1.0 or 0.0."""

    def __init__(self) -> None:
        super().__init__(embeddings=None, dim=2)

    def embed_documents(self, texts):
        return [[1.0, 0.0] if _NEAR in t else [0.0, 1.0] for t in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if _NEAR in text else [0.0, 1.0]


def _bank() -> ReasoningMemoryBank:
    near = MemoryItem(id="near", title="near", scenario="s", lesson=f"{_NEAR} strategy")
    far = MemoryItem(id="far", title="far", scenario="s", lesson="unrelated strategy")
    return ReasoningMemoryBank(items=[near, far], embedder=_MarkerEmbedder())


def _benchmark() -> FunctionBenchmark:
    def answer(text: str) -> dict:
        """Submit the final answer."""
        return {"answer": text}

    return FunctionBenchmark(
        name="retrieval-mini",
        tools={"answer": answer},
        tasks=[
            Task(id="t1", instruction=f"{_NEAR} task", split="train", terminal_tools=("answer",))
        ],
        max_steps=2,
    )


def _solve(min_similarity: float):
    model = ScriptedChatModel(responder=lambda m, t: tool_call_message("answer", text="done"))
    graph = build_episode_graph(
        model=model,
        bank=_bank(),
        retrieval_k=5,
        retrieval_min_similarity=min_similarity,
    )
    benchmark = _benchmark()
    env = benchmark.make_env("t1")
    return run_episode(
        graph,
        env=env,
        task_id="t1",
        benchmark=benchmark.name,
        instruction=benchmark.instruction_for("t1"),
        prompt_template=benchmark.system_prompt(),
        max_steps=2,
    )


def test_default_floor_returns_every_candidate_up_to_k():
    trajectory = _solve(min_similarity=0.0)
    assert sorted(trajectory.retrieved_memory_ids) == ["far", "near"]


def test_high_floor_drops_the_dissimilar_candidate():
    trajectory = _solve(min_similarity=0.5)
    assert trajectory.retrieved_memory_ids == ["near"]
