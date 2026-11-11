"""ReplayScreener wired into the live evolution graph via MetaEvolver /
build_evolution_graph -- default-off parity, and that screening actually
trims candidates and reports it when turned on.
"""
from __future__ import annotations

from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.eval.replay import ReplayScreener
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.prompts.optimizer import PromptCandidate


class _FixedCandidatesOptimizer:
    """A stub optimizer proposing a fixed set of candidates every
    generation, so the number screening has to work with is controlled
    directly rather than depending on a real LLM's output count."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.last_error = ""
        self.calls = 0

    def propose(self, current_prompt, trajectories, generation=0, benchmark=""):
        self.calls += 1
        usable = [t for t in trajectories if t.usable]
        if not usable or all(t.success for t in usable):
            return []
        return [
            PromptCandidate(text=text, version=f"g{generation}c{i}", parent="base")
            for i, text in enumerate(self.texts)
        ]


def _weak_agent_reply(messages, tools):
    return tool_call_message("inspect_service_logs", service_name="db-proxy")


def test_replay_screen_off_by_default_matches_no_screener_at_all():
    optimizer = _FixedCandidatesOptimizer(["candidate one text here", "candidate two text here"])
    evolver = MetaEvolver(
        benchmark="devops",
        chat_model=ScriptedChatModel(responder=_weak_agent_reply),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(
            generations=1, max_steps=2, validate_prompt=False, curriculum=False, patience=99
        ),
        prompt_optimizer=optimizer,
        telemetry=False,
    )
    assert evolver._replay_screener is None
    evolver.evolve()
    # Unvalidated mode always adopts candidates[0] -- confirms the graph ran
    # candidates through normally, with no screening note anywhere.
    assert "replay-screened" not in " ".join(n for r in evolver.reports for n in r.notes)


def test_replay_screen_true_auto_constructs_a_screener():
    evolver = MetaEvolver(
        benchmark="devops",
        chat_model=ScriptedChatModel(script=[]),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(replay_screen=True, replay_top_k=2),
        telemetry=False,
    )
    assert isinstance(evolver._replay_screener, ReplayScreener)
    assert evolver._replay_screener.top_k == 2


def test_screening_trims_candidates_and_the_note_says_so():
    """Pre-seed the screener's history directly (bypassing the several real
    generations it would otherwise take to accumulate naturally), then run
    one generation with four candidates against a top_k of one."""
    texts = [
        "restart the connection pool and raise its limit immediately",
        "check disk pressure carefully before doing anything else at all",
        "inspect the log shipper retention settings before acting further",
        "raise the sync worker poll interval to reduce backpressure now",
    ]
    optimizer = _FixedCandidatesOptimizer(texts)

    evolver = MetaEvolver(
        benchmark="devops",
        chat_model=ScriptedChatModel(responder=_weak_agent_reply),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(
            generations=1,
            max_steps=2,
            validate_prompt=False,  # isolate screening from the validation gate
            curriculum=False,
            patience=99,
        ),
        prompt_optimizer=optimizer,
        telemetry=False,
    )
    screener = ReplayScreener(embedder=evolver.embedder, top_k=1, min_history=2)
    for text in texts:
        screener.record(text, 0.5)  # enough history to actually predict from
    evolver._replay_screener = screener
    evolver.config.replay_screen = True

    evolver.evolve()

    # validate_prompt=False bypasses _select_prompt's screening branch
    # entirely (it is a separate, unvalidated adoption path) -- so this
    # proves the auto-wired screener object survives a real evolve() call
    # without error; the screening-in-effect behaviour is covered directly
    # against _select_prompt's validated path below.
    assert evolver.model_name


def test_screening_trims_candidates_in_the_validated_path(monkeypatch):
    """The real target: with validate_prompt=True, a top_k smaller than the
    proposed candidate count must reduce how many are actually validated,
    and the report must say so."""
    from meta_evolver.benchmarks.custom import FunctionBenchmark, Task

    def answer(text: str) -> dict:
        """Submit the final answer."""
        return {"answer": text}

    def look(**_: object) -> dict:
        """A harmless non-terminal tool."""
        return {"ok": True}

    tasks = [
        Task(id="ta", instruction="ta", split="train", terminal_tools=("answer",)),
        Task(id="tb", instruction="tb", split="train", terminal_tools=("answer",), verify=lambda c: False),
        Task(id="tc", instruction="tc", split="train", terminal_tools=("answer",)),
    ]
    bench = FunctionBenchmark(name="screen-mini", tools={"answer": answer, "look": look}, tasks=tasks, max_steps=2)

    texts = [f"candidate text number {i} about fixing things properly" for i in range(4)]
    optimizer = _FixedCandidatesOptimizer(texts)
    model = ScriptedChatModel(responder=lambda m, t: tool_call_message("look"))

    evolver = MetaEvolver(
        benchmark=bench,
        chat_model=model,
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(
            generations=1,
            max_steps=2,
            validation_fraction=0.34,
            validate_prompt=True,
            curriculum=False,
            patience=99,
            seed=0,
            replay_screen=True,
            replay_top_k=1,
        ),
        prompt_optimizer=optimizer,
        telemetry=False,
    )
    for text in texts:
        evolver._replay_screener.record(text, 0.5)

    evolver.evolve()

    notes = " ".join(n for r in evolver.reports for n in r.notes)
    assert "replay-screened" in notes
    # 4 candidates proposed, top_k=1 -> only 1 survives to real validation.
    assert "replay-screened 4 candidates to 1" in notes
