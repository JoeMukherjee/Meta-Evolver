"""ReplayScreener wired into the real evolution graph.

tests/test_replay.py covers the class in isolation; this covers that
build_evolution_graph()/_select_prompt() actually call it at the right
points, with a fixed 2-candidate optimizer for full control over what
gets proposed each generation, and a 3-task benchmark (BenchmarkAdapter.
sample() only creates a held-out split once the train pool has >=3 tasks --
see tests/test_adapter.py for the same constraint) where one candidate
text is designed to pass validation and the other to fail it, every
generation, so there is a real, discoverable pattern for screening to learn.
"""
from __future__ import annotations

from meta_evolver.benchmarks.custom import FunctionBenchmark, Task
from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.prompts.optimizer import PromptCandidate

_STRONG = "STRONG-MARKER: always confirm the owning service before submitting."
_WEAK = "WEAK-MARKER: submit immediately without checking."


class _FixedCandidateOptimizer:
    """Always proposes the same two candidates -- deterministic control over
    what screening has to discriminate between, independent of whether
    PromptOptimizer's own real failure-gating would fire this generation."""

    def __init__(self) -> None:
        self.last_error = ""
        self.calls = 0

    def propose(self, current_prompt, trajectories, generation=0, benchmark=""):
        self.calls += 1
        return [
            PromptCandidate(text=_STRONG, version=f"g{generation}strong", parent="base"),
            PromptCandidate(text=_WEAK, version=f"g{generation}weak", parent="base"),
        ]


def _benchmark() -> FunctionBenchmark:
    def answer(text: str) -> dict:
        """Submit the final answer."""
        return {"answer": text}

    def look(**_: object) -> dict:
        """A harmless non-terminal tool."""
        return {"ok": True}

    # tb is the task BenchmarkAdapter.sample() holds out as validation for
    # this exact 3-task pool at seed=0 (verified in test_adapter.py/
    # test_rehearsal.py already); ta/tc are training.
    def verify_tb(calls: object) -> bool:
        return any(c.name == "answer" for c in calls)

    tasks = [
        Task(id="ta", instruction="ta", split="train", terminal_tools=("answer",)),
        Task(id="tb", instruction="tb", split="train", terminal_tools=("answer",), verify=verify_tb),
        Task(id="tc", instruction="tc", split="train", terminal_tools=("answer",)),
    ]
    return FunctionBenchmark(
        name="replay-mini", tools={"answer": answer, "look": look}, tasks=tasks, max_steps=2
    )


def _responder(messages, tools):
    system = str(getattr(messages[0], "content", "")) if messages else ""
    task_line = next(
        (
            str(getattr(m, "content", "")).split("\n", 1)[0]
            for m in messages
            if str(getattr(m, "content", "")).startswith("Task: ")
        ),
        "",
    )
    if task_line == "Task: tb":
        # STRONG passes tb (and so the held-out margin); WEAK never does.
        if "STRONG-MARKER" in system:
            return tool_call_message("answer", text="done")
        return tool_call_message("look")
    return tool_call_message("look")  # ta/tc: irrelevant to this test, always "fails"


def _run(generations: int, replay_screen: bool, replay_top_k: int | None):
    model = ScriptedChatModel(responder=_responder)
    optimizer = _FixedCandidateOptimizer()
    evolver = MetaEvolver(
        benchmark=_benchmark(),
        chat_model=model,
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(
            generations=generations,
            max_steps=2,
            validation_fraction=0.34,  # -> holdout ["tb"] on this 3-task pool
            validate_prompt=True,
            optimize_prompt=True,
            curriculum=False,
            patience=99,
            seed=0,
            replay_screen=replay_screen,
            replay_top_k=replay_top_k,
        ),
        prompt_optimizer=optimizer,
        telemetry=False,
    )
    reports = evolver.evolve()
    return evolver, reports


def test_replay_screen_off_by_default_never_mentions_screening():
    _, reports = _run(generations=2, replay_screen=False, replay_top_k=1)
    notes = " ".join(n for r in reports for n in r.notes)
    assert "replay" not in notes.lower()


def test_replay_screener_is_only_constructed_when_enabled():
    evolver_off, _ = _run(generations=1, replay_screen=False, replay_top_k=1)
    assert evolver_off._replay_screener is None

    evolver_on, _ = _run(generations=1, replay_screen=True, replay_top_k=1)
    assert evolver_on._replay_screener is not None


def test_first_generation_has_no_history_so_nothing_is_screened_out():
    """min_history defaults to 3; a fresh screener has zero entries, so
    generation 0 must validate every candidate regardless of top_k."""
    evolver, reports = _run(generations=1, replay_screen=True, replay_top_k=1)
    assert "replay-screened" not in " ".join(reports[0].notes)
    # Both candidates still got a real score recorded (the incumbent's
    # baseline plus 2 candidates == 3 history entries).
    assert len(evolver._replay_screener.history) == 3


def test_screening_trims_candidates_once_there_is_history():
    evolver, reports = _run(generations=2, replay_screen=True, replay_top_k=1)
    gen1_notes = " ".join(reports[1].notes)
    assert "replay-screened 2 candidates to 1" in gen1_notes
    assert evolver._replay_screener.n_screened_out >= 1


def test_agreement_is_reported_once_there_are_two_or_more_real_scores():
    """top_k=None: screening tracks and predicts but never trims, so both
    candidates keep getting validated for real every generation -- a
    legitimate "dry run" mode, and the only way two candidates are both
    still around in the same generation to compute a pairwise agreement
    over once top_k=1 (test_screening_trims_...) would leave just one."""
    _, reports = _run(generations=2, replay_screen=True, replay_top_k=None)
    gen1_notes = " ".join(reports[1].notes)
    assert "replay agreement" in gen1_notes
    assert "replay-screened" not in gen1_notes  # nothing to trim with no top_k


def test_the_strong_candidate_still_wins_real_validation_regardless_of_screening():
    """Screening only decides who gets *measured* -- the actual winner is
    still whichever real score beats the margin, same as with screening off."""
    evolver, _ = _run(generations=2, replay_screen=True, replay_top_k=1)
    assert "STRONG-MARKER" in evolver.prompt_template
