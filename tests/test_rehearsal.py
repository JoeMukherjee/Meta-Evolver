"""The rehearsal gate: a prompt challenger that regresses a previously-passed
task must be rejected, even when it beats the incumbent on the current
held-out split.

A deliberately hand-built scenario rather than the ``devops`` benchmark: the
held-out validation split is seeded once per run (``BenchmarkAdapter.sample``
keys it by ``seed`` alone, not by generation), so within one run it never
rotates -- the *only* way a margin check and a rehearsal check can ever
disagree is if rehearsal also draws on train-task history, which is exactly
what needs verifying here. Four tasks with fixed, table-driven outcomes make
that provable rather than incidental.

    Pool: ta, tb (train) | tc, td (held out, verified at seed=0)
    base : ta=F tb=P | tc=F td=F   (val 0/2 -- room for a "better" candidate)
    V1   : ta=F tb=P | tc=P td=F   (val 1/2, beats base -- adopted gen 0)
    V2   : ta=F tb=F | tc=P td=P   (val 2/2, beats V1 on the surface --
                                     but breaks tb, which gen 0 put in the
                                     rehearsal buffer)
"""
from __future__ import annotations

from meta_evolver.benchmarks.custom import FunctionBenchmark, Task
from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.memory.bank import ReasoningMemoryBank

# task -> {prompt_marker: passes?}. Every task not listed for a marker fails.
_OUTCOMES: dict[str, set[str]] = {
    "ta": set(),  # fails under every prompt -- guarantees the optimizer fires
    "tb": {"base", "V1"},
    "tc": {"V1", "V2"},
    "td": {"V2"},
}


def _look(**_: object) -> dict:
    """A harmless non-terminal tool -- calling it and stopping there is how
    a scripted "fail" outcome is produced (no submission, so the episode
    truncates at the step budget without succeeding)."""
    return {"ok": True}


def _answer(text: str) -> dict:
    """Terminal tool. Any call at all means success -- the *responder*
    decides whether to call it, based on the outcome table."""
    return {"answer": text}


def _make_tasks() -> list[Task]:
    # All four are labelled "train": BenchmarkAdapter.sample() draws its
    # held-out split *from* task_ids("train") itself (tc/td end up held out
    # by the seeded sample, not by this label) -- a task's own split field is
    # a separate, coarser concept used by evaluate(split=...) elsewhere.
    return [
        Task(id=task_id, instruction=task_id, split="train", terminal_tools=("answer",))
        for task_id in _OUTCOMES
    ]


def _marker_of(system_text: str) -> str:
    for marker in ("V2", "V1"):  # longer/newer markers checked first
        if marker in system_text:
            return marker
    return "base"


def _task_of(messages) -> str:
    for m in messages:
        content = str(getattr(m, "content", ""))
        if content.startswith("Task: "):
            for task_id in _OUTCOMES:
                if content.split("\n", 1)[0] == f"Task: {task_id}":
                    return task_id
    return ""


def _agent_reply(messages, tools):
    system = str(getattr(messages[0], "content", "")) if messages else ""
    marker = _marker_of(system)
    task_id = _task_of(messages)
    if marker in _OUTCOMES.get(task_id, set()):
        return tool_call_message("answer", text="done")
    return tool_call_message("look")


class _RehearsalModel(ScriptedChatModel):
    """Plays the agent (table-driven), the inducer (nothing to induce) and
    the optimizer (returns V1 in generation 0, V2 in generation 1)."""

    n_optimization: int = 0

    def __init__(self, **kwargs):
        super().__init__(responder=lambda m, t: None, **kwargs)
        self.n_optimization = 0

    def _reply(self, messages):
        system = str(getattr(messages[0], "content", "")) if messages else ""
        if "distil agent execution traces" in system:
            return tool_call_message("__noop__")  # never called in this test
        if "improve the system instructions" in system:
            # PromptOptimizer proposes n_candidates=2 per generation, so two
            # model calls share each generation's version -- batch by pairs
            # rather than by raw call count.
            batch = self.n_optimization // 2
            version = "V1" if batch == 0 else "V2"
            self.n_optimization += 1
            from langchain_core.messages import AIMessage

            # PromptOptimizer.propose() silently discards any candidate text
            # under 80 characters -- padded well past that on purpose.
            return AIMessage(
                content=(
                    f"You are agent {version}, a careful problem-solving agent. "
                    "Read the task, then act. Call the answer tool once you are done."
                )
            )
        return _agent_reply(messages, self.bound_tools)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._reply(list(messages)))])


def _bench() -> FunctionBenchmark:
    return FunctionBenchmark(
        name="rehearsal-mini",
        tools={"answer": _answer, "look": _look},
        tasks=_make_tasks(),
        max_steps=2,
    )


def _run(rehearsal_size: int) -> MetaEvolver:
    evolver = MetaEvolver(
        benchmark=_bench(),
        chat_model=_RehearsalModel(),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(
            generations=2,
            max_steps=2,
            validation_fraction=0.5,  # -> 2 of the 4 tasks held out, per the module docstring
            validate_prompt=True,
            curriculum=False,
            patience=99,
            seed=0,
            rehearsal_size=rehearsal_size,
            rehearsal_tolerance=0.05,
        ),
        telemetry=False,
    )
    evolver.evolve()
    return evolver


def test_the_scenario_s_own_premise_holds():
    """V2 really does look like an improvement on the surface -- otherwise
    this test would prove nothing about rehearsal specifically."""
    without = _run(rehearsal_size=0)
    assert without.prompt_version != "base"
    assert "V2" in without.prompt_template


def test_rehearsal_rejects_the_regressing_candidate():
    with_gate = _run(rehearsal_size=8)
    assert "V2" not in with_gate.prompt_template, (
        "a candidate that breaks a previously-passed task must not be adopted"
    )
    notes = " ".join(n for r in with_gate.reports for n in r.notes)
    assert "rehearsal" in notes


def test_rehearsal_still_lets_a_clean_improvement_through():
    """The gate must reject V2 specifically -- not just refuse every
    adoption. V1 (generation 0) regresses nothing and must still land."""
    with_gate = _run(rehearsal_size=8)
    assert "V1" in with_gate.prompt_template


def test_rehearsal_size_zero_is_bit_identical_to_before_rehearsal_existed():
    """The default. A challenger that would fail rehearsal must still be
    adopted when the gate is off -- proving it really is off, not just lenient."""
    off = _run(rehearsal_size=0)
    assert "V2" in off.prompt_template
    assert not any("rehearsal" in n for r in off.reports for n in r.notes)


def test_config_defaults_to_no_rehearsal():
    assert EvolutionConfig().rehearsal_size == 0
    assert EvolutionConfig().rehearsal_tolerance == 0.05
