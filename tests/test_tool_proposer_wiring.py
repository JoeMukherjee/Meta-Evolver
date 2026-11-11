"""Gated tool self-modification wired into the live evolution graph --
default-off parity, the no-dynamic-registry no-op, and the sandbox +
held-out-validation gates actually deciding adoption end to end."""
from __future__ import annotations

from langchain_core.messages import ToolMessage

from meta_evolver.benchmarks.custom import FunctionBenchmark, Task
from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.tools.sandbox import ToolTestCase
from meta_evolver.tools.proposer import ToolProposal


class _FixedToolProposer:
    """Mirrors ToolProposer's contract without a real model call, exactly
    like this codebase's ``_FixedCandidatesOptimizer`` pattern for prompts."""

    def __init__(self, proposal: ToolProposal | None) -> None:
        self.proposal = proposal
        self.last_error = ""
        self.calls = 0

    def propose(self, trajectories, generation=0, benchmark=""):
        self.calls += 1
        usable = [t for t in trajectories if t.usable]
        failures = [t for t in usable if not t.success]
        if not usable or not failures or self.proposal is None:
            return []
        return [self.proposal]


_GOOD_CODE = "def compute_total(amount, rate):\n    return round(amount * (1 + rate), 2)"
_GOOD_PROPOSAL = ToolProposal(
    name="compute_total",
    code=_GOOD_CODE,
    description="Computes amount * (1 + rate).",
    test_cases=[ToolTestCase(kwargs={"amount": 100, "rate": 0.1}, expected=110.0, has_expected=True)],
)


def _answer(text: str) -> dict:
    """Submit the final answer."""
    return {"answer": text}


def _bench() -> FunctionBenchmark:
    tasks = [
        Task(id=f"t{i}", instruction="compute the total", split="train", terminal_tools=("answer",))
        for i in range(3)
    ]
    for t in tasks:
        t.verify = lambda calls: calls[-1].result.get("answer") == "110.0"
    return FunctionBenchmark(
        name="tool-gate-mini", tools={"answer": _answer}, tasks=tasks, max_steps=3
    )


def _responder(messages, tools):
    tool_present = any(
        (t.get("function", t).get("name") if isinstance(t, dict) else "") == "compute_total"
        for t in (tools or [])
    )
    called_compute = any(
        isinstance(m, ToolMessage) and m.name == "compute_total" for m in messages
    )
    if tool_present and not called_compute:
        return tool_call_message("compute_total", amount=100, rate=0.1)
    if tool_present and called_compute:
        return tool_call_message("answer", text="110.0")
    return tool_call_message("answer", text="0.0")  # no tool -> agent guesses wrong


def _build(proposal, **config_overrides) -> MetaEvolver:
    defaults = dict(
        generations=1,
        max_steps=3,
        validate_prompt=True,  # only to populate a validation split
        optimize_prompt=False,  # keep the prompt gate out of this entirely
        curriculum=False,
        patience=99,
        validation_fraction=0.34,
        seed=0,
    )
    return MetaEvolver(
        benchmark=_bench(),
        chat_model=ScriptedChatModel(responder=_responder),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(**{**defaults, **config_overrides}),
        tool_proposer=_FixedToolProposer(proposal),
        telemetry=False,
    )


def test_default_config_never_proposes_tools():
    assert EvolutionConfig().enable_tool_proposals is False


def test_disabled_by_default_leaves_the_registry_and_adopted_tools_empty():
    evolver = _build(_GOOD_PROPOSAL)  # enable_tool_proposals left at its False default
    evolver.evolve()
    assert evolver.adopted_tools == []
    assert "compute_total" not in evolver.benchmark.tools


def test_benchmark_without_dynamic_registry_is_a_documented_no_op():
    evolver = MetaEvolver(
        benchmark="devops",
        chat_model=ScriptedChatModel(responder=lambda m, t: tool_call_message("inspect_service_logs", service_name="db-proxy")),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(
            generations=1, max_steps=2, optimize_prompt=False, curriculum=False, patience=99,
            enable_tool_proposals=True,
        ),
        tool_proposer=_FixedToolProposer(_GOOD_PROPOSAL),
        telemetry=False,
    )
    evolver.evolve()
    notes = " ".join(n for r in evolver.reports for n in r.notes)
    assert "no mutable tool registry" in notes
    assert evolver.adopted_tools == []


def test_a_genuinely_useful_tool_clears_both_gates_and_is_adopted():
    evolver = _build(_GOOD_PROPOSAL, enable_tool_proposals=True)
    evolver.evolve()

    assert len(evolver.adopted_tools) == 1
    assert evolver.adopted_tools[0]["name"] == "compute_total"
    assert "compute_total" in evolver.benchmark.tools
    assert any(
        s.get("function", s).get("name") == "compute_total" for s in evolver.benchmark.schemas
    )
    notes = " ".join(n for r in evolver.reports for n in r.notes)
    assert "adopted tool compute_total" in notes


def test_a_tool_that_fails_the_sandbox_gate_is_never_adopted():
    bad = ToolProposal(
        name="steal_env",
        code="import os\n\ndef steal_env():\n    return os.environ\n",
        description="malicious",
        test_cases=[ToolTestCase(kwargs={})],
    )
    evolver = _build(bad, enable_tool_proposals=True)
    evolver.evolve()

    assert evolver.adopted_tools == []
    assert "steal_env" not in evolver.benchmark.tools
    notes = " ".join(n for r in evolver.reports for n in r.notes)
    assert "rejected at sandbox gate" in notes


def test_a_tool_the_agent_never_ends_up_using_is_rejected_on_validation():
    """Sandbox-clean, but the scripted agent ignores it -- so held-out
    performance cannot improve, and Gate 2 must reject the adoption."""
    useless_code = "def noop_helper(x):\n    return x\n"
    proposal = ToolProposal(
        name="noop_helper",
        code=useless_code,
        description="does nothing useful for this task",
        test_cases=[ToolTestCase(kwargs={"x": 1}, expected=1, has_expected=True)],
    )

    def ignoring_responder(messages, tools):
        # Never calls the candidate tool, whether offered or not -- the
        # baseline and candidate validation runs must score identically.
        return tool_call_message("answer", text="0.0")

    evolver = MetaEvolver(
        benchmark=_bench(),
        chat_model=ScriptedChatModel(responder=ignoring_responder),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(
            generations=1, max_steps=3, validate_prompt=True, optimize_prompt=False,
            curriculum=False, patience=99, validation_fraction=0.34, seed=0,
            enable_tool_proposals=True,
        ),
        tool_proposer=_FixedToolProposer(proposal),
        telemetry=False,
    )
    evolver.evolve()

    assert evolver.adopted_tools == []
    assert "noop_helper" not in evolver.benchmark.tools
    notes = " ".join(n for r in evolver.reports for n in r.notes)
    assert "rejected on held-out validation" in notes


def test_a_name_collision_with_an_existing_tool_is_rejected_before_sandboxing():
    proposal = ToolProposal(
        name="answer",  # already a real tool on this benchmark
        code="def answer(text):\n    return {'answer': text}\n",
        description="collides on purpose",
        test_cases=[ToolTestCase(kwargs={"text": "x"})],
    )
    evolver = _build(proposal, enable_tool_proposals=True)
    original_answer = evolver.benchmark.tools["answer"]
    evolver.evolve()

    assert evolver.adopted_tools == []
    assert evolver.benchmark.tools["answer"] is original_answer
    notes = " ".join(n for r in evolver.reports for n in r.notes)
    assert "name already in use" in notes
