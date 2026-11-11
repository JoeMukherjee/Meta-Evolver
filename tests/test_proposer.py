"""ToolProposer: proposes a candidate tool from failure traces, parses its
model's reply defensively, and fails closed on anything malformed or
declined -- offline via ScriptedChatModel."""
from __future__ import annotations

from meta_evolver.core.types import Action, StepRecord, Trajectory
from meta_evolver.llm.client import ScriptedChatModel
from meta_evolver.tools.proposer import ToolProposer

_GOOD_REPLY = """\
Here is my proposal:
```json
{
  "name": "add_tax",
  "description": "Computes a price including tax.",
  "code": "def add_tax(amount, rate):\\n    return round(amount * (1 + rate), 2)",
  "tests": [
    {"kwargs": {"amount": 100, "rate": 0.1}, "expected": 110.0},
    {"kwargs": {"amount": 0, "rate": 0.5}, "assert_code": "RESULT == 0.0"}
  ]
}
```"""


def _traj(task_id: str, success: bool) -> Trajectory:
    return Trajectory(
        task_id=task_id,
        benchmark="devops",
        success=success,
        score=1.0 if success else 0.0,
        steps=[StepRecord(step_idx=1, action=Action(name="look"), observation="obs")],
    )


def test_no_usable_trajectories_returns_nothing():
    proposer = ToolProposer(ScriptedChatModel(script=[]))
    assert proposer.propose([]) == []


def test_all_successes_returns_nothing_unlearned_from():
    proposer = ToolProposer(ScriptedChatModel(script=[]))
    assert proposer.propose([_traj("a", True), _traj("b", True)]) == []


def test_a_well_formed_reply_parses_into_a_proposal_with_test_cases():
    model = ScriptedChatModel(responder=lambda m, t: _ai(_GOOD_REPLY))
    proposer = ToolProposer(model)
    proposals = proposer.propose([_traj("a", False)], generation=2)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.name == "add_tax"
    assert "def add_tax" in p.code
    assert len(p.test_cases) == 2
    assert p.test_cases[0].has_expected and p.test_cases[0].expected == 110.0
    assert p.test_cases[1].assert_code == "RESULT == 0.0"
    assert p.version == "g2t0"


def test_a_model_declining_to_propose_returns_nothing():
    declined = '```json\n{"name": "", "description": "", "code": "", "tests": []}\n```'
    model = ScriptedChatModel(responder=lambda m, t: _ai(declined))
    proposer = ToolProposer(model)
    assert proposer.propose([_traj("a", False)]) == []


def test_malformed_json_fails_closed_not_raises():
    model = ScriptedChatModel(responder=lambda m, t: _ai("not json at all, sorry"))
    proposer = ToolProposer(model)
    assert proposer.propose([_traj("a", False)]) == []


def test_untested_proposal_is_rejected():
    no_tests = '```json\n{"name": "f", "description": "d", "code": "def f():\\n    return 1", "tests": []}\n```'
    model = ScriptedChatModel(responder=lambda m, t: _ai(no_tests))
    proposer = ToolProposer(model)
    assert proposer.propose([_traj("a", False)]) == []


def test_llm_error_sets_last_error_and_returns_nothing():
    class Failing(ScriptedChatModel):
        def invoke(self, messages, **kwargs):
            raise RuntimeError("provider down")

        def bind_tools(self, tools, **kwargs):
            return self

    proposer = ToolProposer(Failing())
    assert proposer.propose([_traj("a", False)]) == []
    assert "provider down" in proposer.last_error


def _ai(content: str):
    from langchain_core.messages import AIMessage

    return AIMessage(content=content)
