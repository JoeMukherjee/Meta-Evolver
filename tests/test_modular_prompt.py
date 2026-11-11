"""Injecting GEPAPromptOptimizer into the live evolution graph.

build_evolution_graph() defaults to PromptOptimizer -- today's monolithic
rewriter -- exactly as before; these tests cover the opt-in path: a
GEPAPromptOptimizer passed in behaves as a real drop-in (last_error exists,
the graph runs to completion, the rehearsal gate still applies to it), and
specifically that it freezes core_role while only ever mutating
planning/tool_policy/error_recovery.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message
from meta_evolver.memory.bank import ReasoningMemoryBank
from meta_evolver.prompts.gepa import GEPAPromptOptimizer, ModularPrompt

_DEFAULT_CORE_ROLE = ModularPrompt().core_role
_MARKER = "MUTATED-COMPONENT-MARKER-TEXT-DISTINCT-FROM-ANYTHING-ELSE"


def test_last_error_exists_and_starts_empty():
    """The exact gap an earlier ad-hoc A/B test in this project had to patch
    around with a throwaway subclass -- fixed in the real class now."""
    optimizer = GEPAPromptOptimizer(model=ScriptedChatModel(script=[]))
    assert optimizer.last_error == ""


def test_last_error_is_set_on_a_reflection_failure():
    class Failing(ScriptedChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages, **kwargs):
            raise RuntimeError("reflection model is down")

    optimizer = GEPAPromptOptimizer(model=Failing(), n_candidates=1, seed=0)
    from meta_evolver.core.types import Action, StepRecord, Trajectory

    failing_traj = Trajectory(
        task_id="t1",
        steps=[StepRecord(step_idx=1, action=Action(name="look"))],
        success=False,
        score=0.0,
    )
    candidates = optimizer.propose("base prompt text", [failing_traj], generation=0)
    # invoke_text wraps the underlying exception as LLMError("Type: message"),
    # and str(exc) on that is what last_error actually stores.
    assert optimizer.last_error == "RuntimeError: reflection model is down"
    # The heuristic fallback still produces a candidate -- a failed
    # reflection call costs quality, not the whole proposal.
    assert len(candidates) == 1


def _weak_agent_reply(messages, tools):
    """Never submits -- every task fails, every generation, guaranteeing
    optimize_prompt fires."""
    return tool_call_message("inspect_service_logs", service_name="db-proxy")


class _GepaTestModel(ScriptedChatModel):
    """Plays the agent (always fails) and the GEPA reflection call (returns
    a fixed, recognizable marker for whichever component it is asked to
    rewrite)."""

    def __init__(self, **kwargs):
        super().__init__(responder=lambda m, t: None, **kwargs)

    def _reply(self, messages):
        system = str(getattr(messages[0], "content", "")) if messages else ""
        if "component" in system.lower() or "rewrite" in system.lower():
            return AIMessage(content=_MARKER)
        return _weak_agent_reply(messages, self.bound_tools)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._reply(list(messages)))])


def test_gepa_freezes_core_role_and_only_mutates_the_other_three():
    model = _GepaTestModel()
    optimizer = GEPAPromptOptimizer(model=model, n_candidates=4, seed=0)
    # Pre-seeded with a known ModularPrompt so propose() does not instead
    # derive generation 0's population by parsing the devops benchmark's own
    # (unrelated) system prompt text into four arbitrary paragraphs.
    optimizer.seed_population(ModularPrompt().render())

    evolver = MetaEvolver(
        benchmark="devops",
        chat_model=model,
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(
            generations=1,
            max_steps=2,
            validate_prompt=False,
            optimize_prompt=True,
            curriculum=False,
            patience=99,
        ),
        prompt_optimizer=optimizer,
        telemetry=False,
    )
    evolver.evolve()

    # validate_prompt=False adopts the first proposal unconditionally, so the
    # winning candidate's rendered text is directly inspectable.
    assert _DEFAULT_CORE_ROLE in evolver.prompt_template, "core_role must never be rewritten"
    assert _MARKER in evolver.prompt_template, "one of the other components must have changed"


def test_gepa_candidates_all_carry_the_frozen_core_role():
    """Not just the adopted one -- every candidate GEPA proposes this
    generation must share the same untouched core_role."""
    model = _GepaTestModel()
    optimizer = GEPAPromptOptimizer(model=model, n_candidates=4, seed=0)
    optimizer.seed_population(ModularPrompt().render())

    from meta_evolver.core.types import Action, StepRecord, Trajectory

    failing = [
        Trajectory(
            task_id=f"t{i}",
            steps=[StepRecord(step_idx=1, action=Action(name="look"))],
            success=False,
            score=0.0,
        )
        for i in range(3)
    ]
    candidates = optimizer.propose(ModularPrompt().render(), failing, generation=0)
    assert len(candidates) >= 4
    for cand in candidates:
        assert _DEFAULT_CORE_ROLE in cand.text


def _plain_optimizer_reply(messages, tools):
    """Distinguishes PromptOptimizer's own system prompt ("improve the
    system instructions...", per prompts/optimizer.py's OPTIMIZER_SYSTEM)
    from the agent role -- a bare responder that only knows the agent role
    would answer the optimizer's own call with a tool call, whose empty
    text content fails propose()'s 80-character minimum and silently yields
    zero candidates. That is a test-design trap, not a system behaviour;
    this is the same distinction test_evolution.py's DualModel makes."""
    system = str(getattr(messages[0], "content", "")) if messages else ""
    if "improve the system instructions" in system:
        return AIMessage(
            content=(
                "You are a disciplined incident-response agent. Confirm the owning "
                "service before patching, restart it, and submit only after the "
                "healthcheck returns 200."
            )
        )
    return _weak_agent_reply(messages, tools)


def test_default_optimizer_is_unaffected_when_none_is_injected():
    """The whole point of the injection point: not passing prompt_optimizer
    must be identical to before it existed."""
    evolver = MetaEvolver(
        benchmark="devops",
        chat_model=ScriptedChatModel(responder=_plain_optimizer_reply),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(
            generations=1, max_steps=2, validate_prompt=False, curriculum=False, patience=99
        ),
        telemetry=False,
    )
    evolver.evolve()
    assert "MUTATED" not in evolver.prompt_template
    assert evolver.prompt_version != "base"  # PromptOptimizer's own g0c0-style versioning
