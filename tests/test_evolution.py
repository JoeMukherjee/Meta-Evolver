"""The outer loop: memory curation, prompt selection, curriculum, reporting.

Runs the whole evolution graph against scripted models, so the coupling
between the three improvement channels is exercised without a network call.
"""
from __future__ import annotations

import json

from meta_evolver.core.evolver import MetaEvolver
from meta_evolver.core.registry import get_benchmark
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.harness.curriculum import Curriculum
from meta_evolver.llm.client import LLMResponse, ScriptedLLMClient, ToolCall
from meta_evolver.memory.bank import ReasoningMemoryBank

INDUCED = json.dumps(
    {
        "memories": [
            {
                "title": "Restart after patching",
                "scenario": "stale_config",
                "polarity": "success",
                "lesson": "Configuration writes do not take effect until the owner restarts.",
                "procedure": "1. patch 2. restart the owner 3. healthcheck 4. submit",
                "triggers": ["restart required", "config written"],
            }
        ]
    }
)


def _call(name, **kwargs):
    return LLMResponse(
        content="", tool_calls=[ToolCall(id=f"c-{name}", name=name, arguments=kwargs)]
    )


def _goal_for(text: str):
    """Read the intended fix out of the incident text.

    Lets one scripted responder solve every task in the suite, so the tests
    exercise the loop rather than a single hand-written trajectory.
    """
    if "401" in text or "Unauthorized" in text:
        return "auth-service", "active_key_version", "v2"
    if "cache" in text.lower():
        return "cache-worker", "ttl_seconds", 600
    if "429" in text:
        return "sync-worker", "poll_interval_ms", 2000
    if "space left" in text or "ingest" in text:
        return "log-shipper", "retention_days", 14
    return "db-proxy", "max_connections", 50


def agent_responder(messages, tools):
    """A competent agent, driven by which tools it has already used."""
    used = {
        m["tool_calls"][0]["function"]["name"]
        for m in messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    }
    text = " ".join(str(m.get("content", "")) for m in messages)
    target, key, value = _goal_for(text)

    if "inspect_service_logs" not in used:
        return _call("inspect_service_logs", service_name=target)
    if "patch_service_config" not in used:
        patch = {key: value}
        if target == "cache-worker":
            patch["eviction_policy"] = "allkeys-lru"
        return _call("patch_service_config", service_name=target, config_patch=patch)
    if "restart_service" not in used:
        return _call("restart_service", service_name=target)
    if "run_healthcheck" not in used:
        return _call("run_healthcheck", endpoint="health")
    return _call(
        "submit_resolution", root_cause="diagnosed", action_taken="patched and restarted"
    )


class DualClient(ScriptedLLMClient):
    """One client playing the agent, the inducer and the optimizer.

    Which role a call belongs to is decided by its system prompt -- which is
    how the real system distinguishes them too.
    """

    def __init__(self):
        super().__init__(responder=lambda m, t: LLMResponse())
        self.n_induction = 0
        self.n_optimization = 0

    def complete(self, messages, tools=None, response_format=None, max_tokens=None, **kwargs):
        self.calls.append(list(messages))
        system = str(messages[0].get("content", "")) if messages else ""
        if "distil agent execution traces" in system:
            self.n_induction += 1
            return LLMResponse(content=INDUCED)
        if "improve the system instructions" in system:
            self.n_optimization += 1
            return LLMResponse(
                content=(
                    "You are a disciplined incident-response agent. Confirm the owning "
                    "service before patching, restart it, and submit only after the "
                    "healthcheck returns 200.\n\n{memory_section}\n{guidance_section}"
                )
            )
        return agent_responder(list(messages), tools)


def build(**overrides) -> MetaEvolver:
    defaults = {
        "generations": 2,
        "max_steps": 10,
        "validate_prompt": False,
        "curriculum": False,
        "patience": 99,
    }
    config = EvolutionConfig(**{**defaults, **overrides})
    return MetaEvolver(
        benchmark="devops",
        client=DualClient(),
        bank=ReasoningMemoryBank(),
        config=config,
        curriculum=Curriculum(enabled=config.curriculum),
        telemetry=False,
    )


def test_evolution_runs_and_reports_every_generation():
    reports = build().evolve()

    assert [r.generation for r in reports] == [0, 1]
    assert all(r.n_tasks > 0 for r in reports)
    assert all(r.n_errors == 0 for r in reports)
    assert reports[0].pass_rate == 1.0


def test_induced_memories_land_in_the_bank():
    evolver = build()
    evolver.evolve()

    assert len(evolver.bank) >= 1
    assert any("restart" in m.lesson.lower() for m in evolver.bank)
    # Dedup: the same lesson induced twice takes one retrieval slot, not two.
    assert len(evolver.bank) <= 2


def test_credit_assignment_records_usage_across_generations():
    evolver = build(generations=3)
    evolver.evolve()

    used = [m for m in evolver.bank if m.uses > 0]
    assert used, "memories retrieved into a prompt must be credited"
    assert all(m.utility > 0.5 for m in used)


def test_prompt_optimization_is_skipped_when_nothing_fails():
    """Rewriting a winning prompt is a coin flip that can only cost."""
    evolver = build()
    evolver.evolve()

    assert evolver.client.n_optimization == 0
    assert evolver.prompt_version == "base"


def test_prompt_is_only_adopted_after_beating_the_incumbent():
    """A proposal is a hypothesis; validation is what makes it an improvement."""

    class WeakClient(DualClient):
        def complete(self, messages, tools=None, response_format=None, max_tokens=None, **kw):
            system = str(messages[0].get("content", "")) if messages else ""
            if "distil" in system or "improve the system instructions" in system:
                return super().complete(messages, tools, response_format, max_tokens, **kw)
            # An agent that only ever reads logs: every task fails, so the
            # optimizer runs -- and its candidate cannot validate any better.
            self.calls.append(list(messages))
            return _call("inspect_service_logs", service_name="db-proxy")

    evolver = MetaEvolver(
        benchmark="devops",
        client=WeakClient(),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(
            generations=1,
            max_steps=4,
            validate_prompt=True,
            validation_fraction=0.34,
            curriculum=False,
            patience=99,
        ),
        telemetry=False,
    )
    reports = evolver.evolve()

    assert evolver.client.n_optimization > 0, "failures must trigger a proposal"
    assert reports[0].validation_pass_rate is not None
    assert evolver.prompt_version == "base", "a candidate that ties must not be adopted"
    assert "kept incumbent" in " ".join(reports[0].notes)


def test_curriculum_escalates_once_the_agent_clears_the_level():
    evolver = build(generations=3, curriculum=True, curriculum_promote_at=0.7)
    evolver.curriculum = Curriculum(enabled=True)
    reports = evolver.evolve()

    levels = [r.curriculum_level for r in reports]
    assert levels[-1] > levels[0], "a solved level must get harder"
    assert evolver.curriculum.describe(levels[-1]) != evolver.curriculum.describe(0.0)


def test_early_stopping_fires_on_a_flat_run():
    reports = build(generations=8, patience=1).evolve()
    assert len(reports) < 8


def test_regression_counters_are_populated():
    reports = build(generations=2).evolve()
    # The scripted agent is stable, so churn is zero -- but the fields exist,
    # which is what the report contract promises.
    assert all(r.regressions == 0 for r in reports)
    assert all(r.recoveries == 0 for r in reports[1:])


def test_evaluate_measures_without_learning():
    evolver = build()
    before = len(evolver.bank)
    result = evolver.evaluate(split="eval")

    assert result["n_tasks"] == len(get_benchmark("devops").task_ids("eval"))
    assert len(evolver.bank) == before, "evaluation must not mutate the bank"


def test_memory_ablation_is_measurable():
    evolver = build()
    evolver.evolve()
    with_memory = evolver.evaluate(split="eval", use_memory=True)
    without = evolver.evaluate(split="eval", use_memory=False)

    assert with_memory["use_memory"] and not without["use_memory"]
    assert set(with_memory) == set(without)


def test_validation_split_never_leaks_into_training():
    bench = get_benchmark("devops")
    train, val = bench.sample(generation=0, validation_fraction=0.34, seed=3)
    assert val
    assert not set(train) & set(val)

    # And the holdout is stable across generations, so numbers are comparable.
    for generation in range(4):
        _, again = bench.sample(generation=generation, validation_fraction=0.34, seed=3)
        assert again == val


def test_progress_table_renders():
    evolver = build()
    evolver.evolve()
    table = evolver.render_progress()
    assert "gen" in table and "pass" in table and "delta:" in table


def test_run_artifacts_are_persisted(tmp_path):
    evolver = build()
    evolver.evolve()
    paths = evolver.save(
        memory_path=tmp_path / "memories.jsonl", prompt_path=tmp_path / "prompt.txt"
    )
    assert paths["memory"].exists() and paths["prompt"].exists()
    assert "{memory_section}" in paths["prompt"].read_text(encoding="utf-8")
