"""Benchmark adapters, harness layers, and the LLM parameter contract."""
from __future__ import annotations

import pytest

from meta_evolver.benchmarks.custom import FunctionBenchmark, Task, ToolCallRecord, derive_schema
from meta_evolver.benchmarks.devops import TASKS, DevOpsIncidentEnv
from meta_evolver.benchmarks.external import ExternalEnvAdapter, TextEnvAdapter
from meta_evolver.core.registry import get_benchmark, list_benchmarks, register_benchmark
from meta_evolver.core.rules import ActionBudget, ObservationNoise
from meta_evolver.core.types import Action, Observation
from meta_evolver.harness.curriculum import Curriculum
from meta_evolver.llm.client import (
    DEPRECATED_SAMPLING_PARAMS,
    LiteLLMClient,
    sampling_params_deprecated,
)
from meta_evolver.tools.routing import ToolRouter

# --- registry ---------------------------------------------------------------


def test_builtin_benchmarks_are_discoverable():
    assert {"devops", "textgame"} <= set(list_benchmarks())


def test_unknown_benchmark_names_the_alternatives():
    with pytest.raises(KeyError, match="devops"):
        get_benchmark("does-not-exist")


def test_registering_a_benchmark_is_one_decorator():
    from meta_evolver.benchmarks.base import BenchmarkAdapter

    @register_benchmark("throwaway-bench")
    class Throwaway(BenchmarkAdapter):
        def task_ids(self, split="train"):
            return ["t1"]

        def make_env(self, task_id, curriculum_level=0.0, seed=0):
            return DevOpsIncidentEnv("db_pool")

    assert get_benchmark("throwaway-bench").name == "throwaway-bench"


# --- Gemini sampling-parameter contract -------------------------------------


@pytest.mark.parametrize(
    "model",
    ["gemini/gemini-3-flash", "gemini-2.5-pro", "vertex_ai/gemini-3-pro", "gemini/gemini-embedding-001"],
)
def test_gemini_routes_are_recognized(model):
    assert sampling_params_deprecated(model)


@pytest.mark.parametrize("model", ["openai/gpt-4.1", "anthropic/claude-opus-4-7", "groq/llama-3.3"])
def test_other_providers_still_accept_sampling_params(model):
    assert not sampling_params_deprecated(model)


def test_deprecated_sampling_params_are_stripped_for_gemini():
    """Google removed the manual sampling overrides; sending them is wrong.

    Enforced at the client rather than per call site, so a config carrying
    `temperature: 0.4` stays valid on Gemini and on everything else.
    """
    client = LiteLLMClient(model="gemini/gemini-3-flash", api_key="test", temperature=0.4)
    request = client._prepare(top_p=0.9, top_k=40, temperature=0.7)

    for param in DEPRECATED_SAMPLING_PARAMS:
        assert param not in request
    assert request["model"] == "gemini/gemini-3-flash"


def test_sampling_params_survive_for_providers_that_accept_them():
    client = LiteLLMClient(model="openai/gpt-4.1", api_key="test", temperature=0.4)
    request = client._prepare(top_p=0.9)
    assert request["temperature"] == 0.4
    assert request["top_p"] == 0.9


# --- DevOps benchmark -------------------------------------------------------


def test_tools_are_scoped_to_the_environment():
    """An agent offered tools it cannot use wastes steps discovering that."""
    env = DevOpsIncidentEnv("db_pool")
    names = {t["function"]["name"] for t in env.available_tools()}
    assert names == {
        "inspect_service_logs",
        "inspect_service_config",
        "query_metrics",
        "patch_service_config",
        "restart_service",
        "run_healthcheck",
        "submit_resolution",
    }


def test_patch_without_restart_fails_the_healthcheck():
    env = DevOpsIncidentEnv("db_pool")
    env.reset()
    env.step(Action(name="patch_service_config",
                    kwargs={"service_name": "db-proxy", "config_patch": {"max_connections": 50}}))
    resp = env.step(Action(name="run_healthcheck", kwargs={"endpoint": "payments/charge"}))
    assert resp.info["result"]["status_code"] == 503
    assert not env.evaluate().success


def test_partial_credit_tracks_the_four_beat_discipline():
    env = DevOpsIncidentEnv("db_pool")
    env.reset()
    assert env.evaluate().score == 0.0

    env.step(Action(name="patch_service_config",
                    kwargs={"service_name": "db-proxy", "config_patch": {"max_connections": 50}}))
    after_patch = env.evaluate().score
    env.step(Action(name="restart_service", kwargs={"service_name": "db-proxy"}))
    assert env.evaluate().score > after_patch


def test_masking_the_symptom_forfeits_the_task():
    """rate_limit exists to catch "find the number, make it bigger".

    Raising the gateway limit removes the 429s and leaves a client polling
    fifty times a second. The environment scores that as a failure.
    """
    env = DevOpsIncidentEnv("rate_limit")
    env.reset()
    env.step(Action(name="patch_service_config",
                    kwargs={"service_name": "api-gateway", "config_patch": {"rate_limit_rpm": 100000}}))
    env.step(Action(name="patch_service_config",
                    kwargs={"service_name": "sync-worker", "config_patch": {"poll_interval_ms": 5000}}))
    env.step(Action(name="restart_service", kwargs={"service_name": "sync-worker"}))
    env.step(Action(name="run_healthcheck", kwargs={"endpoint": "v1/items"}))
    env.step(Action(name="submit_resolution", kwargs={"root_cause": "x", "action_taken": "y"}))

    result = env.evaluate()
    assert not result.success
    assert result.metrics["masked_symptom"]
    assert result.score == 0.0


def test_cache_task_needs_both_keys_fixed():
    env = DevOpsIncidentEnv("cache_oom")
    env.reset()
    env.step(Action(name="patch_service_config",
                    kwargs={"service_name": "cache-worker", "config_patch": {"ttl_seconds": 600}}))
    env.step(Action(name="restart_service", kwargs={"service_name": "cache-worker"}))
    assert env.step(Action(name="run_healthcheck", kwargs={"endpoint": "cache/health"})
                    ).info["result"]["status_code"] == 503

    env.step(Action(name="patch_service_config",
                    kwargs={"service_name": "cache-worker",
                            "config_patch": {"eviction_policy": "allkeys-lru"}}))
    env.step(Action(name="restart_service", kwargs={"service_name": "cache-worker"}))
    assert env.step(Action(name="run_healthcheck", kwargs={"endpoint": "cache/health"})
                    ).info["result"]["status_code"] == 200


def test_eval_split_is_held_out_of_training():
    bench = get_benchmark("devops")
    assert not set(bench.task_ids("train")) & set(bench.task_ids("eval"))
    assert set(bench.task_ids("all")) == set(TASKS)


def test_env_state_exposes_the_generic_verified_flag():
    """VerificationGate reads `verified`, not a benchmark-specific key.

    That naming is what lets one harness layer apply to every benchmark.
    """
    env = DevOpsIncidentEnv("db_pool")
    env.reset()
    assert env.get_env_state()["verified"] is False


# --- harness layers ---------------------------------------------------------


def test_action_budget_truncates_and_warns():
    env = ActionBudget(DevOpsIncidentEnv("db_pool"), budget=3)
    env.reset()
    env.step(Action(name="run_healthcheck", kwargs={"endpoint": "x"}))
    warned = env.step(Action(name="run_healthcheck", kwargs={"endpoint": "x"}))
    assert "BUDGET" in warned.observation.text

    exhausted = env.step(Action(name="run_healthcheck", kwargs={"endpoint": "x"}))
    assert exhausted.truncated


def test_observation_noise_adds_distractors_without_changing_state():
    env = ObservationNoise(DevOpsIncidentEnv("db_pool"), rate=1.0, seed=5)
    env.reset()
    resp = env.step(Action(name="inspect_service_logs", kwargs={"service_name": "db-proxy"}))
    assert any(d in resp.observation.text for d in ObservationNoise.DISTRACTORS)
    assert env.get_env_state()["verified"] is False


def test_curriculum_bands_escalate_distinctly():
    curr = Curriculum()
    assert curr.band_for(0.0).name == "clean"
    assert curr.band_for(0.5).verification_gate
    assert curr.band_for(0.9).budget

    stack = curr.wrap(DevOpsIncidentEnv("db_pool"), level=0.9, seed=1).layers()
    assert "IntermittentFault" in stack and "VerificationGate" in stack
    assert "ObservationNoise" in stack and "ActionBudget" in stack


def test_curriculum_hysteresis_prevents_oscillation():
    curr = Curriculum(step=0.2)
    assert curr.adjust(0.4, pass_rate=0.9) == pytest.approx(0.6)
    assert curr.adjust(0.4, pass_rate=0.5) == pytest.approx(0.4)  # the dead band
    assert curr.adjust(0.4, pass_rate=0.1) == pytest.approx(0.2)


# --- custom / external adapters ---------------------------------------------


def test_tool_schemas_derive_from_signature_and_docstring():
    def lookup(query: str, limit: int = 5) -> dict:
        """Search the index.

        Longer prose that should not reach the model.
        """
        return {}

    schema = derive_schema("lookup", lookup)["function"]
    assert schema["description"] == "Search the index."
    assert schema["parameters"]["properties"]["limit"]["type"] == "integer"
    assert schema["parameters"]["required"] == ["query"]


def test_function_benchmark_scores_a_verifier():
    def answer(text: str) -> dict:
        """Submit the final answer."""
        return {"answer": text}

    bench = FunctionBenchmark(
        name="mini",
        tools={"answer": answer},
        tasks=[
            Task(
                id="t1",
                instruction="Say 42.",
                verify=lambda calls: any("42" in str(c.result) for c in calls),
                terminal_tools=("answer",),
            )
        ],
    )
    env = bench.make_env("t1")
    env.reset()
    resp = env.step(Action(name="answer", kwargs={"text": "42"}))
    assert resp.terminated
    assert env.evaluate().success


def test_tool_errors_become_observations_not_crashes():
    def explode(x: str) -> dict:
        """Always fails."""
        raise ValueError("boom")

    bench = FunctionBenchmark(name="m", tools={"explode": explode}, tasks=[Task(id="t")])
    env = bench.make_env("t")
    env.reset()
    resp = env.step(Action(name="explode", kwargs={"x": "1"}))
    assert "boom" in str(resp.info["result"])
    assert not resp.terminated


def test_broken_verifier_is_surfaced_not_silently_failed():
    """A crashing verifier looks exactly like an agent regression otherwise."""

    def bad(calls: list[ToolCallRecord]):
        raise RuntimeError("verifier bug")

    bench = FunctionBenchmark(name="m", tools={}, tasks=[Task(id="t", verify=bad)])
    env = bench.make_env("t")
    env.reset()
    assert "verifier bug" in env.evaluate().metrics["verifier_error"]


def test_external_adapter_wraps_a_gym_style_environment():
    class Gymish:
        def __init__(self):
            self.n = 0

        def reset(self, seed=None, options=None):
            self.n = 0
            return "you are in a room", {}

        def step(self, action):
            self.n += 1
            return f"did {action}", 1.0 if self.n >= 2 else 0.0, self.n >= 2, False, {"won": self.n >= 2}

    env = TextEnvAdapter(Gymish())
    env.reset()
    env.step(Action(name="do", kwargs={"text": "look"}))
    resp = env.step(Action(name="do", kwargs={"text": "leave"}))
    assert resp.terminated
    assert env.evaluate().success
    assert env.available_tools()[0]["function"]["name"] == "do"


def test_external_adapter_prefers_the_environments_own_verdict():
    class WithEvaluate:
        def reset(self, seed=None, options=None):
            return Observation(text="hi")

        def step(self, action):
            return Observation(text="ok"), 1.0, False, False, {}

        def evaluate(self):
            class R:
                success, score, metrics = False, 0.3, {}

            return R()

    env = ExternalEnvAdapter(WithEvaluate())
    env.reset()
    env.step(Action(name="x"))
    result = env.evaluate()
    assert not result.success and result.score == 0.3


# --- tool routing -----------------------------------------------------------


def test_small_registries_pass_through_unrouted():
    """Ranking seven tools spends an embedding call to change nothing."""
    router = ToolRouter(k=3, min_tools_to_route=10)
    tools = DevOpsIncidentEnv("db_pool").available_tools()
    assert router.select(tools, "anything") == tools


def test_routing_keeps_relevant_tools_and_never_drops_terminal_ones():
    from meta_evolver.benchmarks.base import tool_schema

    tools = [
        tool_schema("search_logs", "Read log lines from a service"),
        tool_schema("restart_service", "Restart a service so config takes effect"),
        *[tool_schema(f"unrelated_{i}", f"Render a chart of type {i}") for i in range(10)],
        tool_schema("submit_resolution", "Close the incident once verified"),
    ]
    kept = {
        t["function"]["name"]
        for t in ToolRouter(k=4, min_tools_to_route=5).select(tools, "read the service logs")
    }
    assert "search_logs" in kept
    assert "submit_resolution" in kept, "a terminal action must never be routed away"
    assert len(kept) <= 5
