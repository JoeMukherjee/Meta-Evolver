"""End-to-end graph behaviour, entirely offline."""
from __future__ import annotations

import pytest

from meta_evolver.core.registry import get_benchmark
from meta_evolver.core.rules import IntermittentFault, VerificationGate
from meta_evolver.graphs.episode import build_episode_graph, run_episode
from meta_evolver.llm.client import LLMError, LLMResponse, ScriptedLLMClient


def solve(benchmark, client, task_id="db_pool", bank=None, wrap=None, max_steps=15):
    graph = build_episode_graph(client=client, bank=bank)
    env = benchmark.make_env(task_id)
    if wrap is not None:
        env = wrap(env)
    return run_episode(
        graph,
        env=env,
        task_id=task_id,
        benchmark=benchmark.name,
        instruction=benchmark.instruction_for(task_id),
        prompt_template=benchmark.system_prompt(),
        max_steps=max_steps,
    )


def test_episode_graph_solves_a_task(solving_client):
    trajectory = solve(get_benchmark("devops"), solving_client)
    assert trajectory.success
    assert trajectory.score == 1.0
    assert trajectory.error == ""
    # diagnose, patch, restart, verify, submit
    assert [s.action.name for s in trajectory.steps][-3:] == [
        "restart_service",
        "run_healthcheck",
        "submit_resolution",
    ]


def test_hopeless_episode_terminates_at_the_step_budget(looping_client):
    trajectory = solve(get_benchmark("devops"), looping_client, max_steps=6)
    assert not trajectory.success
    assert trajectory.n_steps == 6
    assert trajectory.error == ""  # a real failure, not an infrastructure one
    assert trajectory.usable


def test_stagnation_evicts_memory_mid_episode(looping_client):
    from meta_evolver.core.types import MemoryItem
    from meta_evolver.memory.bank import ReasoningMemoryBank

    bank = ReasoningMemoryBank(
        [MemoryItem(id="m1", title="Read logs", lesson="the answer is always in the logs")]
    )
    trajectory = solve(get_benchmark("devops"), looping_client, bank=bank, max_steps=12)
    assert trajectory.retrieved_memory_ids == ["m1"]
    assert trajectory.memory_evicted_at is not None


def test_api_failure_is_recorded_as_an_error_not_a_task_failure():
    """The distinction the whole learning signal depends on.

    A rate-limited episode that scored as a task failure would teach the memory
    bank and the prompt optimizer a lesson about a bug in the provider.
    """

    class Failing(ScriptedLLMClient):
        def complete(self, *args, **kwargs):
            raise LLMError("RateLimitError: 429")

    trajectory = solve(get_benchmark("devops"), Failing())
    assert not trajectory.success
    assert "RateLimitError" in trajectory.error
    assert not trajectory.usable  # excluded from induction, credit and scoring


def test_prose_only_model_is_nudged_then_gives_up():
    client = ScriptedLLMClient(
        responder=lambda messages, tools: LLMResponse(content="Let me think about this.")
    )
    trajectory = solve(get_benchmark("devops"), client, max_steps=10)
    assert trajectory.n_steps == 0
    assert trajectory.error == ""
    assert len(client.calls) <= 5  # bounded, not a runaway


def test_verification_gate_blocks_a_premature_submission():
    """Submitting before the healthcheck passes must be rejected."""
    from meta_evolver.llm.client import ToolCall

    def responder(messages, tools):
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="submit_resolution",
                    arguments={"root_cause": "guessing", "action_taken": "nothing"},
                )
            ],
        )

    trajectory = solve(
        get_benchmark("devops"),
        ScriptedLLMClient(responder=responder),
        wrap=lambda env: VerificationGate(env),
        max_steps=4,
    )
    assert not trajectory.success
    assert all(step.blocked for step in trajectory.steps)


def test_injected_faults_are_flagged_on_the_step(solving_client):
    trajectory = solve(
        get_benchmark("devops"),
        solving_client,
        wrap=lambda env: IntermittentFault(env, failure_rate=1.0, max_faults=1, seed=7),
    )
    assert sum(1 for s in trajectory.steps if s.perturbed) == 1


def test_harness_perturbation_is_reproducible_given_a_seed(solving_client):
    def run(seed):
        client = ScriptedLLMClient(responder=solving_client.responder)
        return solve(
            get_benchmark("devops"),
            client,
            wrap=lambda env: IntermittentFault(env, failure_rate=0.5, seed=seed),
        )

    a, b = run(11), run(11)
    assert [s.perturbed for s in a.steps] == [s.perturbed for s in b.steps]


def test_textgame_benchmark_runs_through_the_same_graph():
    """A string-action environment needs no special case in the engine."""
    from meta_evolver.llm.client import ToolCall

    def responder(messages, tools):
        last = messages[-1]["content"]
        # Walk the admissible list in order: a dumb but valid policy.
        import ast
        import re

        match = re.search(r"Admissible commands: (\[.*\])", str(last))
        commands = ast.literal_eval(match.group(1)) if match else ["look"]
        taken = {
            m["tool_calls"][0]["function"]["arguments"]
            for m in messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        }
        for cmd in commands:
            if f'"{cmd}"' not in " ".join(taken):
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="c", name="do", arguments={"text": cmd})],
                )
        return LLMResponse(
            content="", tool_calls=[ToolCall(id="c", name="do", arguments={"text": commands[0]})]
        )

    bench = get_benchmark("textgame")
    trajectory = solve(
        bench, ScriptedLLMClient(responder=responder), task_id="bathroom_soap_id", max_steps=10
    )
    assert trajectory.error == ""
    assert trajectory.n_steps > 0
    # Text actions render readably for the inducer, not as tool-call noise.
    assert trajectory.steps[0].action.render().startswith("go to")


@pytest.mark.parametrize("task_id", ["db_pool", "jwt_auth", "cache_oom", "rate_limit", "disk_pressure"])
def test_every_devops_task_is_solvable_and_starts_unsolved(task_id):
    env = get_benchmark("devops").make_env(task_id)
    env.reset(options={"task_id": task_id})
    assert not env.evaluate().success
    assert env.evaluate().score == 0.0
