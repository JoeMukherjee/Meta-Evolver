"""The whole system, running with no API key and no network.

    python examples/offline_demo.py

A scripted model stands in for the LLM, so every moving part -- the episode
graph, memory induction, credit assignment, pruning, the curriculum -- runs
deterministically in about a second. Useful for seeing the shape of a run
before spending anything on one, and for checking an install.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_evolver import Curriculum, EvolutionConfig, MetaEvolver, ReasoningMemoryBank
from meta_evolver.llm.client import LLMResponse, ScriptedLLMClient, ToolCall

INDUCED = json.dumps(
    {
        "memories": [
            {
                "title": "Config changes need a restart",
                "scenario": "stale_config",
                "polarity": "success",
                "lesson": "A written configuration is not a live one until the owning service restarts.",
                "procedure": "1. identify the owner 2. patch 3. restart 4. healthcheck 5. submit",
                "triggers": ["restart required", "configuration updated"],
            }
        ]
    }
)

GOALS = {
    "401": ("auth-service", {"active_key_version": "v2"}),
    "cache": ("cache-worker", {"ttl_seconds": 600, "eviction_policy": "allkeys-lru"}),
    "429": ("sync-worker", {"poll_interval_ms": 2000}),
    "space left": ("log-shipper", {"retention_days": 14}),
}


def scripted_agent(messages, tools):
    """A competent agent: diagnose, patch, restart, verify, submit."""
    used = {
        m["tool_calls"][0]["function"]["name"]
        for m in messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    }
    text = " ".join(str(m.get("content", "")) for m in messages)
    service, patch = next(
        (goal for marker, goal in GOALS.items() if marker in text),
        ("db-proxy", {"max_connections": 50}),
    )

    def call(name, **kwargs):
        return LLMResponse(content="", tool_calls=[ToolCall(id="c", name=name, arguments=kwargs)])

    if "inspect_service_logs" not in used:
        return call("inspect_service_logs", service_name=service)
    if "patch_service_config" not in used:
        return call("patch_service_config", service_name=service, config_patch=patch)
    if "restart_service" not in used:
        return call("restart_service", service_name=service)
    if "run_healthcheck" not in used:
        return call("run_healthcheck", endpoint="health")
    return call("submit_resolution", root_cause="diagnosed", action_taken="patched and restarted")


class OfflineClient(ScriptedLLMClient):
    """Plays all three roles, dispatching on the system prompt."""

    def __init__(self):
        super().__init__(responder=lambda m, t: LLMResponse())

    def complete(self, messages, tools=None, response_format=None, max_tokens=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        if "distil agent execution traces" in system:
            return LLMResponse(content=INDUCED)
        if "improve the system instructions" in system:
            return LLMResponse(content="Be rigorous.\n\n{memory_section}\n{guidance_section}")
        return scripted_agent(list(messages), tools)


def main() -> int:
    evolver = MetaEvolver(
        benchmark="devops",
        client=OfflineClient(),
        bank=ReasoningMemoryBank(),
        config=EvolutionConfig(generations=4, max_steps=10, validate_prompt=False, patience=99),
        curriculum=Curriculum(enabled=True),
        telemetry=False,
    )

    print("Evolving offline (scripted model, no network)\n")
    evolver.evolve(on_report=lambda report: print(report.render()))

    print("\n" + evolver.render_progress())
    print(f"\ncurriculum reached: {evolver.curriculum.describe(evolver.curriculum_level)}")
    print(f"memory bank: {evolver.bank.stats()}")
    for memory in evolver.bank:
        print(f"  [{memory.utility:.2f} over {memory.uses} uses] {memory.title}")

    held_out = evolver.evaluate(split="eval", curriculum_level=0.0)
    print(f"\nheld-out pass rate: {held_out['pass_rate'] * 100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
