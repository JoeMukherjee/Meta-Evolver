"""The whole system, running with no API key and no network.

    python examples/offline_demo.py

A scripted LangChain chat model stands in for the LLM, so every moving part --
the episode graph, memory induction, credit assignment, pruning, the
curriculum -- runs deterministically in about a second. Useful for seeing the
shape of a run before spending anything on one, and for checking an install.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from meta_evolver import (
    Curriculum,
    EvolutionConfig,
    MetaEvolver,
    ReasoningMemoryBank,
    ScriptedChatModel,
    tool_call_message,
)

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
    "ingest": ("log-shipper", {"retention_days": 14}),
}


def scripted_agent(messages):
    """A competent agent: diagnose, patch, restart, verify, submit."""
    used = {
        call["name"] for m in messages for call in (getattr(m, "tool_calls", None) or [])
    }
    text = " ".join(str(getattr(m, "content", "")) for m in messages)
    service, patch = next(
        (goal for marker, goal in GOALS.items() if marker in text),
        ("db-proxy", {"max_connections": 50}),
    )

    if "inspect_service_logs" not in used:
        return tool_call_message("inspect_service_logs", service_name=service)
    if "patch_service_config" not in used:
        return tool_call_message(
            "patch_service_config", service_name=service, config_patch=patch
        )
    if "restart_service" not in used:
        return tool_call_message("restart_service", service_name=service)
    if "run_healthcheck" not in used:
        return tool_call_message("run_healthcheck", endpoint="health")
    return tool_call_message(
        "submit_resolution", root_cause="diagnosed", action_taken="patched and restarted"
    )


class OfflineModel(ScriptedChatModel):
    """Plays all three roles, dispatching on the system prompt."""

    def __init__(self):
        super().__init__(responder=lambda m, t: AIMessage(content=""))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        system = str(getattr(messages[0], "content", "")) if messages else ""
        if "distil agent execution traces" in system:
            reply = AIMessage(content=INDUCED)
        elif "improve the system instructions" in system:
            reply = AIMessage(content="Be rigorous.\n\n{memory_section}\n{guidance_section}")
        else:
            reply = scripted_agent(list(messages))
        return ChatResult(generations=[ChatGeneration(message=reply)])


def main() -> int:
    evolver = MetaEvolver(
        benchmark="devops",
        chat_model=OfflineModel(),
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
    print(
        "\nembeddings: "
        + ("remote" if evolver.embedder.remote_available else "local fallback encoder")
        + f" ({evolver.embedder.n_remote} remote, {evolver.embedder.n_fallback} local)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
