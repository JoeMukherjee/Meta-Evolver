"""Shared fixtures.

The point of this file is that the whole engine is testable without a network
call. ``scripted_agent`` builds an LLM client that plays a fixed policy over
the DevOps benchmark, so the episode graph, the evolution loop, the curriculum
and the memory bank all run end to end in CI in under a second.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_evolver.llm.client import LLMResponse, ScriptedLLMClient, ToolCall  # noqa: E402


def call(name: str, **kwargs) -> LLMResponse:
    """One tool call, as an ``LLMResponse``."""
    return LLMResponse(
        content=f"calling {name}",
        tool_calls=[ToolCall(id=f"c-{name}", name=name, arguments=kwargs)],
    )


@pytest.fixture
def scripted_client():
    """Factory for a client that replays a fixed list of responses."""

    def build(*responses: LLMResponse) -> ScriptedLLMClient:
        return ScriptedLLMClient(script=list(responses))

    return build


@pytest.fixture
def solving_client():
    """A client that solves ``db_pool`` correctly, then idles.

    Written as a responder rather than a script so it is robust to the graph
    inserting an extra turn: a script that runs out mid-episode would fail the
    test for a reason unrelated to what it is checking.
    """

    def responder(messages, tools):
        used = {
            json.loads(m["tool_calls"][0]["function"]["arguments"] or "{}").get("service_name", "")
            + ":"
            + m["tool_calls"][0]["function"]["name"]
            for m in messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        }
        names = {u.split(":", 1)[1] for u in used}

        if "inspect_service_logs" not in names:
            return call("inspect_service_logs", service_name="payment-gateway")
        if "inspect_service_config" not in names:
            return call("inspect_service_config", service_name="db-proxy")
        if "patch_service_config" not in names:
            return call(
                "patch_service_config",
                service_name="db-proxy",
                config_patch={"max_connections": 50},
            )
        if "restart_service" not in names:
            return call("restart_service", service_name="db-proxy")
        if "run_healthcheck" not in names:
            return call("run_healthcheck", endpoint="payments/charge")
        return call(
            "submit_resolution",
            root_cause="db-proxy connection pool limit too low",
            action_taken="raised max_connections to 50 and restarted db-proxy",
        )

    return ScriptedLLMClient(responder=responder)


@pytest.fixture
def looping_client():
    """A client that never makes progress -- it re-reads the same logs forever.

    Used to prove stagnation eviction fires and that a hopeless episode still
    terminates cleanly at the step budget instead of hanging.
    """
    return ScriptedLLMClient(
        responder=lambda messages, tools: call(
            "inspect_service_logs", service_name="payment-gateway"
        )
    )
