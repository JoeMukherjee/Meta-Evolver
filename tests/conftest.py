"""Shared fixtures.

The point of this file is that the whole engine is testable without a network
call. The scripted chat models below are real LangChain ``BaseChatModel``
instances, so the graphs exercise exactly the code path a live model takes --
``bind_tools``, ``AIMessage.tool_calls``, ``ToolMessage`` round-tripping --
rather than a parallel mock path that can drift from it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_evolver.llm.client import ScriptedChatModel, tool_call_message  # noqa: E402


def tools_used(messages) -> set[str]:
    """Names of every tool call already made in a transcript."""
    used: set[str] = set()
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            used.add(call["name"])
    return used


def transcript_text(messages) -> str:
    """All message content concatenated, for keyword matching in a responder."""
    return " ".join(str(getattr(m, "content", "")) for m in messages)


@pytest.fixture
def scripted_model():
    """Factory for a model that replays a fixed list of replies."""

    def build(*replies: AIMessage) -> ScriptedChatModel:
        return ScriptedChatModel(script=list(replies))

    return build


def solving_responder(messages, tools):
    """A competent DevOps agent: diagnose, patch, restart, verify, submit.

    Written as a responder rather than a fixed script so it is robust to the
    graph inserting an extra turn -- a script that ran out mid-episode would
    fail tests for a reason unrelated to what they check.
    """
    used = tools_used(messages)
    if "inspect_service_logs" not in used:
        return tool_call_message("inspect_service_logs", service_name="payment-gateway")
    if "inspect_service_config" not in used:
        return tool_call_message("inspect_service_config", service_name="db-proxy")
    if "patch_service_config" not in used:
        return tool_call_message(
            "patch_service_config",
            service_name="db-proxy",
            config_patch={"max_connections": 50},
        )
    if "restart_service" not in used:
        return tool_call_message("restart_service", service_name="db-proxy")
    if "run_healthcheck" not in used:
        return tool_call_message("run_healthcheck", endpoint="payments/charge")
    return tool_call_message(
        "submit_resolution",
        root_cause="db-proxy connection pool limit too low",
        action_taken="raised max_connections to 50 and restarted db-proxy",
    )


@pytest.fixture
def solving_model():
    """A model that solves ``db_pool`` correctly."""
    return ScriptedChatModel(responder=solving_responder)


@pytest.fixture
def looping_model():
    """A model that never makes progress -- it re-reads the same logs forever.

    Used to prove stagnation eviction fires and that a hopeless episode still
    terminates cleanly at the step budget instead of hanging.
    """
    return ScriptedChatModel(
        responder=lambda messages, tools: tool_call_message(
            "inspect_service_logs", service_name="payment-gateway"
        )
    )
