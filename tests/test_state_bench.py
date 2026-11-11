"""STATE-Bench integration -- pure conversion/extraction logic needs no
state_bench install; the Agent/Client classes are import-gated separately."""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from meta_evolver.benchmarks.state_bench import (
    _STATE_BENCH_AVAILABLE,
    build_memory_from_train_trajectories,
    conversation_to_messages,
    flat_to_nested_tool_schema,
    nested_to_flat_tool_schema,
    reply_to_turn_response,
)
from meta_evolver.memory.bank import ReasoningMemoryBank


def test_module_imports_without_state_bench_installed():
    # Getting this far at all is the assertion -- the import at module scope
    # above must not have raised even though state_bench isn't installed.
    import meta_evolver.benchmarks.state_bench as sb

    assert hasattr(sb, "MetaEvolverStateBenchAgent")


def test_flat_to_nested_tool_schema_round_trips():
    flat = {
        "type": "function",
        "name": "search_flights",
        "description": "Search for flights.",
        "parameters": {"type": "object", "properties": {"origin": {"type": "string"}}},
    }
    nested = flat_to_nested_tool_schema(flat)
    assert nested == {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for flights.",
            "parameters": {"type": "object", "properties": {"origin": {"type": "string"}}},
        },
    }
    assert nested_to_flat_tool_schema(nested) == flat


def test_conversation_to_messages_expands_embedded_tool_calls():
    conversation = [
        {"role": "user", "content": "Cancel booking ABC123."},
        {
            "role": "assistant",
            "content": "Let me look that up.",
            "tool_calls": [
                {
                    "name": "get_booking",
                    "arguments": {"booking_id": "ABC123"},
                    "result": {"status": "confirmed"},
                }
            ],
        },
        {"role": "user", "content": "Yes, cancel it."},
    ]
    messages = conversation_to_messages("You are an agent.", conversation)

    # system + user + (assistant w/ tool_calls) + tool result + user
    assert len(messages) == 5
    assert messages[0].content == "You are an agent."
    assert messages[2].tool_calls[0]["name"] == "get_booking"
    assert messages[2].tool_calls[0]["args"] == {"booking_id": "ABC123"}
    assert messages[3].content == "{'status': 'confirmed'}"
    assert messages[3].name == "get_booking"


def test_reply_to_turn_response_extracts_text_and_tool_calls():
    reply = AIMessage(
        content="Looking that up.",
        tool_calls=[{"name": "get_booking", "args": {"booking_id": "X"}, "id": "1", "type": "tool_call"}],
    )
    text, calls = reply_to_turn_response(reply)
    assert text == "Looking that up."
    assert calls == [{"name": "get_booking", "arguments": {"booking_id": "X"}}]


def test_reply_with_no_tool_calls():
    reply = AIMessage(content="All done.")
    text, calls = reply_to_turn_response(reply)
    assert text == "All done."
    assert calls == []


def test_build_memory_from_train_trajectories_reads_the_official_shape(tmp_path):
    domain_dir = tmp_path / "travel"
    domain_dir.mkdir()
    (domain_dir / "task1.json").write_text(
        json.dumps(
            {
                "conversation": [
                    {"role": "system", "content": "You are a travel agent."},
                    {"role": "user", "content": "Book a flight to SFO."},
                    {
                        "role": "assistant",
                        "content": "Booked.",
                        "tool_calls": [
                            {"name": "search_flights", "arguments": {"destination": "SFO"}, "result": {}}
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (domain_dir / "task2.json").write_text(json.dumps({"conversation": []}), encoding="utf-8")

    bank = ReasoningMemoryBank()
    added = build_memory_from_train_trajectories("travel", tmp_path, bank)

    assert added == 2
    lessons = [item.lesson for item in bank.items]
    assert any("search_flights" in lesson for lesson in lessons)
    assert any("no tool calls" in lesson for lesson in lessons)
    assert all(item.benchmark == "state-bench-travel" for item in bank.items)


def test_build_memory_ignores_other_domains(tmp_path):
    (tmp_path / "travel").mkdir()
    (tmp_path / "travel" / "t1.json").write_text(json.dumps({"conversation": []}), encoding="utf-8")
    (tmp_path / "shopping_assistant").mkdir()
    (tmp_path / "shopping_assistant" / "t1.json").write_text(json.dumps({"conversation": []}), encoding="utf-8")

    bank = ReasoningMemoryBank()
    added = build_memory_from_train_trajectories("travel", tmp_path, bank)
    assert added == 1


@pytest.mark.skipif(not _STATE_BENCH_AVAILABLE, reason="state_bench package not installed")
def test_agent_memory_tools_are_read_only_and_wired_to_the_bank():  # pragma: no cover
    from meta_evolver.benchmarks.state_bench import MetaEvolverStateBenchAgent

    schemas = {s["name"] for s in MetaEvolverStateBenchAgent.memory_tool_schemas(object())}
    assert schemas == {"retrieve_memory"}
