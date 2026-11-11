"""LiveChatEnv / ChatBenchmark -- open-ended conversation through the same engine."""
from __future__ import annotations

from meta_evolver.benchmarks.chat import (
    END_CONVERSATION,
    SEND_MESSAGE,
    ChatBenchmark,
    LiveChatEnv,
    ScriptedTurnSource,
)
from meta_evolver.core.registry import get_benchmark
from meta_evolver.core.types import Action
from meta_evolver.graphs.episode import build_episode_graph, run_episode
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message


def test_reset_returns_the_first_human_turn():
    env = LiveChatEnv(turn_source=ScriptedTurnSource(["hello there"]))
    reset = env.reset()
    assert reset.observation.text == "hello there"
    assert not env.ended


def test_reset_with_no_turns_ends_immediately():
    env = LiveChatEnv(turn_source=ScriptedTurnSource([]))
    env.reset()
    assert env.ended


def test_send_message_returns_the_next_human_turn():
    env = LiveChatEnv(turn_source=ScriptedTurnSource(["hi", "how are you"]))
    env.reset()
    resp = env.step(Action(name=SEND_MESSAGE, kwargs={"text": "hello!"}))
    assert resp.observation.text == "how are you"
    assert not resp.terminated
    assert env.transcript[-2:] == [("assistant", "hello!"), ("user", "how are you")]


def test_session_ends_when_the_human_leaves():
    env = LiveChatEnv(turn_source=ScriptedTurnSource(["hi"]))
    env.reset()
    resp = env.step(Action(name=SEND_MESSAGE, kwargs={"text": "hello!"}))
    assert resp.terminated
    assert env.ended


def test_agent_can_end_the_conversation():
    env = LiveChatEnv(turn_source=ScriptedTurnSource(["hi", "bye"]))
    env.reset()
    resp = env.step(Action(name=END_CONVERSATION, kwargs={"reason": "done helping"}))
    assert resp.terminated
    assert env.ended


def test_on_reply_callback_fires_with_the_agent_text():
    seen = []
    env = LiveChatEnv(turn_source=ScriptedTurnSource(["hi"]), on_reply=seen.append)
    env.reset()
    env.step(Action(name=SEND_MESSAGE, kwargs={"text": "hello!"}))
    assert seen == ["hello!"]


def test_extra_tools_are_callable_and_scored_as_observations():
    def lookup(query: str) -> dict:
        """Search something."""
        return {"hits": [query]}

    env = LiveChatEnv(turn_source=ScriptedTurnSource(["hi"]), extra_tools={"lookup": lookup})
    env.reset()
    resp = env.step(Action(name="lookup", kwargs={"query": "widgets"}))
    assert resp.info["result"] == {"hits": ["widgets"]}
    assert not resp.terminated


def test_tool_schemas_include_send_message_and_extra_tools():
    def lookup(query: str) -> dict:
        """Search something."""
        return {}

    env = LiveChatEnv(turn_source=ScriptedTurnSource([]), extra_tools={"lookup": lookup})
    names = {s["function"]["name"] for s in env.available_tools()}
    assert names == {SEND_MESSAGE, END_CONVERSATION, "lookup"}


def test_chat_benchmark_is_registered():
    bench = get_benchmark("chat")
    assert isinstance(bench, ChatBenchmark)
    assert bench.task_ids("all") == ["session"]


def test_chat_benchmark_make_env_without_turn_source_raises():
    bench = ChatBenchmark()
    try:
        bench.make_env("session")
    except RuntimeError as exc:
        assert "turn_source" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_full_conversation_runs_through_the_episode_graph():
    """A scripted model chats for two exchanges, then ends the conversation --
    proving a chat session runs through the exact same graph a benchmark
    episode does, with no special-cased engine code."""
    calls = {"n": 0}

    def responder(messages, tools):
        calls["n"] += 1
        if calls["n"] <= 2:
            return tool_call_message(SEND_MESSAGE, text=f"reply {calls['n']}")
        return tool_call_message(END_CONVERSATION, reason="that's all")

    bench = ChatBenchmark()
    # Three human turns: one consumed at reset, one after each send_message,
    # leaving a third live turn for the agent to end_conversation on rather
    # than the episode ending early from the turn source simply running out.
    bench.turn_source = ScriptedTurnSource(["hi", "thanks, bye", "ok cool"])
    seen_replies: list[str] = []
    bench.on_reply = seen_replies.append

    graph = build_episode_graph(model=ScriptedChatModel(responder=responder))
    env = bench.make_env("session")
    trajectory = run_episode(
        graph,
        env=env,
        task_id="session",
        benchmark=bench.name,
        instruction=bench.instruction_for("session"),
        prompt_template=bench.system_prompt(),
        max_steps=10,
    )

    assert trajectory.error == ""
    assert seen_replies == ["reply 1", "reply 2"]
    assert [s.action.name for s in trajectory.steps] == [
        SEND_MESSAGE,
        SEND_MESSAGE,
        END_CONVERSATION,
    ]
    assert trajectory.steps[-1].terminated
