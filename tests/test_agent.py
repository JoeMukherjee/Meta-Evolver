"""Agent -- the Agno-shaped facade over MetaEvolver."""
from __future__ import annotations

from meta_evolver.agent import Agent
from meta_evolver.benchmarks.chat import END_CONVERSATION, SEND_MESSAGE
from meta_evolver.core.types import MemoryItem
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message


def _echo_agent(reply: str = "hi there", **kwargs) -> Agent:
    model = ScriptedChatModel(script=[tool_call_message(SEND_MESSAGE, text=reply)])
    return Agent(model=model, memory=False, **kwargs)


def test_run_returns_the_agents_reply():
    agent = _echo_agent("the answer is 4")
    assert agent.run("what is 2+2?") == "the answer is 4"


def test_tools_are_reachable_from_run():
    seen = {}

    def add(a: int, b: int) -> dict:
        """Add two numbers."""
        seen["called"] = (a, b)
        return {"sum": a + b}

    model = ScriptedChatModel(
        script=[
            tool_call_message("add", a=2, b=2),
            tool_call_message(SEND_MESSAGE, text="4"),
        ]
    )
    agent = Agent(model=model, tools=[add], memory=False)
    assert agent.run("add 2 and 2") == "4"
    assert seen["called"] == (2, 2)


def test_save_and_load_round_trips_prompt_and_memory(tmp_path):
    agent = _echo_agent()
    agent.evolver.prompt_template = "custom evolved prompt"
    agent.bank.add(MemoryItem(title="t", scenario="s", lesson="learned this"))
    agent.save(tmp_path)

    reloaded = Agent.load(tmp_path, model=ScriptedChatModel(script=[]))
    assert reloaded.evolver.prompt_template == "custom evolved prompt"
    assert any(m.lesson == "learned this" for m in reloaded.bank.items)


def test_attach_swaps_the_benchmark_and_keeps_the_bank():
    agent = _echo_agent()
    bank = agent.bank
    agent.attach("devops")
    assert agent.evolver.benchmark.name == "devops"
    assert agent.bank is bank


def test_chat_runs_a_full_session():
    calls = {"n": 0}

    def responder(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return tool_call_message(SEND_MESSAGE, text="hello!")
        return tool_call_message(END_CONVERSATION, reason="done")

    agent = Agent(model=ScriptedChatModel(responder=responder), memory=False)
    replies = []
    from meta_evolver.benchmarks.chat import ScriptedTurnSource

    trajectory = agent.chat(
        turn_source=ScriptedTurnSource(["hi", "bye"]), on_reply=replies.append
    )
    assert replies == ["hello!"]
    assert trajectory.error == ""
