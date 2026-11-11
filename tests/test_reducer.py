"""EvidenceReducer: below-threshold pass-through, real compression, and
graceful fallback on failure -- all offline via ScriptedChatModel."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from meta_evolver.llm.client import ScriptedChatModel
from meta_evolver.tools.reducer import EvidenceReducer


def test_short_text_is_returned_unchanged_no_model_call():
    calls = []
    model = ScriptedChatModel(
        responder=lambda m, t: calls.append(1) or AIMessage(content="compressed")
    )
    reducer = EvidenceReducer(model, min_chars=2000)
    text = "short observation"
    assert reducer.reduce(text) == text
    assert calls == []
    assert reducer.n_calls == 0


def test_long_text_above_threshold_is_compressed():
    model = ScriptedChatModel(
        responder=lambda m, t: AIMessage(content="error: pool exhausted at 42 connections")
    )
    reducer = EvidenceReducer(model, min_chars=50)
    text = "log line\n" * 20
    out = reducer.reduce(text)
    assert out == "error: pool exhausted at 42 connections"
    assert reducer.n_calls == 1
    assert reducer.chars_saved > 0


def test_reducer_prompt_includes_the_raw_text():
    seen = []

    def responder(messages, tools):
        seen.append(messages[-1].content)
        return AIMessage(content="ok")

    model = ScriptedChatModel(responder=responder)
    reducer = EvidenceReducer(model, min_chars=10)
    reducer.reduce("x" * 100)
    assert "x" * 100 in seen[0]


def test_model_failure_falls_back_to_plain_truncation():
    class Failing(ScriptedChatModel):
        def invoke(self, messages, **kwargs):
            raise RuntimeError("model down")

        def bind_tools(self, tools, **kwargs):
            return self

    reducer = EvidenceReducer(Failing(), min_chars=10, max_output_chars=20)
    text = "a" * 200
    out = reducer.reduce(text)
    assert out == text[:20]
    assert reducer.n_fallbacks == 1
    assert reducer.n_calls == 0


def test_empty_model_reply_falls_back_to_truncation():
    model = ScriptedChatModel(responder=lambda m, t: AIMessage(content=""))
    reducer = EvidenceReducer(model, min_chars=10, max_output_chars=15)
    text = "b" * 100
    assert reducer.reduce(text) == text[:15]
    assert reducer.n_fallbacks == 1


def test_a_compressed_reply_longer_than_max_output_is_itself_truncated():
    long_reply = "y" * 5000
    model = ScriptedChatModel(responder=lambda m, t: AIMessage(content=long_reply))
    reducer = EvidenceReducer(model, min_chars=10, max_output_chars=100)
    out = reducer.reduce("x" * 200)
    assert len(out) == 100
    assert out == long_reply[:100]


def test_fallback_never_raises_regardless_of_exception_type():
    class WeirdFailure(ScriptedChatModel):
        def invoke(self, messages, **kwargs):
            raise ValueError("something unexpected")

        def bind_tools(self, tools, **kwargs):
            return self

    reducer = EvidenceReducer(WeirdFailure(), min_chars=1)
    out = reducer.reduce("some text that is long enough to trigger a call")
    assert out  # did not raise, returned a fallback
