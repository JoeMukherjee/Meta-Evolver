"""LongMemEvalEnv / LongMemEvalBenchmark -- schema-faithful, no network."""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from meta_evolver.benchmarks.longmemeval import (
    LongMemEvalBenchmark,
    LongMemEvalEnv,
    load_longmemeval,
)
from meta_evolver.core.types import Action
from meta_evolver.llm.client import ScriptedChatModel

_RAW = [
    {
        "question_id": "q1",
        "question": "What did the user name their dog?",
        "answer": "Biscuit",
        "question_type": "single-session-user",
        "question_date": "2026-01-05",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2026-01-01", "2026-01-03"],
        "haystack_sessions": [
            [{"role": "user", "content": "I got a new dog, named him Biscuit."}],
            [{"role": "user", "content": "The weather is nice today."}],
        ],
    },
    {
        "question_id": "q2_abs",
        "question": "What is the user's middle name?",
        "answer": "This was never mentioned in the conversation.",
        "question_type": "single-session-user",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2026-01-01"],
        "haystack_sessions": [[{"role": "user", "content": "I like hiking."}]],
    },
]


@pytest.fixture
def data_path(tmp_path):
    path = tmp_path / "longmemeval_mini.json"
    path.write_text(json.dumps(_RAW), encoding="utf-8")
    return path


def test_load_parses_sessions_and_abstention_flag(data_path):
    records = load_longmemeval(data_path)
    assert len(records) == 2
    assert records[0].question_id == "q1"
    assert records[0].sessions[0]["id"] == "s1"
    assert "Biscuit" in records[0].sessions[0]["text"]
    assert not records[0].is_abstention
    assert records[1].is_abstention


def test_env_reset_previews_sessions_without_full_text(data_path):
    record = load_longmemeval(data_path)[0]
    env = LongMemEvalEnv(record)
    reset = env.reset()
    assert "s1" in reset.observation.text
    assert "s2" in reset.observation.text
    assert reset.info["instruction"] == record.question


def test_read_session_returns_full_text(data_path):
    record = load_longmemeval(data_path)[0]
    env = LongMemEvalEnv(record)
    env.reset()
    resp = env.step(Action(name="read_session", kwargs={"session_id": "s1"}))
    assert "Biscuit" in resp.observation.text


def test_search_sessions_finds_the_right_one(data_path):
    record = load_longmemeval(data_path)[0]
    env = LongMemEvalEnv(record)
    env.reset()
    resp = env.step(Action(name="search_sessions", kwargs={"query": "biscuit"}))
    assert "s1" in resp.observation.text
    assert "s2" not in resp.observation.text


def test_submit_answer_terminates_and_no_judge_falls_back_to_substring(data_path):
    record = load_longmemeval(data_path)[0]
    env = LongMemEvalEnv(record, judge_model=None)
    env.reset()
    resp = env.step(Action(name="submit_answer", kwargs={"text": "The dog is named Biscuit."}))
    assert resp.terminated
    result = env.evaluate()
    assert result.success
    assert result.metrics["judge"] == "substring-fallback"


def test_llm_judge_is_invoked_with_the_real_published_template(data_path):
    record = load_longmemeval(data_path)[0]
    seen_prompts = []

    def responder(messages, tools):
        seen_prompts.append(messages[-1].content)
        return AIMessage(content="yes")

    env = LongMemEvalEnv(record, judge_model=ScriptedChatModel(responder=responder))
    env.reset()
    env.step(Action(name="submit_answer", kwargs={"text": "Biscuit"}))
    result = env.evaluate()

    assert result.success
    assert result.metrics["judge"] == "llm"
    assert "Correct Answer: Biscuit" in seen_prompts[0]
    assert "Is the model response correct? Answer yes or no only." in seen_prompts[0]


def test_abstention_question_uses_the_abstention_template(data_path):
    record = load_longmemeval(data_path)[1]
    seen_prompts = []

    def responder(messages, tools):
        seen_prompts.append(messages[-1].content)
        return AIMessage(content="yes")

    env = LongMemEvalEnv(record, judge_model=ScriptedChatModel(responder=responder))
    env.reset()
    env.step(Action(name="submit_answer", kwargs={"text": "I don't have that information."}))
    env.evaluate()
    assert "unanswerable" in seen_prompts[0].lower()


def test_benchmark_splits_train_and_eval_deterministically(data_path):
    bench = LongMemEvalBenchmark(data_path=data_path, eval_fraction=0.5)
    train, eval_ = bench.task_ids("train"), bench.task_ids("eval")
    assert set(train) | set(eval_) == {"q1", "q2_abs"}
    assert not (set(train) & set(eval_))
    assert bench.task_ids("all") == list(bench._by_id)


def test_benchmark_make_env_and_instruction(data_path):
    bench = LongMemEvalBenchmark(data_path=data_path)
    env = bench.make_env("q1")
    assert isinstance(env, LongMemEvalEnv)
    assert bench.instruction_for("q1") == "What did the user name their dog?"
