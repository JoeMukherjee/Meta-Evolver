"""LoCoMoEnv / LoCoMoBenchmark -- schema- and metric-faithful, no network."""
from __future__ import annotations

import json

import pytest

from meta_evolver.benchmarks.locomo import (
    LoCoMoBenchmark,
    LoCoMoEnv,
    f1_score,
    load_locomo,
    multi_answer_f1,
    normalize_answer,
    score_locomo_answer,
)
from meta_evolver.core.types import Action

_RAW = [
    {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1": [
                {"speaker": "Caroline", "dia_id": "D1:1", "text": "I went to the LGBTQ support group on 7 May."},
                {"speaker": "Melanie", "dia_id": "D1:2", "text": "That's great to hear!"},
            ],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_2": [
                {"speaker": "Caroline", "dia_id": "D2:1", "text": "The weather has been nice lately."},
            ],
            "session_2_date_time": "9 May, 2023",
        },
        "qa": [
            {"question": "When did Caroline go to the support group?", "answer": "7 May", "category": 2, "evidence": ["D1:1"]},
            {"question": "Who is Caroline's dog?", "answer": "no information available", "category": 5, "evidence": []},
            {"question": "What are Caroline's hobbies?", "answer": "hiking, painting", "category": 1, "evidence": []},
            # Real-schema category-5 items use ``adversarial_answer``, not
            # ``answer`` -- confirmed against the official locomo10.json.
            {"question": "What is Caroline's blood type?", "adversarial_answer": "not mentioned", "category": 5, "evidence": []},
        ],
    },
    {
        "sample_id": "conv-2",
        "conversation": {
            "speaker_a": "Alex",
            "speaker_b": "Sam",
            "session_1": [{"speaker": "Alex", "dia_id": "D1:1", "text": "I love pizza."}],
            "session_1_date_time": "1 Jan, 2024",
        },
        "qa": [
            {"question": "What food does Alex love?", "answer": "pizza", "category": 4, "evidence": ["D1:1"]},
        ],
    },
]


@pytest.fixture
def data_path(tmp_path):
    path = tmp_path / "locomo_mini.json"
    path.write_text(json.dumps(_RAW), encoding="utf-8")
    return path


# -- scoring, checked against the official repo's algorithm directly --------


def test_normalize_answer_strips_articles_punctuation_and_case():
    assert normalize_answer("The Dog, and a Cat!") == "dog cat"


def test_f1_score_perfect_match_is_one():
    assert f1_score("7 May 2023", "7 May 2023") == 1.0


def test_f1_score_partial_overlap_is_between_zero_and_one():
    score = f1_score("7 May", "7 May 2023")
    assert 0.0 < score < 1.0


def test_multi_hop_f1_averages_over_comma_split_sub_answers():
    score = multi_answer_f1("hiking, painting", "hiking, painting")
    assert score == 1.0


def test_category_3_open_domain_truncates_at_semicolon():
    # ground truth "Paris; the capital of France" -> only "Paris" is scored against
    score = score_locomo_answer(3, "Paris", "Paris; the capital of France")
    assert score == 1.0


def test_category_5_adversarial_scores_refusal_as_correct():
    assert score_locomo_answer(5, "There is no information available about that.", "n/a") == 1.0
    assert score_locomo_answer(5, "Her dog is named Rex.", "n/a") == 0.0


# -- loading ------------------------------------------------------------------


def test_load_parses_sessions_in_numeric_order(data_path):
    convs = load_locomo(data_path)
    assert len(convs) == 2
    assert [s["id"] for s in convs[0].sessions] == ["session_1", "session_2"]
    assert "7 May" in convs[0].sessions[0]["text"]


def test_load_captures_qa_and_category(data_path):
    convs = load_locomo(data_path)
    assert len(convs[0].qa) == 4
    assert convs[0].qa[1]["category"] == 5


def test_adversarial_answer_key_is_used_when_answer_is_absent(data_path):
    from meta_evolver.benchmarks.locomo import LoCoMoTaskRef

    convs = load_locomo(data_path)
    ref = LoCoMoTaskRef(convs[0], 3)  # the adversarial_answer-only item
    assert ref.answer == "not mentioned"


# -- env ------------------------------------------------------------------


def test_env_reset_previews_sessions(data_path):
    bench = LoCoMoBenchmark(data_path=data_path, eval_fraction=0.0)
    env = bench.make_env("conv-1:qa0")
    reset = env.reset()
    assert "session_1" in reset.observation.text
    assert reset.info["instruction"] == "When did Caroline go to the support group?"


def test_read_and_search_sessions(data_path):
    bench = LoCoMoBenchmark(data_path=data_path, eval_fraction=0.0)
    env = bench.make_env("conv-1:qa0")
    env.reset()
    resp = env.step(Action(name="read_session", kwargs={"session_id": "session_1"}))
    assert "support group" in resp.observation.text
    resp = env.step(Action(name="search_sessions", kwargs={"query": "weather"}))
    assert "session_2" in resp.observation.text


def test_submit_answer_scores_against_the_real_category_metric(data_path):
    bench = LoCoMoBenchmark(data_path=data_path, eval_fraction=0.0)
    env = bench.make_env("conv-1:qa0")  # category 2, answer "7 May"
    env.reset()
    resp = env.step(Action(name="submit_answer", kwargs={"text": "7 May"}))
    assert resp.terminated
    result = env.evaluate()
    assert result.success
    assert result.score == 1.0
    assert result.metrics["category_name"] == "temporal"


def test_adversarial_task_scored_via_env(data_path):
    bench = LoCoMoBenchmark(data_path=data_path, eval_fraction=0.0)
    env = bench.make_env("conv-1:qa1")  # category 5
    env.reset()
    env.step(Action(name="submit_answer", kwargs={"text": "no information available"}))
    result = env.evaluate()
    assert result.success
    assert result.metrics["category_name"] == "adversarial"


def test_no_answer_submitted_fails_cleanly(data_path):
    bench = LoCoMoBenchmark(data_path=data_path, eval_fraction=0.0)
    env = bench.make_env("conv-1:qa0")
    env.reset()
    result = env.evaluate()
    assert not result.success
    assert result.metrics["reason"] == "no answer submitted"


# -- benchmark-level splits --------------------------------------------------


def test_eval_split_holds_out_whole_conversations_not_individual_questions(data_path):
    bench = LoCoMoBenchmark(data_path=data_path, eval_fraction=0.5)
    eval_ids = bench.task_ids("eval")
    train_ids = bench.task_ids("train")
    assert eval_ids and train_ids
    # every eval task_id must come from the same held-out conversation
    eval_convs = {tid.split(":")[0] for tid in eval_ids}
    train_convs = {tid.split(":")[0] for tid in train_ids}
    assert not (eval_convs & train_convs)


def test_task_ids_all_covers_every_qa_pair(data_path):
    bench = LoCoMoBenchmark(data_path=data_path, eval_fraction=0.0)
    assert len(bench.task_ids("all")) == 5  # 4 + 1 qa pairs across two conversations


def test_registered_under_locomo():
    from meta_evolver.core.registry import list_benchmarks

    assert "locomo" in list_benchmarks()
