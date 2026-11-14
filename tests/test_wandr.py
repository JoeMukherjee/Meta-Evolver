"""WANDR adapter: loading, split stability, workspace tools, proxy scoring.

All offline -- the dataset rows are a tiny fixture, nothing is downloaded.
"""
import json

import pytest

from meta_evolver.benchmarks.wandr import (
    WandrBenchmark,
    WandrEnv,
    WandrTask,
    check_file,
    load_wandr,
)
from meta_evolver.core.registry import get_benchmark
from meta_evolver.core.types import Action

ROWS = [
    {
        "task_id": f"task-{i:02d}",
        "instruction": f"Find things #{i}",
        "required_output_files": ["answer.json", "evidence.csv"],
        "scored": i != 3,
    }
    for i in range(20)
]


@pytest.fixture()
def data_path(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n", encoding="utf-8")
    return p


def _act(env, name, **kw):
    return env.step(Action(name=name, kwargs=kw))


def test_load_skips_unscored_by_default(data_path):
    assert len(load_wandr(data_path)) == 19
    assert len(load_wandr(data_path, scored_only=False)) == 20


def test_registered_and_constructible(data_path):
    bench = get_benchmark("wandr", data_path=data_path)
    assert isinstance(bench, WandrBenchmark)
    assert len(bench.task_ids("all")) == 19


def test_split_is_disjoint_stable_and_covers_all(data_path):
    a = WandrBenchmark(data_path=data_path)
    b = WandrBenchmark(data_path=data_path, limit=None)
    train, ev = set(a.task_ids("train")), set(a.task_ids("eval"))
    assert not train & ev
    assert train | ev == set(a.task_ids("all"))
    assert ev == set(b.task_ids("eval"))  # depends on ids, not on order/instance


def test_perfect_workspace_scores_one(data_path):
    env = WandrBenchmark(data_path=data_path).make_env("task-00")
    env.reset()
    _act(env, "write_file", path="answer.json", content='{"a": 1}')
    _act(env, "write_file", path="evidence.csv", content="url,claim\nhttps://x,y\n")
    assert _act(env, "finish").terminated
    result = env.evaluate()
    assert result.success and result.score == 1.0
    assert result.metrics["proxy"] is True


def test_partial_and_malformed_outputs_are_penalised(data_path):
    env = WandrBenchmark(data_path=data_path).make_env("task-00")
    env.reset()
    _act(env, "write_file", path="answer.json", content="{not json")
    result = env.evaluate()
    assert not result.success and result.score == 0.0
    assert set(result.metrics["problems"]) == {"answer.json", "evidence.csv"}

    _act(env, "write_file", path="answer.json", content="{}")
    assert env.evaluate().score == 0.5


def test_paths_cannot_escape_workspace(data_path):
    env = WandrBenchmark(data_path=data_path).make_env("task-00")
    env.reset()
    for bad in ("../x.json", "a/../../x.json", ""):
        _act(env, "write_file", path=bad, content="{}")
    assert env.files == {}
    _act(env, "write_file", path="/answer.json", content="{}")  # leading slash is normalised
    assert "answer.json" in env.files


def test_no_required_files_never_succeeds():
    env = WandrEnv(WandrTask({"task_id": "t", "instruction": "i", "required_output_files": []}))
    env.reset()
    assert not env.evaluate().success


def test_export_workspace_roundtrip(data_path, tmp_path):
    env = WandrBenchmark(data_path=data_path).make_env("task-00")
    env.reset()
    _act(env, "write_file", path="out/answer.json", content='{"k": 2}')
    root = env.export_workspace(tmp_path / "export")
    assert json.loads((root / "out" / "answer.json").read_text()) == {"k": 2}


@pytest.mark.parametrize(
    "path,content,ok",
    [
        ("a.json", '{"a": 1}', True),
        ("a.jsonl", '{"a": 1}\n{"b": 2}\n', True),
        ("a.jsonl", '{"a": 1}\nnot-json\n', False),
        ("a.csv", "h\nr\n", True),
        ("a.csv", "only-header\n", False),
        ("a.txt", "anything", True),
        ("a.txt", "   \n", False),
    ],
)
def test_check_file(path, content, ok):
    assert check_file(path, content)[0] is ok
