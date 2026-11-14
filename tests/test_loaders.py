import json

from meta_evolver.eval.loaders import (
    discover_runs,
    load_corpus_trajectories,
    load_eval_citations,
    load_memory_bank,
    task_key,
)


def _step(text, goal="Your task is to: clean some cup and put it in shelf."):
    obs = {"text": "Task: " + goal, "data": {"goal_text": goal}}
    return {
        "raw_action": {"name": "do", "kwargs": {"text": text}},
        "filtered_action": {"name": "do", "kwargs": {"text": text}},
        "raw_observation": obs,
        "filtered_observation": obs,
        "filtered_reward": 0.0,
        "terminated": False,
    }


def _write(path, rows, raw_extra=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in rows] + list(raw_extra)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_corpus_loader(tmp_path):
    p = tmp_path / "traces.jsonl"
    rows = [
        {"task_id": "const", "steps": [_step("go to a"), _step("take b")], "success": True,
         "final_reward": 1.0, "error": None, "rollout_idx": 2},
        {"task_id": "const", "steps": [], "success": False, "error": "env.reset raised"},
        {"task_id": "const", "steps": [{"junk": 1}], "success": False},
    ]
    _write(p, rows, raw_extra=["not json", "[1,2]"])
    trajs, stats = load_corpus_trajectories(p)
    assert stats["bad_rows"] == 2 and stats["loaded"] == 3
    t = trajs[0]
    assert t.usable and t.success and t.rollout_index == 2
    assert t.instruction.startswith("Your task is to") and t.task_id == task_key(t.instruction)
    assert t.steps[0].action.render() == "go to a"
    assert t.retrieved_memory_ids == []
    assert not trajs[1].usable and trajs[1].error
    assert not trajs[2].steps  # malformed step dropped


def test_memory_bank_loader(tmp_path):
    p = tmp_path / "ours_full.jsonl"
    q = "Your task is to: clean some cup and put it in shelf."
    rows = [
        {"title": "A", "description": "d", "content": "c", "embedding": [0.1],
         "source": {"task_query": q}},
        {"title": "A", "content": "dup"},
        {"content": "no title"},
    ]
    _write(p, rows)
    texts, meta, stats = load_memory_bank(p, condition="ours")
    assert list(texts) == ["ours/A"] and "c" in texts["ours/A"]
    assert meta["ours/A"]["task_key"] == task_key(q)
    assert stats["duplicate_ids"] == 1 and stats["bad_rows"] == 1


def test_eval_citations_and_discovery(tmp_path):
    run = tmp_path / "alfworld_headline_x"
    _write(run / "eval" / "round1" / "ours_eval_in_distribution.jsonl",
           [{"seed": 0, "split": "s", "success": True, "retrieved_titles": ["A", 3]},
            {"seed": 1, "success": True}])
    rows, stats = load_eval_citations(run / "eval" / "round1" / "ours_eval_in_distribution.jsonl")
    assert len(rows) == 1 and rows[0]["retrieved_titles"] == ["A"] and stats["bad_rows"] == 1
    assert discover_runs(tmp_path) == [run]
    assert discover_runs(tmp_path / "missing") == []
    assert load_eval_citations(tmp_path / "nope.jsonl")[0] == []
