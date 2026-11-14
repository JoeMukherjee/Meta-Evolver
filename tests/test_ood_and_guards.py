"""OOD recall eval, mining quality gate, guarded skill edits, WANDR prompt."""
import json

import pytest

from meta_evolver.core.types import Action, StepRecord, Trajectory
from meta_evolver.eval.ood_recall import TfidfRetriever, evaluate_ood_recall, shift_score
from meta_evolver.skills import SkillLibrary, mine_skills
from meta_evolver.skills.distill import error_fraction, is_clean_success


def traj(task, instr, ok, verbs=("go", "take"), obs=None, mem=(), bench="b"):
    obs = obs or ["ok"] * len(verbs)
    return Trajectory(
        task_id=task, benchmark=bench, instruction=instr, success=ok, score=float(ok),
        steps=[StepRecord(step_idx=i, action=Action(name="do", kwargs={"text": f"{v} x {i}"}), observation=o)
               for i, (v, o) in enumerate(zip(verbs, obs))],
        retrieved_memory_ids=list(mem),
    )


# -- OOD recall eval ------------------------------------------------------------


def _corpus():
    memory_texts = {
        "m-heat": "strategy for warming food: use the microwave then serve",
        "m-clean": "strategy for washing dishes: use the sink and soap",
        "m-noise1": "generic advice about being careful and checking results",
        "m-noise2": "always read the observation before acting on it",
    }
    train = [
        traj("t1", "heat the potato", True, mem=["m-heat"]),
        traj("t2", "heat the soup bowl", True, mem=["m-heat"]),
        traj("t3", "clean the mug", True, mem=["m-clean"]),
        traj("t4", "clean the plate", True, mem=["m-clean"]),
    ]
    return memory_texts, train


def test_shift_score_orders_reworded_below_unrelated():
    train = ["heat the potato", "clean the mug"]
    assert shift_score("heat the potato", train) == 0.0
    assert 0.0 < shift_score("heat some soup", train) <= 1.0
    assert shift_score("translate french poem", train) == 1.0


def test_graph_wins_where_similarity_has_no_text_to_match():
    """The test task says 'heat' but the memory text says 'warming' and 'microwave'
    -- text similarity over memory bodies finds nothing, the graph's learned
    concept->memory evidence does."""
    texts, train = _corpus()
    test = [traj("x1", "heat it up", True, mem=["m-heat"]), traj("x2", "heat dinner fast", True, mem=["m-heat"])]
    out = evaluate_ood_recall(train, test, texts, k=2)
    assert out["n"] == 2
    assert out["recall"]["graph"] == 1.0 and out["recall"]["similarity"] == 0.0
    assert out["recall"]["fused"] == 1.0  # fusion keeps the graph's win


def test_similarity_wins_when_wording_matches_memory_text_and_fusion_keeps_both():
    texts, train = _corpus()
    texts["m-sink"] = "polish silverware with a cloth"
    test = [traj("y1", "washing dishes in the sink", True, mem=["m-clean"])]
    out = evaluate_ood_recall(train, test, texts, k=2)
    assert out["recall"]["similarity"] == 1.0 and out["recall"]["fused"] == 1.0


def test_unreachable_gold_is_not_counted_and_failures_are_skipped():
    texts, train = _corpus()
    test = [traj("z1", "heat it", True, mem=["never-seen"]), traj("z2", "heat it", False, mem=["m-heat"])]
    assert evaluate_ood_recall(train, test, texts)["n"] == 0


def test_buckets_report_per_shift_level():
    texts, train = _corpus()
    test = [traj("a", "heat the potato", True, mem=["m-heat"]), traj("b", "warm some stew: heat", True, mem=["m-heat"])]
    out = evaluate_ood_recall(train, test, texts, k=2)
    assert sum(b["n"] for b in out["by_shift"].values()) == out["n"]
    json.dumps(out)


def test_tfidf_retriever_ranks_by_overlap():
    r = TfidfRetriever({"a": "heat the soup in a pan", "b": "wash the car"})
    assert r.top("heat soup", 1) == ["a"] and r.top("nothing matches zzz", 3) == []


# -- mining quality gate ------------------------------------------------------------


def test_flailing_success_is_not_a_clean_procedure():
    flail = traj("a", "heat x", True, ("go", "take", "heat"), ["error: cannot", "nothing happens", "ok"])
    assert error_fraction(flail) > 0.34 and not is_clean_success(flail)
    assert is_clean_success(traj("b", "heat y", True, ("go", "take", "heat")))
    assert not is_clean_success(traj("c", "heat z", True, ("go", "take"), ["ok", "invalid action"]))  # ends in an error


def test_skills_are_not_mined_from_flailing_successes():
    verbs = ("go", "take", "heat")
    messy = ["error: cannot", "invalid", "ok"]
    batch = [traj("a", "heat apple", True, verbs, messy), traj("b", "heat egg", True, verbs, messy)]
    assert mine_skills(batch, min_support=2) == []
    clean = [traj("a", "heat apple", True, verbs), traj("b", "heat egg", True, verbs)]
    assert len(mine_skills(clean, min_support=2)) == 1


def test_triggers_never_contain_digits():
    verbs = ("go", "take")
    batch = [traj("a", "heat order42 now", True, verbs), traj("b", "heat order42 later", True, verbs)]
    (s,) = mine_skills(batch, min_support=2)
    assert all(not any(ch.isdigit() for ch in t) for t in s.when_to_use) and "heat" in s.when_to_use


# -- guarded edits -------------------------------------------------------------------


def _lib():
    lib = SkillLibrary()
    (s,) = lib.induce([traj("a", "heat apple", True, ("go", "take", "heat")), traj("b", "heat egg", True, ("go", "take", "heat"))])
    return lib, s


def test_patch_requires_a_prior_view_by_the_same_actor():
    lib, s = _lib()
    with pytest.raises(PermissionError, match="must view"):
        lib.patch(s.id, actor="reviewer", description="better")
    lib.view(s.id, actor="someone-else")
    with pytest.raises(PermissionError):
        lib.patch(s.id, actor="reviewer", description="better")
    lib.view(s.id, actor="reviewer")
    lib.patch(s.id, actor="reviewer", description="better")
    assert lib.skills[s.id].description == "better" and lib.skills[s.id].patch_count == 1


def test_locked_skill_refuses_autonomous_patch_and_unknown_skill_errors():
    lib, s = _lib()
    s.locked = True
    lib.view(s.id, actor="a")
    with pytest.raises(PermissionError, match="locked"):
        lib.patch(s.id, actor="a", description="x")
    with pytest.raises(KeyError):
        lib.patch("sk-nope", actor="a")


def test_patch_is_audited_rollbackable_and_tracks_reuse_after_patch(tmp_path):
    lib = SkillLibrary(tmp_path / "s.json")
    (s,) = lib.induce([traj("a", "heat apple", True, ("go", "take", "heat")), traj("b", "heat egg", True, ("go", "take", "heat"))])
    lib.view(s.id, "r")
    lib.patch(s.id, actor="r", steps=["new step"])
    assert lib.history(s.id)[-1]["op"] == "patch" and not s.reused_since_patch
    lib.credit([([s.id], True, None, "t1")])
    assert s.reused_since_patch  # the patched version was actually run
    old_steps = lib.history(s.id)[-1]["before"]["steps"]
    assert lib.rollback(s.id) and lib.skills[s.id].steps == old_steps


# -- WANDR prompt ---------------------------------------------------------------------


def test_wandr_system_prompt_states_the_file_contract(tmp_path):
    from meta_evolver.benchmarks.wandr import WandrBenchmark

    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"task_id": "t", "instruction": "i", "required_output_files": ["a.json"]}) + "\n")
    prompt = WandrBenchmark(data_path=p).system_prompt()
    assert "write_file" in prompt and "source URL" in prompt and len(prompt) < 1800
