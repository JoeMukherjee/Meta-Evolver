"""Skill creation, retrieval, evidence-gated lifecycle, versioning, export."""
import json

from meta_evolver.core.types import Action, StepRecord, Trajectory
from meta_evolver.skills import Skill, SkillConfig, SkillLibrary, mine_skills


def traj(task, instruction, success, verbs, bench="textgame", gen=0, skills=()):
    return Trajectory(
        task_id=task,
        benchmark=bench,
        instruction=instruction,
        steps=[
            StepRecord(step_idx=i, action=Action(name="do", kwargs={"text": f"{v} apple {i + 1}"}), observation="ok")
            for i, v in enumerate(verbs)
        ],
        success=success,
        score=float(success),
        generation=gen,
        retrieved_skill_ids=list(skills),
    )


HEAT = ["go", "take", "heat", "put"]


def heat_batch():
    return [
        traj("a", "heat the apple", True, HEAT),
        traj("b", "heat a potato quickly", True, HEAT),
        traj("c", "clean the plate", True, ["go", "clean"]),
        traj("d", "heat the egg", False, ["go", "look"]),
    ]


# -- mining -------------------------------------------------------------------


def test_mining_finds_the_pattern_successes_share_across_distinct_tasks():
    (skill,) = mine_skills(heat_batch(), min_support=2, min_precision=0.6)
    assert skill.signature == "go>take>heat>put"
    assert skill.status == "candidate"
    assert skill.steps == ["go apple <n>", "take apple <n>", "heat apple <n>", "put apple <n>"]
    assert "heat" in skill.when_to_use
    assert set(skill.origin_task_ids) == {"a", "b"}


def test_one_task_repeated_is_not_a_skill():
    batch = [traj("a", "heat apple", True, HEAT, gen=g) for g in range(5)]
    assert mine_skills(batch, min_support=2) == []


def test_a_pattern_that_failures_also_do_is_filtered_by_precision():
    batch = [
        traj("a", "heat apple", True, ["go", "take"]),
        traj("b", "heat egg", True, ["go", "take"]),
        traj("c", "heat pie", False, ["go", "take"]),
        traj("d", "heat tea", False, ["go", "take"]),
        traj("e", "heat jam", False, ["go", "take"]),
    ]
    assert mine_skills(batch, min_support=2, min_precision=0.6) == []


def test_maximal_pattern_subsumes_its_subpatterns():
    skills = mine_skills(heat_batch(), min_support=2, min_precision=0.6)
    assert [s.signature for s in skills] == ["go>take>heat>put"]  # not also go>take, take>heat, ...


def test_infrastructure_errors_are_not_mined():
    batch = heat_batch()
    for t in batch:
        t.error = "rate limited"
    assert mine_skills(batch) == []


# -- library: creation + dedup + versioning -----------------------------------------


def test_induce_registers_candidates_and_rediscovery_does_not_duplicate():
    lib = SkillLibrary()
    first = lib.induce(heat_batch(), generation=0)
    again = lib.induce(heat_batch(), generation=1)
    assert len(first) == 1 and again == []
    assert lib.stats() == {"candidate": 1, "active": 0, "deprecated": 0}


def test_new_procedure_over_same_tools_becomes_a_child_version():
    lib = SkillLibrary(config=SkillConfig(promote_min_uses=1, promote_min_win_tasks=1, promote_min_lcb=0.0))
    (parent,) = lib.induce(heat_batch())
    lib.credit([([parent.id], True, "d", "a")], generation=1)
    lib.lifecycle(generation=1)
    assert parent.status == "active"

    variant = Skill(name="heat-v2", signature="take>go>heat>put", tools=["go", "take", "heat", "put"],
                    benchmark="textgame", when_to_use=["heat"], steps=["x"])
    child, is_new = lib.add_candidate(variant, generation=2)
    assert is_new and child.parent_id == parent.id and child.version == 2


# -- retrieval ----------------------------------------------------------------


def test_candidates_are_offered_only_in_trial_slots():
    lib = SkillLibrary()
    lib.induce(heat_batch())
    got = lib.retrieve("heat the potato for dinner", k=3)
    assert len(got) == 1 and got[0].status == "candidate"  # one trial slot, nothing proven yet


def test_irrelevant_query_returns_nothing():
    lib = SkillLibrary()
    lib.induce(heat_batch())
    assert lib.retrieve("translate this sentence into french", k=3) == []


def test_deprecated_skills_are_never_retrieved():
    lib = SkillLibrary()
    (s,) = lib.induce(heat_batch())
    s.status = "deprecated"
    assert lib.retrieve("heat the potato") == []


def test_proven_skill_outranks_untested_candidate_and_candidate_still_gets_a_slot():
    lib = SkillLibrary(config=SkillConfig(trial_slots=1))
    proven = Skill(name="heat-proven", signature="a>b", tools=["a", "b"], when_to_use=["heat", "potato"],
                   steps=["s"], benchmark="textgame", status="active", uses=10, wins=9, win_tasks=["x", "y"])
    cand = Skill(name="heat-cand", signature="c>d", tools=["c", "d"], when_to_use=["heat", "potato"],
                 steps=["s"], benchmark="textgame")
    lib.skills = {proven.id: proven, cand.id: cand}
    got = lib.retrieve("heat the potato", k=2)
    assert [s.name for s in got] == ["heat-proven", "heat-cand"]


# -- lifecycle gates -----------------------------------------------------------


def _lib_with(skill, **cfg):
    lib = SkillLibrary(config=SkillConfig(**cfg))
    lib.skills[skill.id] = skill
    return lib


def _cand(**kw):
    return Skill(name="s", signature="a>b", tools=["a", "b"], steps=["x"], benchmark="t", **kw)


def test_promotion_requires_wins_on_distinct_tasks():
    one_task = _cand(uses=5, wins=5, win_tasks=["only"])
    lib = _lib_with(one_task)
    assert lib.lifecycle(1)["promoted"] == 0 and one_task.status == "candidate"

    two_tasks = _cand(uses=5, wins=5, win_tasks=["t1", "t2"])
    lib = _lib_with(two_tasks)
    assert lib.lifecycle(1)["promoted"] == 1 and two_tasks.status == "active"
    assert two_tasks.promoted_generation == 1


def test_a_lucky_start_does_not_promote():
    s = _cand(uses=1, wins=1, win_tasks=["t1"])
    lib = _lib_with(s)
    lib.lifecycle(1)
    assert s.status == "candidate"


def test_deprecation_needs_enough_evidence_not_one_bad_episode():
    s = _cand(uses=1, wins=0)
    _lib_with(s).lifecycle(1)
    assert s.status == "candidate"  # one failure is not enough
    bad = _cand(uses=12, wins=0)
    _lib_with(bad).lifecycle(1)
    assert bad.status == "deprecated"


def test_active_skill_that_stops_working_is_deprecated():
    s = _cand(status="active", uses=14, wins=1, win_tasks=["t1", "t2"])
    _lib_with(s).lifecycle(5)
    assert s.status == "deprecated"


def test_stale_untested_candidates_expire():
    s = _cand()
    lib = _lib_with(s, stale_after=3)
    lib.lifecycle(2)
    assert s.status == "candidate"
    lib.lifecycle(3)
    assert s.status == "deprecated"


def test_promoting_a_child_supersedes_its_parent():
    parent = _cand(status="active", uses=10, wins=8, win_tasks=["t1", "t2"])
    child = Skill(name="c", signature="b>a", tools=["a", "b"], steps=["x"], benchmark="t",
                  parent_id=parent.id, version=2, uses=5, wins=5, win_tasks=["t1", "t2"])
    lib = SkillLibrary()
    lib.skills = {parent.id: parent, child.id: child}
    out = lib.lifecycle(7)
    assert child.status == "active" and parent.status == "deprecated" and out["superseded"] == 1


def test_duplicate_actives_merge_and_pool_evidence():
    a = _cand(status="active", uses=6, wins=5, win_tasks=["x"])
    b = Skill(name="s2", signature="a>b", tools=["a", "b"], steps=["x"], benchmark="t",
              status="active", uses=4, wins=3, win_tasks=["y"], version=2)
    lib = SkillLibrary()
    lib.skills = {a.id: a, b.id: b}
    assert lib.merge_duplicates() == 1
    assert (a.uses, a.wins, sorted(a.win_tasks)) == (10, 8, ["x", "y"]) and b.status == "deprecated"


def test_max_active_evicts_the_weakest():
    lib = SkillLibrary(config=SkillConfig(max_active=1))
    strong = Skill(name="s1", signature="a>b", tools=["a", "b"], steps=["x"], benchmark="t",
                   status="active", uses=20, wins=19, win_tasks=["1", "2"])
    weak = Skill(name="s2", signature="c>d", tools=["c", "d"], steps=["x"], benchmark="t",
                 status="active", uses=4, wins=3, win_tasks=["1", "2"])
    lib.skills = {strong.id: strong, weak.id: weak}
    lib.lifecycle(1)
    assert strong.status == "active" and weak.status == "deprecated"


# -- credit, persistence, export -----------------------------------------------------


def test_credit_updates_evidence_and_domain_stats():
    lib = SkillLibrary()
    (s,) = lib.induce(heat_batch())
    n = lib.credit([([s.id, "unknown-id"], True, "kitchen", "t1"), ([s.id], False, "kitchen", "t2")], generation=3)
    assert n == 2 and (s.uses, s.wins) == (2, 1) and s.win_tasks == ["t1"]
    assert s.domain_stats["kitchen"] == [1.0, 2.0] and s.last_used_generation == 3


def test_library_round_trips_through_disk(tmp_path):
    p = tmp_path / "skills.json"
    lib = SkillLibrary(p)
    (s,) = lib.induce(heat_batch())
    lib.credit([([s.id], True, None, "t1")])
    lib.lifecycle(1)
    again = SkillLibrary(p)
    assert again.skills[s.id].uses == 1 and again.skills[s.id].signature == s.signature


def test_export_writes_agent_skills_layout(tmp_path):
    lib = SkillLibrary()
    (s,) = lib.induce(heat_batch())
    s.status = "active"
    (path,) = lib.export_markdown(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith(f"---\nname: {s.name}\ndescription: ")
    assert "## Steps" in text and "1. go apple <n>" in text
    assert lib.export_markdown(tmp_path / "none", statuses=("deprecated",)) == []


def test_skill_renders_as_a_memory_item_for_the_existing_prompt_layer():
    (s,) = mine_skills(heat_batch())
    item = s.as_memory_item()
    assert item.scenario == "skill" and "go apple <n>" in item.procedure
    assert "STRATEGY" in item.render(1)


def test_refiner_failure_never_breaks_induction():
    def boom(skill, trajs):
        raise RuntimeError("model down")

    lib = SkillLibrary(refiner=boom)
    assert len(lib.induce(heat_batch())) == 1


def test_json_serialisable_state():
    lib = SkillLibrary()
    lib.induce(heat_batch())
    json.dumps([s.model_dump() for s in lib.skills.values()])


# -- audit ledger, rollback, dry-run, retirement pointers (the Hermes-inspired safety rails) --


def _promotable():
    return _cand(uses=5, wins=5, win_tasks=["t1", "t2"])


def test_every_change_is_one_ledger_row_and_can_be_rolled_back(tmp_path):
    lib = SkillLibrary(tmp_path / "skills.json")
    s = _promotable()
    lib.skills[s.id] = s
    lib._audit("create", None, s, 0, actor="induce")
    lib.lifecycle(4)
    assert s.status == "active"
    ops = [r["op"] for r in lib.history(s.id)]
    assert ops == ["create", "promote"]

    assert lib.rollback(s.id, generation=5) is True
    assert lib.skills[s.id].status == "candidate"  # restored from the recorded 'before'
    assert lib.history(s.id)[-1]["op"] == "rollback"  # and the undo is itself on the record


def test_rolling_back_a_creation_retires_the_skill_and_nothing_to_undo_is_false(tmp_path):
    lib = SkillLibrary(tmp_path / "s.json")
    (s,) = lib.induce(heat_batch())
    assert lib.rollback(s.id) is True and lib.skills[s.id].status == "deprecated"
    assert lib.rollback("sk-unknown") is False


def test_preview_changes_nothing_on_disk_or_in_the_ledger(tmp_path):
    lib = SkillLibrary(tmp_path / "s.json")
    s = _promotable()
    lib.skills[s.id] = s
    plan = lib.preview_lifecycle(4)
    assert plan["counts"]["promoted"] == 1 and plan["changes"] == {s.id: ("candidate", "active")}
    assert s.status == "candidate" and lib.history() == [] and not (tmp_path / "s.json").exists()


def test_retired_skills_point_at_what_replaced_them():
    parent = _cand(status="active", uses=10, wins=8, win_tasks=["t1", "t2"])
    child = Skill(name="c", signature="b>a", tools=["a", "b"], steps=["x"], benchmark="t",
                  parent_id=parent.id, version=2, uses=5, wins=5, win_tasks=["t1", "t2"])
    lib = SkillLibrary()
    lib.skills = {parent.id: parent, child.id: child}
    lib.lifecycle(7)
    assert parent.superseded_by == child.id

    a = _cand(status="active", uses=6, wins=5, win_tasks=["x"])
    b = Skill(name="s2", signature="a>b", tools=["a", "b"], steps=["x"], benchmark="t",
              status="active", uses=4, wins=3, win_tasks=["y"], version=2)
    lib2 = SkillLibrary()
    lib2.skills = {a.id: a, b.id: b}
    lib2.merge_duplicates()
    assert b.merged_into == a.id


def test_exported_description_is_trigger_first_and_bounded(tmp_path):
    lib = SkillLibrary()
    (s,) = lib.induce(heat_batch())
    s.status = "active"
    (path,) = lib.export_markdown(tmp_path)
    desc = next(l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("description: "))
    assert desc.startswith("description: Use when the task involves: heat") and len(desc) <= 252
