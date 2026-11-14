"""Learning *when* a skill works: negative and positive preconditions from outcomes."""
from meta_evolver.skills import Skill, SkillLibrary


def _skill(**kw):
    return Skill(name="s", signature="a>b", tools=["a", "b"], steps=["x"], benchmark="t",
                 when_to_use=["deploy"], status="active", **kw)


def _lib_with_outcomes(outcomes):
    lib = SkillLibrary()
    s = _skill()
    lib.skills[s.id] = s
    lib.credit([([s.id], ok, "t", f"task{i}", concepts) for i, (concepts, ok) in enumerate(outcomes)])
    return lib, s


WIN = ["deploy", "service", "restart"]
FAIL = ["deploy", "database", "migration"]


def test_failures_sharing_a_concept_the_wins_lack_become_a_negative_precondition():
    lib, s = _lib_with_outcomes([(WIN, True)] * 4 + [(FAIL, False)] * 3)
    (change,) = lib.refine(generation=5)
    assert "database" in change["avoid_added"] and "migration" in change["avoid_added"]
    assert "deploy" not in change["avoid_added"]  # present in the wins too: not discriminating
    assert set(s.avoid_when) >= {"database", "migration"}


def test_a_skill_with_an_avoid_concept_is_no_longer_offered_for_that_task():
    lib, s = _lib_with_outcomes([(WIN, True)] * 4 + [(FAIL, False)] * 3)
    assert [x.id for x in lib.retrieve("deploy the service now", k=2)] == [s.id]
    lib.refine(generation=1)
    assert lib.retrieve("deploy the database migration", k=2) == []  # vetoed
    assert [x.id for x in lib.retrieve("deploy the service now", k=2)] == [s.id]  # still fine elsewhere


def test_concepts_shared_by_wins_and_absent_from_failures_become_positive_preconditions():
    lib, s = _lib_with_outcomes([(WIN, True)] * 4 + [(FAIL, False)] * 3)
    lib.refine(generation=1)
    assert {"service", "restart"} <= set(s.when_to_use)


def test_one_bad_episode_or_thin_evidence_teaches_nothing():
    lib, s = _lib_with_outcomes([(WIN, True)] * 6 + [(FAIL, False)])
    assert lib.refine() == [] and s.avoid_when == []
    lib2, s2 = _lib_with_outcomes([(WIN, True), (FAIL, False), (FAIL, False)])
    assert lib2.refine(min_outcomes=5) == []  # fewer than min_outcomes episodes


def test_refine_is_idempotent_audited_and_reversible(tmp_path):
    lib = SkillLibrary(tmp_path / "s.json")
    s = _skill()
    lib.skills[s.id] = s
    lib.credit([([s.id], ok, "t", f"k{i}", c) for i, (c, ok) in enumerate([(WIN, True)] * 4 + [(FAIL, False)] * 3)])
    assert len(lib.refine(2)) == 1
    assert lib.refine(3) == []  # already matches its evidence
    ops = [r["op"] for r in lib.history(s.id)]
    assert "patch" in ops and lib.history(s.id)[-1]["actor"] == "refiner"
    assert lib.rollback(s.id)  # the refinement can be undone


def test_locked_skills_are_never_refined():
    lib, s = _lib_with_outcomes([(WIN, True)] * 4 + [(FAIL, False)] * 3)
    s.locked = True
    other = _skill()
    other.name, other.signature = "o", "c>d"
    other.id = ""
    other.model_post_init(None)
    lib.skills[other.id] = other
    lib.credit([([other.id], ok, "t", f"o{i}", c) for i, (c, ok) in enumerate([(WIN, True)] * 4 + [(FAIL, False)] * 3)])
    changes = lib.refine()  # must not raise, and must still refine the unlocked skill
    assert s.avoid_when == [] and [c["skill"] for c in changes] == [other.id]


def test_outcomes_are_capped_and_survive_a_save_load(tmp_path):
    lib = SkillLibrary(tmp_path / "s.json")
    s = _skill()
    lib.skills[s.id] = s
    lib.credit([([s.id], True, None, f"t{i}", ["deploy"]) for i in range(80)])
    assert len(s.outcomes) == 60
    lib.save()
    again = SkillLibrary(tmp_path / "s.json")
    assert len(again.skills[s.id].outcomes) == 60 and again.skills[s.id].outcomes[0][1] is True


def test_four_element_credit_tuples_still_work():
    lib = SkillLibrary()
    s = _skill()
    lib.skills[s.id] = s
    assert lib.credit([([s.id], True, "d", "t1")]) == 1 and s.outcomes == []
