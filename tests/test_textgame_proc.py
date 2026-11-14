"""The procedural OOD text game: determinism, solvability, and that the shift is real."""
from meta_evolver.benchmarks.textgame_proc import ProcTextGameBenchmark, generate_layouts
from meta_evolver.core.registry import get_benchmark
from meta_evolver.core.types import Action
from meta_evolver.eval.ood_recall import shift_score
from meta_evolver.knowledge import extract_concepts


def oracle(env, layout) -> bool:
    """Solve a task with perfect knowledge, to prove every generated task is solvable."""
    obj, loc, goal = layout["object"], layout["object_location"], layout["goal_receptacle"]
    steps = [f"go to {loc}"]
    if loc.split(" ")[0] in {"drawer", "cabinet", "dishwasher", "garbagecan", "laundryhamper", "fridge", "microwave", "safe", "toolbox"}:
        steps.append(f"open {loc}")
    steps.append(f"take {obj} from {loc}")
    if layout["requires_clean"]:
        steps.append(f"clean {obj}")
    steps += [f"go to {goal}", f"put {obj} in/on {goal}"]
    env.reset()
    for s in steps:
        env.step(Action(name="do", kwargs={"text": s}))
    return env.evaluate().success


def test_registered_and_deterministic_for_a_seed():
    a = get_benchmark("textgame_proc", seed=3)
    b = ProcTextGameBenchmark(seed=3)
    c = ProcTextGameBenchmark(seed=4)
    assert a.layouts == b.layouts and a.layouts != c.layouts


def test_sizes_and_disjoint_splits():
    b = ProcTextGameBenchmark(n_train=24, n_eval=16)
    tr, ev = b.task_ids("train"), b.task_ids("eval")
    assert (len(tr), len(ev)) == (24, 16) and not set(tr) & set(ev) and len(set(b.task_ids("all"))) == 40


def test_every_generated_task_is_solvable():
    b = ProcTextGameBenchmark(n_train=24, n_eval=24, seed=1)
    unsolved = [t for t in b.task_ids("all") if not oracle(b.make_env(t), b.layouts[t])]
    assert unsolved == []


def test_trap_tasks_hide_the_object_in_a_closed_container():
    b = ProcTextGameBenchmark(n_train=4, n_eval=24, seed=2)
    traps = [lay for lay in b.layouts.values() if "trap" in lay["shift"] and lay["split"] == "eval"]
    assert traps and all(
        lay["object_location"].split(" ")[0] in {"garbagecan", "laundryhamper", "safe", "fridge", "toolbox"}
        for lay in traps
    )


def test_eval_is_measurably_shifted_from_train_in_wording():
    """The shift axis the lifting layer targets must exist: lexical-shift eval
    instructions share fewer concepts with the training instructions than a
    training instruction shares with its own siblings."""
    b = ProcTextGameBenchmark(n_train=40, n_eval=40, seed=0)
    train_instr = [b.instruction_for(t) for t in b.task_ids("train")]
    lex = [b.instruction_for(t) for t in b.task_ids("eval") if b.layouts[t]["shift"] == ["lexical"]]
    assert lex
    shifted = sum(shift_score(i, train_instr) for i in lex) / len(lex)
    within = sum(shift_score(t, [x for x in train_instr if x != t]) for t in train_instr) / len(train_instr)
    assert shifted > within
    # and the synonym vocabulary really is different from training's
    train_vocab = {c for i in train_instr for c in extract_concepts(i)}
    assert {"wash", "rinse", "scrub", "bring", "relocate"} & {c for i in lex for c in extract_concepts(i)}
    assert not {"wash", "rinse", "scrub", "relocate"} & train_vocab


def test_specs_tag_each_eval_task_with_its_shift_axes():
    b = ProcTextGameBenchmark(n_train=4, n_eval=8, seed=0)
    tags = {t for s in b.specs() if s.split == "eval" for t in s.tags}
    assert {"lexical", "domain", "trap"} <= tags


def test_the_original_textgame_is_unchanged():
    from meta_evolver.benchmarks.textworld import LAYOUTS, TextGameEnv

    env = TextGameEnv(task_id="kitchen_knife_id")
    assert env.layouts is LAYOUTS and env.reset().info["instruction"] == "put a clean knife in drawer 1"
