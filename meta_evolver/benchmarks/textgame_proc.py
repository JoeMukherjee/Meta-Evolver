"""A procedural, out-of-distribution text game -- our own harness test.

The built-in ``textgame`` has five hand-written tasks: enough to test that the
adaptive controller breaks a retrieval loop, far too few to say anything
statistical about transfer. This generates as many tasks as a budget allows,
deterministically from a seed, with the held-out split shifted along the three
axes the knowledge-graph and skill layers claim to help with:

* **Lexical shift** -- the same job in different words ("wash ... and place it
  in" for "put a clean ... in"). Text similarity has little to match; a lifted
  concept link (``clean`` ~ ``wash``) is exactly what should bridge it.
* **Domain shift** -- rooms and objects the agent never trained on (bedroom,
  garage, office), so a memory keyed on kitchen vocabulary is off-target.
* **Prior traps** -- the object hidden in the receptacle a naive prior checks
  last, which is what makes a confidently wrong retrieved strategy costly.

Each eval task is tagged with the axis it exercises (``lexical``, ``domain``,
``trap``, or a combination) so a result can be reported per shift rather than
as one average that hides where a layer helps.

Self-contained and cheap: the simulator is the one ``textgame`` already ships,
with no model in the environment, so a run's cost is only the agent's own
tokens. That is the point -- a harness test should not cost more than the
harness.
"""
from __future__ import annotations

from typing import Any

from meta_evolver.benchmarks import textworld
from meta_evolver.benchmarks.base import BenchmarkAdapter
from meta_evolver.benchmarks.textworld import TextGameEnv
from meta_evolver.core.registry import register_benchmark
from meta_evolver.core.seeding import derive_rng
from meta_evolver.core.types import TaskSpec

#: room -> (receptacles, objects that plausibly belong there)
ROOMS: dict[str, tuple[list[str], list[str]]] = {
    "kitchen": (
        ["countertop 1", "countertop 2", "drawer 1", "drawer 2", "cabinet 1", "fridge 1", "sinkbasin 1"],
        ["knife 1", "fork 1", "spoon 1", "mug 1", "plate 1", "bowl 1", "pan 1"],
    ),
    "bathroom": (
        ["countertop 1", "cabinet 1", "cabinet 2", "toilet 1", "sinkbasin 1", "shelf 1"],
        ["soapbottle 1", "cloth 1", "towel 1", "toothbrush 1"],
    ),
    "bedroom": (
        ["desk 1", "bed 1", "drawer 1", "drawer 2", "shelf 1", "laundryhamper 1", "safe 1"],
        ["book 1", "pillow 1", "lamp 1", "watch 1", "cellphone 1"],
    ),
    "garage": (
        ["workbench 1", "shelf 1", "shelf 2", "cabinet 1", "garbagecan 1", "toolbox 1"],
        ["hammer 1", "wrench 1", "screwdriver 1", "flashlight 1"],
    ),
    "office": (
        ["desk 1", "desk 2", "drawer 1", "shelf 1", "cabinet 1", "garbagecan 1", "safe 1"],
        ["pen 1", "notebook 1", "stapler 1", "keyboard 1", "mouse 1"],
    ),
}
TRAIN_ROOMS = ("kitchen", "bathroom")
EVAL_ROOMS = ("bedroom", "garage", "office")

#: Wording families. Train uses the first; eval uses the rest -- same job.
TRAIN_CLEAN = ["put a clean {obj} in {goal}", "put a clean {obj} on {goal}"]
TRAIN_PLAIN = ["put {obj} on {goal}", "put {obj} in {goal}"]
EVAL_CLEAN = [
    "wash the {obj} and place it in {goal}",
    "I need the {obj} rinsed and stored in {goal}",
    "scrub {obj} until spotless, then leave it in {goal}",
]
EVAL_PLAIN = ["bring {obj} over to {goal}", "move {obj} to {goal}", "relocate {obj} to {goal}"]


def _name(obj: str) -> str:
    return obj.rsplit(" ", 1)[0]


def _make_layout(rng, room: str, split: str, shift: set[str], idx: int) -> tuple[str, dict[str, Any]]:
    receptacles, objects = ROOMS[room]
    obj = rng.choice(objects)
    pool = list(receptacles)
    rng.shuffle(pool)
    location = pool[0]
    goal = next(r for r in pool[1:] if r != location)
    # A trap: the object sits where a first-guess prior (flat surfaces) looks last.
    if "trap" in shift:
        closed = [r for r in receptacles if r.split(" ")[0] in {"garbagecan", "laundryhamper", "safe", "fridge", "toolbox"}]
        if closed:
            location = rng.choice(closed)
            if goal == location:
                goal = next(r for r in pool if r != location)
    requires_clean = rng.random() < 0.5
    if split == "train" or "lexical" not in shift:
        template = rng.choice(TRAIN_CLEAN if requires_clean else TRAIN_PLAIN)
    else:
        template = rng.choice(EVAL_CLEAN if requires_clean else EVAL_PLAIN)
    layout = {
        "instruction": template.format(obj=_name(obj), goal=goal),
        "split": split,
        "receptacles": receptacles[:],
        "object": obj,
        "object_location": location,
        "goal_receptacle": goal,
        "requires_clean": requires_clean,
        "room": room,
        "shift": sorted(shift),
    }
    return f"{split[:2]}_{room}_{_name(obj)}_{idx:03d}", layout


def generate_layouts(n_train: int = 24, n_eval: int = 16, seed: int = 0) -> dict[str, dict[str, Any]]:
    rng = derive_rng(seed, "textgame_proc")
    layouts: dict[str, dict[str, Any]] = {}
    for i in range(n_train):
        tid, lay = _make_layout(rng, TRAIN_ROOMS[i % len(TRAIN_ROOMS)], "train", set(), i)
        layouts[tid] = lay
    # Eval cycles through the shift axes so each is represented evenly.
    axes: list[set[str]] = [{"lexical"}, {"domain"}, {"trap"}, {"lexical", "domain", "trap"}]
    for i in range(n_eval):
        shift = axes[i % len(axes)]
        room = EVAL_ROOMS[i % len(EVAL_ROOMS)] if "domain" in shift else TRAIN_ROOMS[i % len(TRAIN_ROOMS)]
        tid, lay = _make_layout(rng, room, "eval", shift, i)
        layouts[tid] = lay
    return layouts


@register_benchmark("textgame_proc")
class ProcTextGameBenchmark(BenchmarkAdapter):
    description = "Procedural text game; eval is shifted in wording, domain and prior-trap placement."

    def __init__(self, n_train: int = 24, n_eval: int = 16, seed: int = 0, max_steps: int = 20) -> None:
        self.max_steps = int(max_steps)
        self.layouts = generate_layouts(int(n_train), int(n_eval), int(seed))

    def task_ids(self, split: str = "train") -> list[str]:
        if split == "all":
            return list(self.layouts)
        return [t for t, lay in self.layouts.items() if lay["split"] == split]

    def make_env(self, task_id: str, curriculum_level: float = 0.0, seed: int = 0) -> TextGameEnv:
        return TextGameEnv(task_id=task_id, max_steps=self.max_steps, seed=seed, layouts=self.layouts)

    def instruction_for(self, task_id: str) -> str:
        return self.layouts.get(task_id, {}).get("instruction", task_id)

    def specs(self) -> list[TaskSpec]:
        return [
            TaskSpec(
                task_id=tid,
                instruction=lay["instruction"],
                split=lay["split"],
                difficulty=0.75 if lay["split"] == "eval" else 0.4,
                tags=["embodied", "search", *lay["shift"]],
            )
            for tid, lay in self.layouts.items()
        ]

    def system_prompt(self) -> str:
        return textworld.TextGameBenchmark(max_steps=self.max_steps).system_prompt()
