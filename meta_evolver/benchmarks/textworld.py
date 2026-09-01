"""Text-game benchmark -- embodied search with an admissible-action space.

An ALFWorld-shaped environment: a room of containers, an object to find, and a
place to put it. Included because it exercises a failure mode the DevOps suite
cannot -- *search under a misleading prior*.

That is the setting where retrieval-augmented agents break down. The bank
returns "kitchen knives are in countertop 1-3"; this room's knife is in a
drawer; the agent checks the three countertops, finds nothing, and checks them
again, because the memory says so and the observations are only weak evidence
against it. The out-of-distribution layouts here are built to trigger exactly
that, which is what makes them a usable test of the adaptive controller rather
than a demo of it.

Self-contained: no ALFWorld install, no TextWorld engine, no dataset download.
The simulator is small enough to read and deterministic given a seed, so a
regression in eviction behaviour shows up as a failing test rather than as a
drifting number on a benchmark nobody can run in CI.
"""
from __future__ import annotations

from typing import Any

from meta_evolver.benchmarks.base import BenchmarkAdapter, tool_schema
from meta_evolver.core.env import ActionableEnv
from meta_evolver.core.registry import register_benchmark
from meta_evolver.core.seeding import derive_rng
from meta_evolver.core.types import (
    Action,
    EnvResetResponse,
    EnvResponse,
    EvaluationResult,
    Observation,
    TaskSpec,
)

#: Layout families. `in_distribution` rooms put the target where a naive prior
#: expects it; `out_of_distribution` rooms deliberately do not.
LAYOUTS: dict[str, dict[str, Any]] = {
    "kitchen_knife_id": {
        "instruction": "put a clean knife in drawer 1",
        "split": "train",
        "receptacles": [
            "countertop 1",
            "countertop 2",
            "drawer 1",
            "drawer 2",
            "dishwasher 1",
            "cabinet 1",
        ],
        "object": "knife 1",
        "object_location": "countertop 2",
        "goal_receptacle": "drawer 1",
        "requires_clean": True,
    },
    "kitchen_mug_id": {
        "instruction": "put a clean mug in cabinet 1",
        "split": "train",
        "receptacles": ["countertop 1", "sink 1", "cabinet 1", "cabinet 2", "shelf 1"],
        "object": "mug 1",
        "object_location": "countertop 1",
        "goal_receptacle": "cabinet 1",
        "requires_clean": True,
    },
    "bathroom_soap_id": {
        "instruction": "put soapbottle 1 on countertop 1",
        "split": "train",
        "receptacles": ["countertop 1", "cabinet 1", "toilet 1", "sinkbasin 1"],
        "object": "soapbottle 1",
        "object_location": "cabinet 1",
        "goal_receptacle": "countertop 1",
        "requires_clean": False,
    },
    "kitchen_knife_ood": {
        # The knife is in the last place a countertop-first prior would look.
        "instruction": "put a clean knife in drawer 1",
        "split": "eval",
        "receptacles": [
            "countertop 1",
            "countertop 2",
            "countertop 3",
            "drawer 1",
            "drawer 2",
            "garbagecan 1",
        ],
        "object": "knife 1",
        "object_location": "garbagecan 1",
        "goal_receptacle": "drawer 1",
        "requires_clean": True,
    },
    "bedroom_book_ood": {
        # No family resemblance to the kitchen tasks at all: every retrieved
        # strategy will be off-target, and following one costs the episode.
        "instruction": "put book 1 on desk 1",
        "split": "eval",
        "receptacles": ["desk 1", "bed 1", "drawer 1", "shelf 1", "laundryhamper 1"],
        "object": "book 1",
        "object_location": "laundryhamper 1",
        "goal_receptacle": "desk 1",
        "requires_clean": False,
    },
}


class TextGameEnv(ActionableEnv):
    """A room, a hidden object, and a goal receptacle."""

    env_type = "text_game"

    tool_schemas = [
        tool_schema(
            "do",
            (
                "Perform one action in the room. It must be one of the admissible "
                "commands listed in the observation, copied exactly."
            ),
            {"text": {"type": "string", "description": "e.g. 'go to drawer 1'"}},
            ["text"],
        )
    ]

    def __init__(self, task_id: str = "kitchen_knife_id", max_steps: int = 25, seed: int = 0):
        self.task_id = task_id if task_id in LAYOUTS else "kitchen_knife_id"
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.layout: dict[str, Any] = {}
        self.location = "start"
        self.opened: set[str] = set()
        self.holding: str | None = None
        self.cleaned = False
        self.placed = False
        self.step_count = 0
        self.last_feedback = ""
        self._reset_state()

    def _reset_state(self) -> None:
        self.layout = dict(LAYOUTS[self.task_id])
        # Receptacle order is shuffled per seed so an agent cannot memorize a
        # position in the list instead of learning to search.
        receptacles = list(self.layout["receptacles"])
        derive_rng(self.seed, self.task_id).shuffle(receptacles)
        self.layout["receptacles"] = receptacles
        self.location = "start"
        self.opened = set()
        self.holding = None
        self.cleaned = False
        self.placed = False
        self.step_count = 0
        self.last_feedback = "You are in the middle of the room. Looking around you see several receptacles."

    # -- step loop ---------------------------------------------------------

    def reset(self, seed=None, options=None) -> EnvResetResponse:
        opts = options or {}
        requested = opts.get("task_id")
        if requested in LAYOUTS:
            self.task_id = requested
        if seed is not None:
            self.seed = int(seed)
        self._reset_state()
        return EnvResetResponse(
            observation=self.observe(),
            info={
                "task_id": self.task_id,
                "instruction": self.layout["instruction"],
                "title": self.layout["instruction"],
            },
        )

    def step(self, action: Action) -> EnvResponse:
        self.step_count += 1
        text = str(action.kwargs.get("text", action.name)).strip().lower()
        self.last_feedback = self._apply(text)

        result = self.evaluate()
        return EnvResponse(
            observation=self.observe(),
            reward=result.score,
            terminated=self.placed,
            truncated=self.step_count >= self.max_steps,
            info={"result": {"feedback": self.last_feedback}, "step": self.step_count},
        )

    def _apply(self, text: str) -> str:
        receptacles = self.layout["receptacles"]
        obj = self.layout["object"]

        if text.startswith("go to "):
            target = text[len("go to ") :].strip()
            if target not in receptacles:
                return f"There is no {target} here."
            self.location = target
            if target in self.opened or not _is_closed(target):
                return f"You arrive at {target}. {self._contents(target)}"
            return f"You arrive at {target}. The {target} is closed."

        if text.startswith("open "):
            target = text[len("open ") :].strip()
            if target != self.location:
                return f"You need to go to {target} first."
            self.opened.add(target)
            return f"You open {target}. {self._contents(target)}"

        if text.startswith("examine "):
            target = text[len("examine ") :].strip()
            if target != self.location:
                return f"You need to go to {target} first."
            return self._contents(target)

        if text.startswith("take "):
            # "take knife 1 from drawer 2"
            parts = text[len("take ") :].split(" from ")
            item = parts[0].strip()
            if item != obj:
                return f"You see no {item} here."
            if self.location != self.layout["object_location"]:
                return f"The {item} is not at {self.location}."
            if _is_closed(self.location) and self.location not in self.opened:
                return f"The {self.location} is closed."
            self.holding = obj
            return f"You pick up the {obj}."

        if text.startswith("clean "):
            if self.holding != obj:
                return f"You are not holding the {obj}."
            self.cleaned = True
            return f"You clean the {obj}."

        if text.startswith("put ") or text.startswith("move "):
            goal = self.layout["goal_receptacle"]
            if self.holding != obj:
                return f"You are not holding the {obj}."
            if goal not in text:
                return f"You need to put it in/on {goal}."
            if self.location != goal:
                return f"You need to go to {goal} first."
            if self.layout["requires_clean"] and not self.cleaned:
                return f"The {obj} is dirty. Clean it first."
            self.placed = True
            return f"You put the {obj} in/on {goal}. Task complete."

        return "Nothing happens. Use one of the admissible commands exactly as written."

    def _contents(self, receptacle: str) -> str:
        if receptacle == self.layout["object_location"] and self.holding is None:
            return f"On {receptacle} you see a {self.layout['object']}."
        return f"On {receptacle} you see nothing useful."

    # -- views -------------------------------------------------------------

    def admissible_commands(self) -> list[str]:
        commands: list[str] = []
        obj = self.layout["object"]
        for receptacle in self.layout["receptacles"]:
            commands.append(f"go to {receptacle}")
        if self.location != "start":
            if _is_closed(self.location) and self.location not in self.opened:
                commands.append(f"open {self.location}")
            commands.append(f"examine {self.location}")
            if self.holding is None:
                commands.append(f"take {obj} from {self.location}")
            else:
                commands.append(f"put {obj} in/on {self.location}")
        if self.holding == obj and self.layout["requires_clean"] and not self.cleaned:
            commands.append(f"clean {obj}")
        return commands

    def observe(self) -> Observation:
        commands = self.admissible_commands()
        text = (
            f"Task: {self.layout['instruction']}\n"
            f"{self.last_feedback}\n"
            f"You are at: {self.location}. Holding: {self.holding or 'nothing'}.\n"
            f"Step {self.step_count}/{self.max_steps}\n"
            f"Admissible commands: {commands}"
        )
        return Observation(
            text=text,
            data={
                "admissible_commands": commands,
                "location": self.location,
                "holding": self.holding,
            },
        )

    def evaluate(self) -> EvaluationResult:
        # Dense partial credit: found it, holding it, cleaned it, placed it.
        found = self.layout["object_location"] in self.opened or self.holding is not None
        stages = [found, self.holding is not None, self.cleaned or not self.layout["requires_clean"], self.placed]
        return EvaluationResult(
            success=self.placed,
            score=1.0 if self.placed else 0.25 * sum(bool(s) for s in stages),
            metrics={
                "found": bool(found),
                "holding": self.holding is not None,
                "cleaned": self.cleaned,
                "placed": self.placed,
                "steps": self.step_count,
                "receptacles_visited": len(self.opened),
            },
        )

    def get_env_state(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "location": self.location,
            "holding": self.holding,
            "opened": sorted(self.opened),
            "verified": self.placed,
        }


def _is_closed(receptacle: str) -> bool:
    """Containers you must open; surfaces you can just look at."""
    return receptacle.split(" ")[0] in {
        "drawer",
        "cabinet",
        "dishwasher",
        "garbagecan",
        "laundryhamper",
        "fridge",
        "microwave",
        "safe",
    }


@register_benchmark("textgame")
class TextGameBenchmark(BenchmarkAdapter):
    """Embodied search with in-distribution and out-of-distribution layouts."""

    description = "ALFWorld-style object search; the eval split is adversarial to memory priors."

    def __init__(self, max_steps: int = 25) -> None:
        self.max_steps = int(max_steps)

    def task_ids(self, split: str = "train") -> list[str]:
        if split == "all":
            return list(LAYOUTS)
        return [t for t, spec in LAYOUTS.items() if spec["split"] == split]

    def make_env(self, task_id: str, curriculum_level: float = 0.0, seed: int = 0):
        return TextGameEnv(task_id=task_id, max_steps=self.max_steps, seed=seed)

    def instruction_for(self, task_id: str) -> str:
        return LAYOUTS.get(task_id, {}).get("instruction", task_id)

    def specs(self) -> list[TaskSpec]:
        return [
            TaskSpec(
                task_id=tid,
                instruction=spec["instruction"],
                split=spec["split"],
                difficulty=0.75 if spec["split"] == "eval" else 0.4,
                tags=["embodied", "search"],
            )
            for tid, spec in LAYOUTS.items()
        ]

    def system_prompt(self) -> str:
        return (
            super().system_prompt()
            + "\nCall the `do` tool with one admissible command, copied exactly as\n"
            "printed. You cannot see inside a closed container until you go to it\n"
            "and open it. A receptacle you have already examined will not change.\n"
        )
