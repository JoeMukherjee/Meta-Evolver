"""MetaAdapter: the artifact, discount_usage, and SeededOptimizer -- the
mechanism that keeps an attached prompt from being applied directly.
"""
from __future__ import annotations

import json

import pytest

from meta_evolver.adapter import MetaAdapter, SeededOptimizer, discount_usage
from meta_evolver.core.types import MemoryItem
from meta_evolver.prompts.optimizer import PromptCandidate


def item(**kwargs) -> MemoryItem:
    base = {"scenario": "general", "lesson": "a lesson", "title": "t"}
    return MemoryItem(**{**base, **kwargs})


# -- discount_usage -----------------------------------------------------------


def test_discount_usage_scales_wins_and_uses():
    items = [item(wins=8, uses=10)]
    discount_usage(items, 0.5)
    assert items[0].wins == 4
    assert items[0].uses == 5


def test_discount_usage_zero_gives_a_brand_new_item():
    items = [item(wins=8, uses=10)]
    discount_usage(items, 0.0)
    assert items[0].wins == 0
    assert items[0].uses == 0


def test_discount_usage_one_imports_at_full_strength():
    items = [item(wins=8, uses=10)]
    discount_usage(items, 1.0)
    assert items[0].wins == 8
    assert items[0].uses == 10


def test_discount_usage_clamps_out_of_range_factors():
    items = [item(wins=8, uses=10)]
    discount_usage(items, 5.0)
    assert items[0].wins == 8 and items[0].uses == 10
    items2 = [item(wins=8, uses=10)]
    discount_usage(items2, -1.0)
    assert items2[0].wins == 0 and items2[0].uses == 0


def test_discount_usage_scales_domain_stats_too():
    it = item()
    it.record_domain_usage("travel", True)
    it.record_domain_usage("travel", True)
    discount_usage([it], 0.5)
    assert it.domain_stats["travel"] == [1.0, 1.0]


# -- MetaAdapter ---------------------------------------------------------------


class _FakeReport:
    def __init__(self, pass_rate):
        self.pass_rate = pass_rate


class _FakeBenchmark:
    name = "mini"


class _FakeEvolver:
    """Just enough of MetaEvolver's surface for from_evolver()/save()."""

    def __init__(self, tmp_path):
        from meta_evolver.memory.bank import ReasoningMemoryBank

        self.model_name = "scripted"
        self.benchmark = _FakeBenchmark()
        self.prompt_template = "you are a careful agent"
        self.prompt_version = "g1c0"
        self.curriculum_level = 0.4
        self.rehearsal_ids = ["t1", "t2"]
        self.reports = [_FakeReport(0.5), _FakeReport(0.75)]
        self.bank = ReasoningMemoryBank()
        self.bank.add(item(id="m1", lesson="restart before resubmitting"))
        self.telemetry_run_dir = tmp_path

    def save(self, memory_path=None, prompt_path=None):
        out = {}
        out["memory"] = self.bank.save(memory_path)
        from pathlib import Path

        Path(prompt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(prompt_path).write_text(self.prompt_template, encoding="utf-8")
        out["prompt"] = prompt_path
        return out


def test_from_evolver_captures_the_real_fields(tmp_path):
    evolver = _FakeEvolver(tmp_path)
    adapter = MetaAdapter.from_evolver(evolver)
    assert adapter.model_name == "scripted"
    assert adapter.benchmark == "mini"
    assert adapter.prompt_text == "you are a careful agent"
    assert adapter.prompt_version == "g1c0"
    assert adapter.curriculum_level == 0.4
    assert adapter.rehearsal_ids == ["t1", "t2"]
    assert adapter.source_generations == 2
    assert adapter.source_pass_rate == 0.75  # the *last* report, not the first
    assert adapter.created_at  # a real timestamp, not left blank
    assert not adapter.legacy


def test_from_evolver_with_no_adopted_tools_attribute_defaults_to_empty():
    """A pre-Phase-4 MetaEvolver stand-in (no ``adopted_tools`` at all) must
    still build a manifest -- the field is additive, not required."""
    evolver = _FakeEvolver(None)  # _FakeEvolver never sets adopted_tools
    assert not hasattr(evolver, "adopted_tools")
    adapter = MetaAdapter.from_evolver(evolver)
    assert adapter.tools == []


def test_from_evolver_captures_adopted_tools_tagged_with_permissions():
    evolver = _FakeEvolver(None)
    evolver.adopted_tools = [
        {"name": "compute_total", "code": "def compute_total(): ...", "description": "d",
         "generation": 0, "score": 1.0, "baseline_score": 0.0}
    ]
    adapter = MetaAdapter.from_evolver(evolver)
    assert len(adapter.tools) == 1
    assert adapter.tools[0]["name"] == "compute_total"
    assert adapter.tools[0]["permissions"] == "pure_function"


def test_save_writes_a_manifest_with_content_hashes(tmp_path):
    evolver = _FakeEvolver(tmp_path)
    adapter = MetaAdapter.from_evolver(evolver)
    paths = adapter.save(evolver, tmp_path / "out")

    assert paths["manifest"].exists()
    assert paths["memory"].exists()
    assert paths["prompt"].exists()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["prompt_version"] == "g1c0"
    assert manifest["content_sha256"]["prompt"]
    assert manifest["content_sha256"]["memory"]


def test_save_appends_one_archive_line_per_call(tmp_path):
    evolver = _FakeEvolver(tmp_path)
    adapter = MetaAdapter.from_evolver(evolver)
    out_dir = tmp_path / "out"
    paths_1 = adapter.save(evolver, out_dir)
    paths_2 = adapter.save(evolver, out_dir)

    assert paths_1["archive"] == paths_2["archive"] == out_dir / "archive.jsonl"
    lines = paths_2["archive"].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["prompt_version"] == "g1c0"
    assert entry["source_pass_rate"] == 0.75
    assert entry["tool_names"] == []


def test_save_archive_line_records_adopted_tool_names(tmp_path):
    evolver = _FakeEvolver(tmp_path)
    evolver.adopted_tools = [{"name": "compute_total", "code": "", "description": ""}]
    adapter = MetaAdapter.from_evolver(evolver)
    paths = adapter.save(evolver, tmp_path / "out")
    entry = json.loads(paths["archive"].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["tool_names"] == ["compute_total"]


def test_load_round_trips_a_saved_adapter(tmp_path):
    evolver = _FakeEvolver(tmp_path)
    MetaAdapter.from_evolver(evolver).save(evolver, tmp_path / "out")

    loaded = MetaAdapter.load(tmp_path / "out")
    assert loaded.prompt_text == "you are a careful agent"
    assert loaded.rehearsal_ids == ["t1", "t2"]
    assert not loaded.legacy


def test_load_on_a_directory_with_no_manifest_is_legacy(tmp_path):
    empty_dir = tmp_path / "old-save"
    empty_dir.mkdir()
    (empty_dir / "memory.jsonl").write_text("", encoding="utf-8")
    (empty_dir / "prompt.txt").write_text("an old prompt", encoding="utf-8")

    loaded = MetaAdapter.load(empty_dir)
    assert loaded.legacy is True
    # No manifest fields, but the sidecar prompt file is still recoverable.
    assert loaded.prompt_text == ""  # load() alone doesn't reach into sidecars
    assert loaded.memory_items(empty_dir) == []


def test_memory_items_parses_the_portable_jsonl(tmp_path):
    evolver = _FakeEvolver(tmp_path)
    MetaAdapter.from_evolver(evolver).save(evolver, tmp_path / "out")

    loaded = MetaAdapter.load(tmp_path / "out")
    items = loaded.memory_items(tmp_path / "out")
    assert len(items) == 1
    assert items[0].id == "m1"
    assert "restart" in items[0].lesson


# -- SeededOptimizer ------------------------------------------------------------


class _FakeInner:
    """Records calls; returns a fixed candidate list, or [] to simulate
    "nothing failed this generation"."""

    def __init__(self, candidates=None):
        self.candidates = candidates or []
        self.calls = 0
        self.last_error = "some prior error"

    def propose(self, current_prompt, trajectories, generation=0, benchmark=""):
        self.calls += 1
        return list(self.candidates)


def test_seeded_optimizer_passes_through_with_no_seed():
    inner = _FakeInner(candidates=[PromptCandidate(text="x", version="v1", parent="p")])
    wrapped = SeededOptimizer(inner)
    out = wrapped.propose("prompt", [], generation=0)
    assert [c.version for c in out] == ["v1"]
    assert inner.calls == 1


def test_seeded_optimizer_prepends_the_seed():
    inner = _FakeInner(candidates=[PromptCandidate(text="x", version="v1", parent="p")])
    wrapped = SeededOptimizer(inner)
    wrapped.seed("imported text", version="adapter:g0c0")
    out = wrapped.propose("prompt", [], generation=0)
    assert [c.version for c in out] == ["adapter:g0c0", "v1"]
    assert out[0].text == "imported text"


def test_seeded_optimizer_returns_the_seed_even_when_inner_proposes_nothing():
    """The property this whole class exists for: a generation with no
    failures yet must not silently drop a pending import."""
    inner = _FakeInner(candidates=[])
    wrapped = SeededOptimizer(inner)
    wrapped.seed("imported text", version="adapter:g0c0")
    out = wrapped.propose("prompt", [], generation=0)
    assert [c.version for c in out] == ["adapter:g0c0"]


def test_seeded_optimizer_seed_is_consumed_once():
    inner = _FakeInner(candidates=[])
    wrapped = SeededOptimizer(inner)
    wrapped.seed("imported text")
    first = wrapped.propose("prompt", [], generation=0)
    second = wrapped.propose("prompt", [], generation=1)
    assert len(first) == 1
    assert second == []


def test_seeded_optimizer_forwards_last_error_from_inner():
    wrapped = SeededOptimizer(_FakeInner())
    wrapped.propose("prompt", [], generation=0)
    assert wrapped.last_error == "some prior error"


# -- Agent.attach_adapter, end to end -----------------------------------------
#
# BenchmarkAdapter.sample() only creates a held-out split at all once the
# train pool has >=3 tasks (see benchmarks/base.py) -- a 2-task pool silently
# falls into "adopt unconditionally", which is not what these tests are
# meant to exercise. Pool ["ta","tb","tc"], seed=0: holdout is ["tb"] (probed
# directly via derive_rng), so "tb" is validation, "ta"/"tc" are training.
# The imported prompt is judged purely by whether it passes "tb" -- proving
# adoption is real validation, not the import taking effect directly.


def _import_benchmark(pass_tb_under_imported: bool):
    from meta_evolver.benchmarks.custom import FunctionBenchmark, Task

    def answer(text: str) -> dict:
        """Submit the final answer."""
        return {"answer": text}

    def look(**_: object) -> dict:
        """A harmless non-terminal tool."""
        return {"ok": True}

    def verify_tb(calls):
        return any(c.name == "answer" for c in calls)

    tasks = [
        Task(id="ta", instruction="ta", split="train", terminal_tools=("answer",)),
        Task(
            id="tb",
            instruction="tb",
            split="train",
            terminal_tools=("answer",),
            verify=verify_tb if pass_tb_under_imported else (lambda calls: False),
        ),
        Task(id="tc", instruction="tc", split="train", terminal_tools=("answer",)),
    ]
    return FunctionBenchmark(
        name="import-mini", tools={"answer": answer, "look": look}, tasks=tasks, max_steps=2
    )


def _responder_factory(imported_passes: bool):
    from meta_evolver.llm.client import tool_call_message

    def _reply(messages, tools):
        system = str(getattr(messages[0], "content", "")) if messages else ""
        task_line = next(
            (
                str(getattr(m, "content", "")).split("\n", 1)[0]
                for m in messages
                if str(getattr(m, "content", "")).startswith("Task: ")
            ),
            "",
        )
        is_imported = "IMPORTED-PROMPT-MARKER" in system
        if task_line == "Task: tb":
            if is_imported and imported_passes:
                return tool_call_message("answer", text="done")
            if not is_imported:
                return tool_call_message("look")  # base prompt never solves tb
            return tool_call_message("look")  # imported prompt fails tb in this scenario
        # ta: always fails, so PromptOptimizer's own propose() has something
        # to react to alongside whatever is seeded.
        return tool_call_message("look")

    return _reply


def _saved_adapter_dir(tmp_path) -> "Path":
    from pathlib import Path

    from meta_evolver.agent import Agent
    from meta_evolver.llm.client import ScriptedChatModel

    model = ScriptedChatModel(responder=_responder_factory(imported_passes=True))
    source = Agent(model=model, memory=True)
    source.evolver.prompt_template = "IMPORTED-PROMPT-MARKER: solve tb directly."
    source.evolver.prompt_version = "g3c1"
    source.bank.add(item(id="learned-1", lesson="always answer tb with done"))

    out_dir = tmp_path / "source-adapter"
    source.save(out_dir)
    return out_dir


def test_attach_adapter_merges_memory_with_discount(tmp_path):
    from meta_evolver.agent import Agent
    from meta_evolver.llm.client import ScriptedChatModel

    adapter_dir = _saved_adapter_dir(tmp_path)
    target = Agent(model=ScriptedChatModel(script=[]), memory=True)
    target.attach_adapter(adapter_dir, discount=0.5)

    assert any(m.id == "learned-1" for m in target.bank.items)


def test_attach_adapter_prompt_is_adopted_when_it_beats_the_incumbent(tmp_path):
    from meta_evolver.agent import Agent
    from meta_evolver.graphs.evolution import EvolutionConfig
    from meta_evolver.llm.client import ScriptedChatModel

    adapter_dir = _saved_adapter_dir(tmp_path)
    model = ScriptedChatModel(responder=_responder_factory(imported_passes=True))
    target = Agent(model=model, memory=False, evolve_prompt=True)
    target.attach(_import_benchmark(pass_tb_under_imported=True))
    target.attach_adapter(adapter_dir)
    target.evolver.config = EvolutionConfig(
        generations=1,
        max_steps=2,
        validation_fraction=0.34,
        validate_prompt=True,
        curriculum=False,
        patience=99,
        seed=0,
    )

    target.evolve()

    assert "IMPORTED-PROMPT-MARKER" in target.evolver.prompt_template
    assert "adapter:g3c1" in target.evolver.prompt_version


def test_attach_adapter_prompt_is_rejected_when_it_does_not_beat_the_incumbent(tmp_path):
    """The whole point: an imported prompt that does not actually validate
    better must not become the active prompt just because it was attached."""
    from meta_evolver.agent import Agent
    from meta_evolver.graphs.evolution import EvolutionConfig
    from meta_evolver.llm.client import ScriptedChatModel

    adapter_dir = _saved_adapter_dir(tmp_path)
    model = ScriptedChatModel(responder=_responder_factory(imported_passes=False))
    target = Agent(model=model, memory=False, evolve_prompt=True)
    target.attach(_import_benchmark(pass_tb_under_imported=False))
    target.attach_adapter(adapter_dir)
    target.evolver.config = EvolutionConfig(
        generations=1,
        max_steps=2,
        validation_fraction=0.34,
        validate_prompt=True,
        curriculum=False,
        patience=99,
        seed=0,
    )

    target.evolve()

    assert "IMPORTED-PROMPT-MARKER" not in target.evolver.prompt_template
