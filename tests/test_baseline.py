"""Phase 0 measurement plumbing: subsets, CLI kwargs, token budget, arms."""
from __future__ import annotations

import json

import pytest

from meta_evolver.benchmarks.base import stratified_subset
from meta_evolver.benchmarks.custom import FunctionBenchmark, Task
from meta_evolver.cli.main import _bench_kwargs, _coerce
from meta_evolver.core.types import Trajectory
from meta_evolver.eval.baseline import ARMS, by_category, run_baseline, summarize
from meta_evolver.graphs.evolution import EvolutionConfig
from meta_evolver.llm.client import ScriptedChatModel, tool_call_message

# -- stratified_subset --------------------------------------------------------


def test_subset_returns_everything_when_no_limit():
    items = list(range(10))
    assert stratified_subset(items, None, 0, key=lambda x: x % 2) == items
    assert stratified_subset(items, 50, 0, key=lambda x: x % 2) == items


def test_subset_is_deterministic_in_seed():
    items = list(range(100))
    key = lambda x: x % 5  # noqa: E731
    a = stratified_subset(items, 20, 7, key=key)
    b = stratified_subset(items, 20, 7, key=key)
    c = stratified_subset(items, 20, 8, key=key)
    assert a == b
    assert a != c


def test_subset_keeps_every_category_represented():
    # 90 of one category, 10 of another: a plain random 20 would often miss
    # the rare one entirely, which is the failure this function exists for.
    items = [("common", i) for i in range(90)] + [("rare", i) for i in range(10)]
    picked = stratified_subset(items, 20, 0, key=lambda x: x[0])
    labels = {label for label, _ in picked}
    assert labels == {"common", "rare"}
    assert len(picked) == 20


def test_subset_preserves_input_order():
    items = list(range(50))
    picked = stratified_subset(items, 10, 3, key=lambda x: x % 2)
    assert picked == sorted(picked)


def test_subset_honours_the_exact_count_with_awkward_ratios():
    items = [(f"c{i % 7}", i) for i in range(100)]
    assert len(stratified_subset(items, 13, 1, key=lambda x: x[0])) == 13


# -- CLI kwargs ---------------------------------------------------------------


def test_coerce_gives_numbers_and_bools_their_types():
    assert _coerce("50") == 50
    assert _coerce("0.25") == 0.25
    assert _coerce("true") is True
    assert _coerce("False") is False
    assert _coerce("locomo10.json") == "locomo10.json"


def test_bench_kwargs_forwards_nothing_by_default():
    assert _bench_kwargs(None, None, []) == {}


def test_bench_kwargs_collects_data_path_limit_and_pairs(tmp_path):
    kwargs = _bench_kwargs(tmp_path / "d.json", 50, ["eval_fraction=0.5", "seed=3"])
    assert kwargs["data_path"] == tmp_path / "d.json"
    assert kwargs["limit"] == 50
    assert kwargs["eval_fraction"] == 0.5
    assert kwargs["seed"] == 3


def test_bench_kwargs_rejects_a_malformed_pair():
    import typer

    with pytest.raises(typer.BadParameter):
        _bench_kwargs(None, None, ["limit"])


# -- token budget -------------------------------------------------------------


def test_max_total_tokens_defaults_to_unbounded():
    assert EvolutionConfig().max_total_tokens is None


def _metered_model(per_call: int = 500) -> ScriptedChatModel:
    """A solving model whose replies report token usage.

    ``ScriptedChatModel`` alone reports none, so a budget test against it
    would pass for the wrong reason -- zero spend never exceeds any budget.
    """

    def responder(messages, tools):
        used = {
            tc["name"]
            for m in messages
            if hasattr(m, "tool_calls")
            for tc in (m.tool_calls or [])
        }
        if "submit_resolution" in used:
            reply = tool_call_message("run_healthcheck", endpoint="payments/charge")
        else:
            reply = tool_call_message(
                "submit_resolution", root_cause="pool limit", action_taken="raised it"
            )
        reply.usage_metadata = {
            "input_tokens": per_call,
            "output_tokens": per_call,
            "total_tokens": per_call * 2,
        }
        return reply

    return ScriptedChatModel(responder=responder)


def test_evolve_stops_once_the_token_budget_is_spent():
    """The budget must cut a run short of its generation count."""
    from meta_evolver.core.evolver import MetaEvolver

    config = dict(
        generations=4,
        induce_memories=False,
        optimize_prompt=False,
        curriculum=False,
        patience=99,  # isolate the budget from early stopping
    )

    def run(max_total_tokens):
        evolver = MetaEvolver(
            benchmark="devops",
            chat_model=_metered_model(),
            config=EvolutionConfig(max_total_tokens=max_total_tokens, **config),
            telemetry=False,
        )
        try:
            return evolver.evolve()
        finally:
            evolver.close()

    unbounded = run(None)
    bounded = run(1)

    assert len(unbounded) == 4
    assert len(bounded) == 1
    assert bounded[0].tokens > 0  # the budget was compared against real spend


# -- reporting helpers --------------------------------------------------------


def _trajectory(task_id: str, success: bool, category: str | None) -> Trajectory:
    from meta_evolver.core.types import Action, StepRecord

    return Trajectory(
        task_id=task_id,
        steps=[StepRecord(step_idx=1, action=Action(name="answer"))],
        success=success,
        score=1.0 if success else 0.0,
        metrics={"category_name": category} if category else {},
    )


def test_by_category_splits_on_the_benchmarks_own_label():
    out = by_category(
        [
            _trajectory("a", True, "temporal"),
            _trajectory("b", False, "temporal"),
            _trajectory("c", True, "multi-hop"),
        ]
    )
    assert out["temporal"]["n"] == 2
    assert out["temporal"]["pass_rate"] == 0.5
    assert out["multi-hop"]["pass_rate"] == 1.0


def test_by_category_excludes_errored_episodes():
    errored = Trajectory(task_id="x", error="llm: boom", metrics={"category_name": "temporal"})
    out = by_category([errored, _trajectory("a", True, "temporal")])
    assert out["temporal"]["n"] == 1


def test_summarize_reports_spread_across_seeds():
    rows = [
        {"arm": "bank_only", "seed": 0, "pass_rate": 0.4, "avg_score": 0.4, "train_tokens": 10, "eval_tokens": 5},
        {"arm": "bank_only", "seed": 1, "pass_rate": 0.6, "avg_score": 0.6, "train_tokens": 10, "eval_tokens": 5},
    ]
    s = summarize(rows)["bank_only"]
    assert s["n_seeds"] == 2
    assert s["pass_rate_mean"] == pytest.approx(0.5)
    assert s["pass_rate_stdev"] > 0
    assert s["tokens_mean"] == pytest.approx(15)


# -- the whole runner, offline ------------------------------------------------


def _answer_benchmark() -> FunctionBenchmark:
    def answer(text: str) -> dict:
        """Submit the final answer."""
        return {"answer": text}

    tasks = [
        Task(
            id=f"t{i}",
            instruction="Say 42.",
            verify=lambda calls: any("42" in str(c.result) for c in calls),
            terminal_tools=("answer",),
            split="train" if i < 2 else "eval",
        )
        for i in range(4)
    ]
    return FunctionBenchmark(name="mini", tools={"answer": answer}, tasks=tasks)


def test_run_baseline_writes_every_arm_and_seed(tmp_path):
    target = run_baseline(
        benchmark_name="mini",
        benchmark_factory=_answer_benchmark,
        model_factory=lambda: ScriptedChatModel(
            responder=lambda messages, tools: tool_call_message("answer", text="42")
        ),
        model_name="scripted",
        seeds=(0, 1),
        generations=1,
        out_dir=tmp_path,
        trace=False,
    )
    data = json.loads(target.read_text(encoding="utf-8"))

    assert data["benchmark"] == "mini"
    assert len(data["rows"]) == len(ARMS) * 2
    assert set(data["summary"]) == set(ARMS)
    # The scripted model always answers correctly, so every arm passes and the
    # eval split is the one being scored.
    for row in data["rows"]:
        assert row["pass_rate"] == 1.0
        assert row["n_tasks"] == 2


def test_no_memory_arm_learns_nothing(tmp_path):
    target = run_baseline(
        benchmark_name="mini",
        benchmark_factory=_answer_benchmark,
        model_factory=lambda: ScriptedChatModel(
            responder=lambda messages, tools: tool_call_message("answer", text="42")
        ),
        model_name="scripted",
        seeds=(0,),
        arms=("no_memory",),
        generations=1,
        out_dir=tmp_path,
        trace=False,
    )
    rows = json.loads(target.read_text(encoding="utf-8"))["rows"]
    assert rows[0]["train_tokens"] == 0
    assert rows[0]["memories"] == 0


def test_extra_arms_for_the_graph_and_skill_layers_are_known_and_summarised():
    from meta_evolver.eval.baseline import ALL_ARMS, ARMS, EXTRA_ARMS, summarize

    assert ARMS == ("no_memory", "bank_only", "bank_evolve")  # the original three are unchanged
    assert set(EXTRA_ARMS) == {"bank_graph", "bank_graph_skills", "bank_graph_lift"}
    rows = [{"arm": a, "seed": 0, "pass_rate": 0.5, "avg_score": 0.5, "train_tokens": 1, "eval_tokens": 1}
            for a in ALL_ARMS]
    assert set(summarize(rows)) == set(ALL_ARMS)
