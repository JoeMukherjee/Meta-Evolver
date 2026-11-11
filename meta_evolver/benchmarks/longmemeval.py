"""LongMemEval -- benchmarking chat assistants on long-term interactive memory.

Wu et al., ICLR 2025 (arXiv:2410.10813). https://github.com/xiaowu0162/LongMemEval

500 questions, each embedded in a long, multi-session "haystack" chat history
(roughly 115k tokens / 30-40 sessions for the ``_s`` split this adapter
targets). There is no way to put that in one prompt turn, so the agent has to
*page through it with tools* -- ``list_sessions``, ``read_session``,
``search_sessions`` -- before it can ``submit_answer``. That is the actual
long-term-memory test: whether retrieval finds and uses the right session,
not whether the model can read a wall of text in one shot.

Six question types probe distinct memory failure modes (information
extraction, multi-session reasoning, knowledge updates, temporal reasoning,
user preference, abstention on unanswerable questions); a question id
containing ``_abs`` is an abstention item, scored by whether the agent
correctly declines rather than by matching a factual answer.

Scoring reuses this benchmark's own published judge prompts **verbatim**,
copied from ``src/evaluation/evaluate_qa.py`` in the official repo, rather
than inventing new grading criteria -- fidelity to the real protocol matters
more here than convenience, and a homegrown rubric would make results
incomparable with everyone else's reported numbers.

Data is not bundled (the official release is 100MB+). Download the cleaned
split and point ``data_path`` at it::

    wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

::

    from meta_evolver.benchmarks.longmemeval import LongMemEvalBenchmark
    bench = LongMemEvalBenchmark(data_path="longmemeval_s_cleaned.json", judge_model=my_model)

Known limitation: ``evaluate()`` calls the judge model synchronously inside
an otherwise-async episode graph, exactly the real-I/O caveat
:mod:`meta_evolver.graphs.episode`'s own docstring warns about -- correct for
a single episode, a throughput cost under many concurrent rollouts. Without a
``judge_model`` at all, scoring falls back to case-insensitive substring
containment, which is far less faithful to the real protocol and only meant
for smoke-testing the adapter without paying for judge calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meta_evolver.benchmarks.base import BenchmarkAdapter, stratified_subset, tool_schema
from meta_evolver.core.env import ActionableEnv
from meta_evolver.core.registry import register_benchmark
from meta_evolver.core.types import (
    Action,
    EnvResetResponse,
    EnvResponse,
    EvaluationResult,
    Observation,
    TaskSpec,
)
from langchain_core.language_models.chat_models import BaseChatModel
from meta_evolver.llm.client import invoke_text
from langchain_core.messages import HumanMessage

#: Judge prompt templates, copied verbatim from LongMemEval's own
#: ``src/evaluation/evaluate_qa.py`` (``get_anscheck_prompt``), so grading
#: matches the published protocol exactly.
_JUDGE_TEMPLATES: dict[str, str] = {
    "default": (
        "I will give you a question, a correct answer, and a response from a "
        "model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response is equivalent to the correct "
        "answer or contains all the intermediate steps to get the correct "
        "answer, you should also answer yes. If the response only contains a "
        "subset of the information required by the answer, answer no. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: "
        "{response}\n\nIs the model response correct? Answer yes or no only."
    ),
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a "
        "model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response is equivalent to the correct "
        "answer or contains all the intermediate steps to get the correct "
        "answer, you should also answer yes. If the response only contains a "
        "subset of the information required by the answer, answer no. In "
        "addition, do not penalize off-by-one errors for the number of days. "
        "If the question asks for the number of days/weeks/months, etc., and "
        "the model makes off-by-one errors (e.g., predicting 19 days when the "
        "answer is 18), the model's response is still correct. \n\nQuestion: "
        "{question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a "
        "model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response contains some previous "
        "information along with an updated answer, the response should be "
        "considered as correct as long as the updated answer is the required "
        "answer.\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel "
        "Response: {response}\n\nIs the model response correct? Answer yes or "
        "no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized "
        "response, and a response from a model. Please answer yes if the "
        "response satisfies the desired response. Otherwise, answer no. The "
        "model does not need to reflect all the points in the rubric. The "
        "response is correct as long as it recalls and utilizes the user's "
        "personal information correctly.\n\nQuestion: {question}\n\nRubric: "
        "{answer}\n\nModel Response: {response}\n\nIs the model response "
        "correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a "
        "response from a model. Please answer yes if the model correctly "
        "identifies the question as unanswerable. The model could say that "
        "the information is incomplete, or some other information is given "
        "but the asked information is not.\n\nQuestion: {question}\n\n"
        "Explanation: {answer}\n\nModel Response: {response}\n\nDoes the "
        "model correctly identify the question as unanswerable? Answer yes "
        "or no only."
    ),
}


def _judge_template_for(question_type: str, is_abstention: bool) -> str:
    if is_abstention:
        return _JUDGE_TEMPLATES["abstention"]
    return _JUDGE_TEMPLATES.get(question_type, _JUDGE_TEMPLATES["default"])


class LongMemEvalRecord:
    """One parsed question from the official dataset JSON."""

    __slots__ = (
        "question_id",
        "question",
        "answer",
        "question_type",
        "question_date",
        "sessions",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.question_id: str = str(raw["question_id"])
        self.question: str = str(raw["question"])
        self.answer: str = str(raw["answer"])
        self.question_type: str = str(raw.get("question_type", "default"))
        self.question_date: str = str(raw.get("question_date", ""))
        session_ids = raw.get("haystack_session_ids") or []
        dates = raw.get("haystack_dates") or []
        sessions_raw = raw.get("haystack_sessions") or []
        self.sessions: list[dict[str, Any]] = []
        for i, turns in enumerate(sessions_raw):
            sid = session_ids[i] if i < len(session_ids) else f"session_{i}"
            date = dates[i] if i < len(dates) else ""
            text = "\n".join(
                f"{t.get('role', '?')}: {t.get('content', '')}" for t in turns
            )
            self.sessions.append({"id": sid, "date": date, "text": text})

    @property
    def is_abstention(self) -> bool:
        return "_abs" in self.question_id


def load_longmemeval(data_path: str | Path) -> list[LongMemEvalRecord]:
    """Parse the official dataset JSON (``longmemeval_s_cleaned.json`` etc.)."""
    raw = json.loads(Path(data_path).read_text(encoding="utf-8"))
    return [LongMemEvalRecord(r) for r in raw]


class LongMemEvalEnv(ActionableEnv):
    """Page through one question's session haystack, then answer."""

    env_type = "longmemeval"

    def __init__(self, record: LongMemEvalRecord, judge_model: BaseChatModel | None = None) -> None:
        self.record = record
        self.judge_model = judge_model
        self.submitted: str | None = None
        self.finished = False

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> EnvResetResponse:
        self.submitted = None
        self.finished = False
        previews = "\n".join(
            f"- {s['id']} ({s['date']}): {s['text'][:80]!r}..." for s in self.record.sessions
        )
        return EnvResetResponse(
            observation=Observation(
                text=(
                    f"{len(self.record.sessions)} chat sessions are available. "
                    f"Use list_sessions/read_session/search_sessions to find "
                    f"what's relevant, then submit_answer.\n\n{previews}"
                )
            ),
            info={"instruction": self.record.question, "title": self.record.question},
        )

    def step(self, action: Action) -> EnvResponse:
        name, kwargs = action.name, dict(action.kwargs or {})

        if name == "list_sessions":
            text = "\n".join(f"{s['id']} ({s['date']})" for s in self.record.sessions)
            return EnvResponse(observation=Observation(text=text or "(no sessions)"))

        if name == "read_session":
            sid = str(kwargs.get("session_id", ""))
            match = next((s for s in self.record.sessions if s["id"] == sid), None)
            text = match["text"] if match else f"unknown session_id {sid!r}"
            return EnvResponse(observation=Observation(text=text))

        if name == "search_sessions":
            query = str(kwargs.get("query", "")).lower()
            hits = [s["id"] for s in self.record.sessions if query in s["text"].lower()]
            text = f"sessions containing {query!r}: {hits}" if hits else "no matches"
            return EnvResponse(observation=Observation(text=text))

        if name == "submit_answer":
            self.submitted = str(kwargs.get("text", ""))
            self.finished = True
            return EnvResponse(
                observation=Observation(text="answer submitted"), terminated=True
            )

        return EnvResponse(
            observation=Observation(text=f"unknown tool {name!r}"),
            info={"result": {"error": f"unknown tool {name!r}"}},
        )

    def observe(self) -> Observation:
        return Observation(text=self.submitted or "")

    def evaluate(self) -> EvaluationResult:
        if self.submitted is None:
            return EvaluationResult(success=False, score=0.0, metrics={"reason": "no answer submitted"})

        if self.judge_model is None:
            correct = self.record.answer.strip().lower() in self.submitted.lower()
            return EvaluationResult(
                success=correct,
                score=1.0 if correct else 0.0,
                metrics={"judge": "substring-fallback", "question_type": self.record.question_type},
            )

        template = _judge_template_for(self.record.question_type, self.record.is_abstention)
        prompt = template.format(
            question=self.record.question, answer=self.record.answer, response=self.submitted
        )
        try:
            verdict = invoke_text(self.judge_model, [HumanMessage(content=prompt)]).strip().lower()
        except Exception as exc:
            return EvaluationResult(
                success=False, score=0.0, metrics={"judge_error": f"{type(exc).__name__}: {exc}"}
            )
        correct = verdict.startswith("yes")
        return EvaluationResult(
            success=correct,
            score=1.0 if correct else 0.0,
            metrics={
                "judge": "llm",
                "question_type": self.record.question_type,
                "is_abstention": self.record.is_abstention,
            },
        )

    def available_tools(self) -> list[dict[str, Any]]:
        return [
            tool_schema("list_sessions", "List all chat session ids and dates available."),
            tool_schema(
                "read_session",
                "Read the full text of one chat session.",
                {"session_id": {"type": "string"}},
                required=["session_id"],
            ),
            tool_schema(
                "search_sessions",
                "Find sessions whose text contains a substring (case-insensitive).",
                {"query": {"type": "string"}},
                required=["query"],
            ),
            tool_schema(
                "submit_answer",
                "Submit your final answer to the question.",
                {"text": {"type": "string"}},
                required=["text"],
            ),
        ]

    def get_env_state(self) -> dict[str, Any]:
        return {"question_id": self.record.question_id, "submitted": self.submitted}

    def close(self) -> None:
        return None


@register_benchmark("longmemeval")
class LongMemEvalBenchmark(BenchmarkAdapter):
    """One task per question; ``eval`` split is a fixed held-out slice.

    ``eval_fraction`` picks a stable, seed-independent slice of the loaded
    questions for the ``eval`` split -- the sample ordering in the source
    file has no train/eval distinction of its own, so one has to be imposed
    to keep validation honest across generations, same discipline as
    :meth:`BenchmarkAdapter.sample`.
    """

    def __init__(
        self,
        data_path: str | Path,
        judge_model: BaseChatModel | None = None,
        eval_fraction: float = 0.2,
        limit: int | None = None,
        seed: int = 0,
    ) -> None:
        self.description = "LongMemEval: long-term interactive memory for chat assistants."
        self.judge_model = judge_model
        # ``limit`` keeps every question type represented, so a budgeted run
        # still reports all six categories rather than whichever came first.
        self.records = stratified_subset(
            load_longmemeval(data_path), limit, seed, key=lambda r: r.question_type
        )
        self._by_id = {r.question_id: r for r in self.records}
        n_eval = max(1, int(round(len(self.records) * eval_fraction))) if self.records else 0
        # Stratified, and seeded independently of ``seed``: the held-out set
        # must cover every question type and must not move when the training
        # subset is re-drawn, or two runs' eval numbers are not comparable.
        self._eval_ids = {
            r.question_id
            for r in stratified_subset(self.records, n_eval, 0, key=lambda r: r.question_type)
        }

    def task_ids(self, split: str = "train") -> list[str]:
        if split == "all":
            return list(self._by_id)
        if split == "eval":
            return [qid for qid in self._by_id if qid in self._eval_ids]
        return [qid for qid in self._by_id if qid not in self._eval_ids]

    def make_env(self, task_id: str, curriculum_level: float = 0.0, seed: int = 0) -> LongMemEvalEnv:
        record = self._by_id.get(task_id)
        if record is None:
            raise KeyError(f"unknown question_id {task_id!r}")
        return LongMemEvalEnv(record, judge_model=self.judge_model)

    def instruction_for(self, task_id: str) -> str:
        record = self._by_id.get(task_id)
        return record.question if record else task_id

    def specs(self) -> list[TaskSpec]:
        return [
            TaskSpec(
                task_id=r.question_id,
                instruction=r.question,
                split="eval" if r.question_id in self._eval_ids else "train",
                tags=[r.question_type] + (["abstention"] if r.is_abstention else []),
            )
            for r in self.records
        ]


__all__ = [
    "LongMemEvalBenchmark",
    "LongMemEvalEnv",
    "LongMemEvalRecord",
    "load_longmemeval",
]
