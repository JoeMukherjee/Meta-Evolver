"""LoCoMo -- Evaluating Very Long-Term Conversational Memory of LLM Agents.

Maharana, Lee, Tulyakov, Bansal, Barbieri & Fang, ACL 2024.
https://github.com/snap-research/locomo

Ten conversations, each ~300 turns across dozens of dated sessions, with
1,540 annotated questions across five categories. Unlike LongMemEval the
dataset ships *in* the official repo (``data/locomo10.json``, ~3MB) rather
than needing a separate download.

Same shape as :mod:`meta_evolver.benchmarks.longmemeval`: the agent pages
through sessions with tools (``list_sessions``/``read_session``/
``search_sessions``) rather than getting the whole conversation dumped into
one turn, then ``submit_answer``. Scoring reproduces LoCoMo's own published
metric -- Porter-stemmed token F1 against the normalized ground truth,
per-category exactly as ``task_eval/evaluation.py`` computes it in the
official repo -- rather than a homegrown rubric, for the same reason as
LongMemEval: comparability with published numbers.

Category semantics (LoCoMo's own numbering, preserved as-is):

    1  multi-hop      -- comma-split sub-answers, max F1 per sub-answer, averaged
    2  temporal       -- single F1
    3  open-domain    -- ground truth truncated to before ';', single F1
    4  single-hop     -- single F1
    5  adversarial    -- correct iff the response refuses ("no information
                          available" / "not mentioned") -- there is no answer

F1 uses Porter stemming when ``nltk`` is installed (matching the official
metric exactly); without it, scoring falls back to unstemmed token overlap,
which is very close but not bit-identical to published numbers -- a
documented approximation, not a silent divergence.

::

    from meta_evolver.benchmarks.locomo import LoCoMoBenchmark
    bench = LoCoMoBenchmark(data_path="repos/locomo/data/locomo10.json")
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter
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

_CATEGORY_NAMES: dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

try:  # pragma: no cover - exercised indirectly by whichever env has nltk
    from nltk.stem import PorterStemmer

    _stemmer: Any = PorterStemmer()
except Exception:  # pragma: no cover
    _stemmer = None


def _stem(word: str) -> str:
    return _stemmer.stem(word) if _stemmer is not None else word


def normalize_answer(text: str) -> str:
    """LoCoMo's own normalization: lowercase, strip punctuation, drop
    articles/'and', collapse whitespace -- copied faithfully from
    ``task_eval/evaluation.py``."""
    text = text.replace(",", "")
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    """Stemmed token F1 between one prediction and one ground truth string."""
    pred_tokens = [_stem(w) for w in normalize_answer(prediction).split()]
    gt_tokens = [_stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    """Category 1 (multi-hop): comma-split sub-answers, max F1 per ground
    truth sub-answer against any predicted sub-answer, averaged."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    if not ground_truths:
        return 0.0
    scores = [max(f1_score(p, gt) for p in predictions) for gt in ground_truths]
    return sum(scores) / len(scores)


_REFUSAL_PHRASES = ("no information available", "not mentioned")


def score_locomo_answer(category: int, response: str, ground_truth: str) -> float:
    """Reproduces ``eval_question_answering`` in the official repo exactly."""
    if category == 5:
        low = response.lower()
        return 1.0 if any(p in low for p in _REFUSAL_PHRASES) else 0.0
    if category == 1:
        return multi_answer_f1(response, ground_truth)
    if category == 3:
        ground_truth = ground_truth.split(";")[0].strip()
    return f1_score(response, ground_truth)


class LoCoMoConversation:
    """One parsed conversation sample (10 total in the official release)."""

    __slots__ = ("sample_id", "speaker_a", "speaker_b", "sessions", "qa")

    def __init__(self, raw: dict[str, Any]) -> None:
        self.sample_id: str = str(raw["sample_id"])
        conv = raw["conversation"]
        self.speaker_a: str = conv.get("speaker_a", "")
        self.speaker_b: str = conv.get("speaker_b", "")

        session_keys = sorted(
            (k for k in conv if re.fullmatch(r"session_\d+", k)),
            key=lambda k: int(k.split("_")[1]),
        )
        self.sessions: list[dict[str, Any]] = []
        for key in session_keys:
            turns = conv[key]
            date = conv.get(f"{key}_date_time", "")
            text = "\n".join(f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns)
            self.sessions.append({"id": key, "date": date, "text": text})

        self.qa: list[dict[str, Any]] = list(raw.get("qa", []))


def load_locomo(data_path: str | Path) -> list[LoCoMoConversation]:
    """Parse the official ``locomo10.json``."""
    raw = json.loads(Path(data_path).read_text(encoding="utf-8"))
    return [LoCoMoConversation(r) for r in raw]


class LoCoMoTaskRef:
    """One (conversation, question) pair -- the unit meta-evolver treats as a task."""

    def __init__(self, conversation: LoCoMoConversation, qa_index: int) -> None:
        self.conversation = conversation
        qa = conversation.qa[qa_index]
        self.task_id = f"{conversation.sample_id}:qa{qa_index}"
        self.question = str(qa["question"])
        # Category 5 (adversarial) items store their answer under
        # ``adversarial_answer`` in the official release, not ``answer`` --
        # confirmed against the real dataset (444 of 446 category-5 items).
        self.answer = str(qa.get("answer", qa.get("adversarial_answer", "")))
        self.category = int(qa.get("category", 4))
        self.evidence: list[str] = list(qa.get("evidence", []))


class LoCoMoEnv(ActionableEnv):
    """Page through one conversation's sessions, then answer one question."""

    env_type = "locomo"

    def __init__(self, task: LoCoMoTaskRef) -> None:
        self.task = task
        self.submitted: str | None = None
        self.finished = False

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> EnvResetResponse:
        self.submitted = None
        self.finished = False
        sessions = self.task.conversation.sessions
        previews = "\n".join(f"- {s['id']} ({s['date']}): {s['text'][:80]!r}..." for s in sessions)
        return EnvResetResponse(
            observation=Observation(
                text=(
                    f"Conversation between {self.task.conversation.speaker_a} and "
                    f"{self.task.conversation.speaker_b}, {len(sessions)} sessions.\n"
                    f"Use list_sessions/read_session/search_sessions, then submit_answer.\n\n{previews}"
                )
            ),
            info={"instruction": self.task.question, "title": self.task.question},
        )

    def step(self, action: Action) -> EnvResponse:
        name, kwargs = action.name, dict(action.kwargs or {})
        sessions = self.task.conversation.sessions

        if name == "list_sessions":
            text = "\n".join(f"{s['id']} ({s['date']})" for s in sessions)
            return EnvResponse(observation=Observation(text=text or "(no sessions)"))

        if name == "read_session":
            sid = str(kwargs.get("session_id", ""))
            match = next((s for s in sessions if s["id"] == sid), None)
            text = match["text"] if match else f"unknown session_id {sid!r}"
            return EnvResponse(observation=Observation(text=text))

        if name == "search_sessions":
            query = str(kwargs.get("query", "")).lower()
            hits = [s["id"] for s in sessions if query in s["text"].lower()]
            text = f"sessions containing {query!r}: {hits}" if hits else "no matches"
            return EnvResponse(observation=Observation(text=text))

        if name == "submit_answer":
            self.submitted = str(kwargs.get("text", ""))
            self.finished = True
            return EnvResponse(observation=Observation(text="answer submitted"), terminated=True)

        return EnvResponse(
            observation=Observation(text=f"unknown tool {name!r}"),
            info={"result": {"error": f"unknown tool {name!r}"}},
        )

    def observe(self) -> Observation:
        return Observation(text=self.submitted or "")

    def evaluate(self) -> EvaluationResult:
        if self.submitted is None:
            return EvaluationResult(success=False, score=0.0, metrics={"reason": "no answer submitted"})
        score = score_locomo_answer(self.task.category, self.submitted, self.task.answer)
        return EvaluationResult(
            success=score >= 0.5,
            score=score,
            metrics={
                "category": self.task.category,
                "category_name": _CATEGORY_NAMES.get(self.task.category, "unknown"),
                "stemmed": _stemmer is not None,
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
        return {"task_id": self.task.task_id, "submitted": self.submitted}

    def close(self) -> None:
        return None


@register_benchmark("locomo")
class LoCoMoBenchmark(BenchmarkAdapter):
    """One task per (conversation, question) pair.

    ``eval_fraction`` holds out a stable slice of the 10 conversations
    (whole conversations, not individual questions, so no session context a
    training-split question relies on ever leaks into the eval split) --
    same discipline as :meth:`BenchmarkAdapter.sample`.
    """

    def __init__(
        self,
        data_path: str | Path,
        eval_fraction: float = 0.2,
        limit: int | None = None,
        seed: int = 0,
    ) -> None:
        self.description = "LoCoMo: very long-term conversational memory QA."
        self.conversations = load_locomo(data_path)
        n_eval = max(1, int(round(len(self.conversations) * eval_fraction))) if self.conversations else 0
        self._eval_sample_ids = {c.sample_id for c in self.conversations[:n_eval]}

        refs = [LoCoMoTaskRef(conv, i) for conv in self.conversations for i in range(len(conv.qa))]
        # Stratified by (split, category): a budgeted run keeps all five
        # categories on *both* sides of the conversation-level eval split.
        refs = stratified_subset(
            refs,
            limit,
            seed,
            key=lambda r: (r.conversation.sample_id in self._eval_sample_ids, r.category),
        )
        self._tasks: dict[str, LoCoMoTaskRef] = {ref.task_id: ref for ref in refs}

    def _in_eval(self, task_id: str) -> bool:
        return self._tasks[task_id].conversation.sample_id in self._eval_sample_ids

    def task_ids(self, split: str = "train") -> list[str]:
        if split == "all":
            return list(self._tasks)
        if split == "eval":
            return [tid for tid in self._tasks if self._in_eval(tid)]
        return [tid for tid in self._tasks if not self._in_eval(tid)]

    def make_env(self, task_id: str, curriculum_level: float = 0.0, seed: int = 0) -> LoCoMoEnv:
        ref = self._tasks.get(task_id)
        if ref is None:
            raise KeyError(f"unknown task_id {task_id!r}")
        return LoCoMoEnv(ref)

    def instruction_for(self, task_id: str) -> str:
        ref = self._tasks.get(task_id)
        return ref.question if ref else task_id

    def specs(self) -> list[TaskSpec]:
        return [
            TaskSpec(
                task_id=tid,
                instruction=ref.question,
                split="eval" if ref.conversation.sample_id in self._eval_sample_ids else "train",
                tags=[_CATEGORY_NAMES.get(ref.category, "unknown")],
            )
            for tid, ref in self._tasks.items()
        ]


__all__ = [
    "LoCoMoBenchmark",
    "LoCoMoConversation",
    "LoCoMoEnv",
    "LoCoMoTaskRef",
    "f1_score",
    "load_locomo",
    "multi_answer_f1",
    "normalize_answer",
    "score_locomo_answer",
]
