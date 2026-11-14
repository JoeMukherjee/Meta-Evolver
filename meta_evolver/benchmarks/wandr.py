"""WANDR integration -- Perplexity's wide-and-deep research benchmark.

https://github.com/perplexityai/wandr   (technical report: arXiv 2608.14747)
https://huggingface.co/datasets/perplexity-ai/wandr

WANDR tasks are structured, high-volume research jobs: broad discovery,
enrichment, extraction, entity disambiguation, and an evidence-backed answer
written to *files* (``required_output_files``). The official score comes from
a task-local verifier that fetches the cited pages, normalizes and
deduplicates entities, and calls a judge model -- it needs Docker, the Harbor
runner, and paid provider keys, so it is run with ``./scripts/wandr`` in the
official repo.

What this adapter is for
------------------------
Meta-Evolver's evolution loop needs a cheap, deterministic per-episode signal
to select prompts and memories on. This module gives it:

* :func:`load_wandr` / :func:`fetch_wandr` -- read the public ``test`` /
  ``smoke`` JSONL (task id, instruction, required output files).
* :class:`WandrEnv` -- a file-workspace environment (``write_file``,
  ``read_file``, ``list_files``, ``finish``) in which the agent produces the
  task's required files.
* A **proxy** score: the fraction of required files that are present,
  non-empty, and well-formed for their extension (``.json``, ``.jsonl``,
  ``.csv``). It measures *output-contract compliance* -- the failure mode
  that zeroes an official run -- and is NOT the WANDR score. Every result
  carries ``metrics["proxy"] = True`` so no report can mistake it for one.

Reported WANDR numbers must come from the official verifier. Use
:meth:`WandrEnv.export_workspace` to hand the produced files to it.

Search is deliberately not simulated here: the agent supplies its own
retrieval (web tools, the memory bank). The env only owns the workspace.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from meta_evolver.benchmarks.base import BenchmarkAdapter, tool_schema
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

HF_BASE = "https://huggingface.co/datasets/perplexity-ai/wandr/resolve/main/data"
_SPLIT_FILES = {"test": "test.jsonl", "smoke": "smoke.jsonl"}

#: Soft cap so a runaway agent cannot fill memory through ``write_file``.
MAX_FILE_BYTES = 2_000_000


class WandrTask:
    """One WANDR task row (only the fields the engine needs)."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.task_id: str = str(raw["task_id"])
        self.instruction: str = str(raw.get("instruction") or "")
        self.required_files: list[str] = [str(p) for p in (raw.get("required_output_files") or [])]
        self.scored: bool = bool(raw.get("scored", True))
        self.instruction_status: str = str(raw.get("instruction_status") or "")


def load_wandr(path: str | Path, scored_only: bool = True) -> list[WandrTask]:
    """Parse a WANDR ``test.jsonl`` / ``smoke.jsonl``."""
    tasks: list[WandrTask] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        task = WandrTask(json.loads(line))
        if scored_only and not task.scored:
            continue
        tasks.append(task)
    return tasks


def fetch_wandr(split: str = "test", cache_dir: str | Path | None = None) -> Path:
    """Download one public split to ``cache_dir`` (no API key needed)."""
    if split not in _SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(_SPLIT_FILES)}, got {split!r}")
    cache = Path(cache_dir or Path.home() / ".cache" / "meta_evolver" / "wandr")
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / _SPLIT_FILES[split]
    if not dest.exists():
        url = f"{HF_BASE}/{_SPLIT_FILES[split]}"
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https host
            dest.write_bytes(resp.read())
    return dest


# ---------------------------------------------------------------------------
# Output-contract checks (the proxy score)
# ---------------------------------------------------------------------------


def _safe_rel(path: str) -> str | None:
    """Normalise an agent-supplied path; ``None`` if it escapes the workspace."""
    p = PurePosixPath(path.replace("\\", "/").lstrip("/"))
    if not p.parts or p.is_absolute() or ".." in p.parts:
        return None
    return str(p)


def check_file(path: str, content: str) -> tuple[bool, str]:
    """``(well_formed, reason)`` for one produced file, by extension."""
    if not content.strip():
        return False, "empty"
    suffix = PurePosixPath(path).suffix.lower()
    try:
        if suffix == ".json":
            json.loads(content)
        elif suffix == ".jsonl":
            for line in content.splitlines():
                if line.strip():
                    json.loads(line)
        elif suffix == ".csv":
            rows = list(csv.reader(io.StringIO(content)))
            if len(rows) < 2:
                return False, "csv needs a header and at least one row"
    except (ValueError, csv.Error) as exc:
        return False, f"malformed: {exc}"
    return True, "ok"


class WandrEnv(ActionableEnv):
    """A file workspace in which the agent writes a task's required outputs."""

    env_type = "wandr"

    def __init__(self, task: WandrTask) -> None:
        self.task = task
        self.files: dict[str, str] = {}
        self.finished = False

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> EnvResetResponse:
        self.files = {}
        self.finished = False
        required = "\n".join(f"- {p}" for p in self.task.required_files) or "(none declared)"
        return EnvResetResponse(
            observation=Observation(
                text=(
                    "Write every required output file with write_file, then call finish.\n"
                    f"Required files:\n{required}"
                )
            ),
            info={"instruction": self.task.instruction, "title": self.task.task_id},
        )

    def step(self, action: Action) -> EnvResponse:
        name, kwargs = action.name, dict(action.kwargs or {})

        if name == "write_file":
            rel = _safe_rel(str(kwargs.get("path", "")))
            content = str(kwargs.get("content", ""))
            if rel is None:
                return EnvResponse(observation=Observation(text="invalid path (must stay inside the workspace)"))
            if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                return EnvResponse(observation=Observation(text=f"file too large (> {MAX_FILE_BYTES} bytes)"))
            self.files[rel] = content
            ok, reason = check_file(rel, content)
            return EnvResponse(observation=Observation(text=f"wrote {rel} ({len(content)} chars): {reason if not ok else 'ok'}"))

        if name == "read_file":
            rel = _safe_rel(str(kwargs.get("path", "")))
            text = self.files.get(rel or "", f"no such file {kwargs.get('path')!r}")
            return EnvResponse(observation=Observation(text=text))

        if name == "list_files":
            text = "\n".join(f"{p} ({len(c)} chars)" for p, c in sorted(self.files.items()))
            return EnvResponse(observation=Observation(text=text or "(workspace empty)"))

        if name == "finish":
            self.finished = True
            return EnvResponse(observation=Observation(text="submitted"), terminated=True)

        return EnvResponse(
            observation=Observation(text=f"unknown tool {name!r}"),
            info={"result": {"error": f"unknown tool {name!r}"}},
        )

    def observe(self) -> Observation:
        return Observation(text=f"{len(self.files)} file(s) written")

    def evaluate(self) -> EvaluationResult:
        required = {_safe_rel(p) or p: p for p in self.task.required_files}
        if not required:
            return EvaluationResult(
                success=False,
                score=0.0,
                metrics={"proxy": True, "reason": "task declares no required output files"},
            )
        good, problems = 0, {}
        for rel in required:
            if rel not in self.files:
                problems[rel] = "missing"
                continue
            ok, reason = check_file(rel, self.files[rel])
            if ok:
                good += 1
            else:
                problems[rel] = reason
        score = good / len(required)
        return EvaluationResult(
            success=good == len(required),
            score=score,
            metrics={
                "proxy": True,  # contract compliance, NOT the official WANDR score
                "required": len(required),
                "valid": good,
                "problems": problems,
            },
        )

    def export_workspace(self, dest: str | Path) -> Path:
        """Write the produced files under ``dest`` for the official verifier."""
        root = Path(dest)
        for rel, content in self.files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return root

    def available_tools(self) -> list[dict[str, Any]]:
        return [
            tool_schema(
                "write_file",
                "Create or overwrite a file in the workspace.",
                {"path": {"type": "string"}, "content": {"type": "string"}},
                required=["path", "content"],
            ),
            tool_schema("read_file", "Read a workspace file.", {"path": {"type": "string"}}, required=["path"]),
            tool_schema("list_files", "List workspace files."),
            tool_schema("finish", "Submit the workspace for scoring."),
        ]

    def get_env_state(self) -> dict[str, Any]:
        return {"task_id": self.task.task_id, "files": sorted(self.files)}

    def close(self) -> None:
        return None


def _is_eval(task_id: str, eval_fraction: float) -> bool:
    """Stable, order-independent holdout: hash of the id, not its position."""
    h = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)
    return (h % 10_000) / 10_000 < eval_fraction


@register_benchmark("wandr")
class WandrBenchmark(BenchmarkAdapter):
    """WANDR tasks with a stable hash-based train/eval split."""

    def __init__(
        self,
        data_path: str | Path | None = None,
        eval_fraction: float = 0.3,
        limit: int | None = None,
        scored_only: bool = True,
        split: str = "test",
    ) -> None:
        self.description = "WANDR: wide-and-deep research (proxy contract score; verify with the official harness)."
        path = Path(data_path) if data_path else fetch_wandr(split)
        tasks = load_wandr(path, scored_only=scored_only)
        if limit is not None:
            tasks = sorted(tasks, key=lambda t: t.task_id)[:limit]
        self.eval_fraction = eval_fraction
        self._tasks: dict[str, WandrTask] = {t.task_id: t for t in tasks}

    def task_ids(self, split: str = "train") -> list[str]:
        if split == "all":
            return list(self._tasks)
        is_eval = split == "eval"
        return [t for t in self._tasks if _is_eval(t, self.eval_fraction) == is_eval]

    def make_env(self, task_id: str, curriculum_level: float = 0.0, seed: int = 0) -> WandrEnv:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task_id {task_id!r}")
        return WandrEnv(task)

    def instruction_for(self, task_id: str) -> str:
        task = self._tasks.get(task_id)
        return task.instruction if task else task_id

    def specs(self) -> list[TaskSpec]:
        return [
            TaskSpec(
                task_id=tid,
                instruction=t.instruction,
                split="eval" if _is_eval(tid, self.eval_fraction) else "train",
                tags=["wandr"],
            )
            for tid, t in self._tasks.items()
        ]
