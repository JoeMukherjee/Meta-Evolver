"""Turn on-disk run artifacts into the types the offline evaluators consume.

Formats handled (inspected from real ALFWorld runs; every field is read
defensively, a malformed row is skipped and counted, never raised):

* ``corpus/traces.jsonl`` -- one episode per line: ``steps`` (each with
  ``raw_action``/``filtered_action`` ``{name, kwargs}`` and
  ``raw_observation``/``filtered_observation`` ``{text, data}``), ``success``,
  ``final_reward``, ``error``, ``rollout_idx``. The instruction is not a
  top-level field; it is recovered from the first observation's
  ``data.goal_text``. These files record **no** memory citations.
* ``banks/*.jsonl`` -- one memory per line: ``title``, ``description``,
  ``content``, ``embedding`` (ignored), ``source.{task_id, task_query, ...}``.
  Memories have no id field; one is derived as ``"<condition>/<title>"``.
* ``eval/round*/<cond>_eval_<split>.jsonl`` -- one episode per line with
  ``seed``, ``success`` and ``retrieved_titles`` (the *only* place real
  citations are recorded) but no instruction or task id.

All file contents are treated as untrusted data and only ever parsed as JSON.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from meta_evolver.core.types import Action, StepRecord, Trajectory


def iter_jsonl(path: str | Path, stats: Counter | None = None) -> Iterator[dict]:
    """Yield dict rows; count unreadable / non-dict lines in ``stats['bad_rows']``."""
    stats = stats if stats is not None else Counter()
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError:
        stats["unreadable_files"] += 1
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                stats["bad_rows"] += 1
                continue
            if isinstance(row, dict):
                yield row
            else:
                stats["bad_rows"] += 1


def task_key(instruction: str) -> str:
    """Stable task id from instruction text (corpus rows share one constant id)."""
    norm = re.sub(r"\s+", " ", instruction.strip().lower())
    return "alfworld-" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]


def _text(obj: Any) -> str:
    if isinstance(obj, dict):
        t = obj.get("text", "")
        return t if isinstance(t, str) else ""
    return obj if isinstance(obj, str) else ""


def _instruction(steps: list[dict]) -> str:
    for s in steps:
        for key in ("raw_observation", "filtered_observation"):
            obs = s.get(key)
            if isinstance(obs, dict):
                data = obs.get("data")
                if isinstance(data, dict) and isinstance(data.get("goal_text"), str) and data["goal_text"].strip():
                    return data["goal_text"].strip()
                m = re.search(r"Your task is to:[^\n]*", _text(obs))
                if m:
                    return m.group(0).strip()
    return ""


def _step(i: int, raw: dict) -> StepRecord | None:
    act = raw.get("filtered_action") or raw.get("raw_action")
    if not isinstance(act, dict) or not isinstance(act.get("name"), str):
        return None
    kwargs = act.get("kwargs") if isinstance(act.get("kwargs"), dict) else {}
    obs = _text(raw.get("filtered_observation")) or _text(raw.get("raw_observation"))
    reward = raw.get("filtered_reward", raw.get("raw_reward", 0.0))
    return StepRecord(
        step_idx=i,
        thought=str(raw.get("policy_raw_response") or "")[:2000],
        action=Action(name=act["name"], kwargs=kwargs),
        observation=obs,
        reward=float(reward) if isinstance(reward, (int, float)) else 0.0,
        terminated=bool(raw.get("terminated")),
        truncated=bool(raw.get("truncated")),
        blocked=bool(raw.get("blocked_reason")),
    )


def load_corpus_trajectories(path: str | Path, benchmark: str = "alfworld") -> tuple[list[Trajectory], Counter]:
    """Parse a ``traces.jsonl``. ``retrieved_memory_ids`` is left empty: the
    file does not record which memories were shown."""
    stats: Counter = Counter()
    out: list[Trajectory] = []
    for row in iter_jsonl(path, stats):
        try:
            raw_steps = [s for s in (row.get("steps") or []) if isinstance(s, dict)]
            steps = [r for r in (_step(i, s) for i, s in enumerate(raw_steps)) if r is not None]
            instr = _instruction(raw_steps)
            err = row.get("error")
            reward = row.get("final_reward", 0.0)
            reward = float(reward) if isinstance(reward, (int, float)) else 0.0
            if not instr:
                stats["no_instruction"] += 1
            out.append(
                Trajectory(
                    task_id=task_key(instr) if instr else f"unknown-{row.get('episode_id', len(out))}",
                    benchmark=benchmark,
                    instruction=instr,
                    steps=steps,
                    success=bool(row.get("success")),
                    score=reward,
                    final_reward=reward,
                    error=str(err) if err else "",
                    rollout_index=int(row.get("rollout_idx") or 0),
                )
            )
            stats["loaded"] += 1
            if out[-1].error:
                stats["infra_error"] += 1
        except Exception:  # noqa: BLE001 - untrusted rows must never abort a load
            stats["bad_rows"] += 1
    return out, stats


def load_memory_bank(path: str | Path, condition: str | None = None) -> tuple[dict[str, str], dict[str, dict], Counter]:
    """Return ``({memory_id: text}, {memory_id: {task_key, task_query, title}}, stats)``."""
    stats: Counter = Counter()
    texts: dict[str, str] = {}
    meta: dict[str, dict] = {}
    cond = condition or Path(path).stem
    for row in iter_jsonl(path, stats):
        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            stats["bad_rows"] += 1
            continue
        body = " ".join(str(row.get(k) or "") for k in ("title", "description", "content")).strip()
        src = row.get("source") if isinstance(row.get("source"), dict) else {}
        query = src.get("task_query") if isinstance(src.get("task_query"), str) else ""
        mid = f"{cond}/{title.strip()}"
        if mid in texts:
            stats["duplicate_ids"] += 1
            continue
        texts[mid] = body
        meta[mid] = {"title": title.strip(), "task_query": query, "task_key": task_key(query) if query else ""}
        stats["loaded"] += 1
    return texts, meta, stats


def load_eval_citations(path: str | Path) -> tuple[list[dict], Counter]:
    """Real per-episode citations: ``{seed, split, success, retrieved_titles}``.

    These carry no instruction, so they cannot feed ``evaluate_ood_recall``.
    """
    stats: Counter = Counter()
    rows: list[dict] = []
    for row in iter_jsonl(path, stats):
        titles = row.get("retrieved_titles")
        if not isinstance(titles, list):
            stats["bad_rows"] += 1
            continue
        rows.append(
            {
                "seed": row.get("seed"),
                "split": row.get("split", ""),
                "success": bool(row.get("success")),
                "retrieved_titles": [t for t in titles if isinstance(t, str)],
            }
        )
        stats["loaded"] += 1
    return rows, stats


def discover_runs(runs_root: str | Path, prefix: str = "alfworld_headline_") -> list[Path]:
    root = Path(runs_root)
    return sorted(p for p in root.glob(prefix + "*") if p.is_dir()) if root.is_dir() else []
