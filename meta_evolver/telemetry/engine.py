"""Run telemetry -- JSONL streams plus a readable summary.

Written for post-hoc debugging of a long run, which means two things:

* Episodes stream to disk as they finish, one JSON object per line. A run that
  dies in generation four still leaves generations one to three on disk.
* Every record carries the identifiers needed to join it against the others --
  generation, task, prompt version, curriculum level, and the memory ids that
  were retrieved. The question you actually want to answer later is "which
  memory was in the prompt when this started failing", and that join is only
  possible if the ids were written down at the time.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from meta_evolver.core.types import GenerationReport, Trajectory


class TelemetryEngine:
    """Streams episodes and generation reports to a run directory."""

    def __init__(self, run_dir: str | Path = "runs/latest", enabled: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.enabled = bool(enabled)
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.reports_path = self.run_dir / "generations.jsonl"
        self.started = time.time()
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    # -- writes ------------------------------------------------------------

    def log_episode(self, trajectory: Trajectory, curriculum_level: float = 0.0) -> None:
        if not self.enabled:
            return
        record = {
            "ts": time.time(),
            "generation": trajectory.generation,
            "task_id": trajectory.task_id,
            "benchmark": trajectory.benchmark,
            "success": trajectory.success,
            "score": trajectory.score,
            "n_steps": trajectory.n_steps,
            "duration_ms": trajectory.duration_ms,
            "error": trajectory.error,
            "prompt_version": trajectory.prompt_version,
            "curriculum_level": curriculum_level,
            "retrieved_memory_ids": trajectory.retrieved_memory_ids,
            "memory_evicted_at": trajectory.memory_evicted_at,
            "metrics": trajectory.metrics,
            "actions": [s.action.render() for s in trajectory.steps],
            "n_perturbed": sum(1 for s in trajectory.steps if s.perturbed),
            "n_blocked": sum(1 for s in trajectory.steps if s.blocked),
        }
        _append(self.episodes_path, record)

    def log_generation(self, report: GenerationReport) -> None:
        if not self.enabled:
            return
        _append(self.reports_path, report.model_dump())

    def save_summary(self, extra: dict[str, Any] | None = None) -> Path | None:
        if not self.enabled:
            return None
        path = self.run_dir / "summary.json"
        reports = self.read_generations()
        payload = {
            "run_dir": str(self.run_dir),
            "duration_s": time.time() - self.started,
            "generations": reports,
            "final_pass_rate": reports[-1]["pass_rate"] if reports else None,
            "best_pass_rate": max((r["pass_rate"] for r in reports), default=None),
            **(extra or {}),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    # -- reads -------------------------------------------------------------

    def read_generations(self) -> list[dict[str, Any]]:
        return list(_read(self.reports_path))

    def read_episodes(self) -> list[dict[str, Any]]:
        return list(_read(self.episodes_path))

    # -- rendering ---------------------------------------------------------

    @staticmethod
    def render_progress(reports: Sequence[GenerationReport]) -> str:
        """A compact per-generation table.

        Shows the curriculum level next to the pass rate on purpose: a flat
        rate at a rising difficulty is progress, and a table that omits the
        difficulty makes it look like a stall.
        """
        if not reports:
            return "(no generations recorded)"
        header = (
            f"{'gen':>3} {'pass':>6} {'steps':>6} {'score':>6} "
            f"{'mem':>10} {'prompt':>8} {'curr':>5}  notes"
        )
        rows = [header, "-" * len(header)]
        for r in reports:
            mem = f"{r.memories_before}+{r.memories_added}-{r.memories_pruned}"
            note = "; ".join(r.notes)[:60]
            rows.append(
                f"{r.generation:>3} {r.pass_rate * 100:>5.1f}% {r.avg_steps:>6.2f} "
                f"{r.avg_score:>6.2f} {mem:>10} {r.prompt_version:>8} "
                f"{r.curriculum_level:>5.2f}  {note}"
            )
        first, last = reports[0], reports[-1]
        rows.append("-" * len(header))
        rows.append(
            f"delta: pass {(last.pass_rate - first.pass_rate) * 100:+.1f} pts, "
            f"steps {last.avg_steps - first.avg_steps:+.2f}, "
            f"curriculum {last.curriculum_level - first.curriculum_level:+.2f}"
        )
        return "\n".join(rows)


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # A partial final line is normal after a kill; skip it rather than
            # failing the whole read.
            continue
