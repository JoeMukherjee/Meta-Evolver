"""Writes the causal graph to Neo4j while a run happens.

Design constraints, in the order they mattered:

**It must never break a run.** This is an observability surface. A Neo4j that
is down, slow, or misconfigured has to cost you the picture and nothing else,
so every write is wrapped and failures are counted rather than raised. The
count is reported at the end, because silently recording nothing is its own
kind of failure.

**It must be live.** The point of watching a run in the browser is watching it
*while it runs*. Writes go in as each generation completes rather than being
batched to the end, so a long run is legible from the first minute.

**It must be re-runnable.** Everything is ``MERGE``, keyed by run and id, so
re-recording a generation updates rather than duplicating. A resumed run that
replays a generation produces one graph, not two overlapping ones.

Cost is one round trip per generation, not per episode: all of a generation's
episodes, memories and edges go in a single ``UNWIND`` transaction.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from meta_evolver.core.types import GenerationReport, MemoryItem, Trajectory
from meta_evolver.graph_view.schema import SCHEMA_STATEMENTS
from meta_evolver.storage.base import redact

#: Read when no URL is passed, so a deployment can point every entry point at
#: one graph without touching a command line.
GRAPH_URL_ENV = "META_EVOLVER_GRAPH_URL"

_WRITE_GENERATION = """
MERGE (r:Run {id: $run_id})
  ON CREATE SET r.benchmark = $benchmark, r.started_at = timestamp()
  SET r.last_generation = $generation, r.updated_at = timestamp()

MERGE (g:Generation {run_id: $run_id, index: $generation})
  SET g += $generation_props
MERGE (r)-[:HAS_GENERATION]->(g)

MERGE (p:Prompt {run_id: $run_id, version: $prompt_version})
  ON CREATE SET p.text = $prompt_text, p.created_generation = $generation
  SET p.adopted = true
MERGE (g)-[:USED_PROMPT]->(p)

MERGE (l:CurriculumLevel {run_id: $run_id, level: $curriculum_level})
  SET l.name = $curriculum_name
MERGE (g)-[:AT_LEVEL]->(l)

WITH r, g
UNWIND $episodes AS ep
  MERGE (e:Episode {uid: ep.uid})
    SET e += ep
  MERGE (g)-[:RAN]->(e)
  MERGE (t:Task {benchmark: ep.benchmark, task_id: ep.task_id})
    ON CREATE SET t.instruction = ep.instruction
  MERGE (e)-[:ON_TASK]->(t)
  WITH r, g, e, ep
  UNWIND (CASE WHEN ep.retrieved = [] THEN [null] ELSE ep.retrieved END) AS mem_id
    FOREACH (_ IN CASE WHEN mem_id IS NULL THEN [] ELSE [1] END |
      MERGE (m:Memory {run_id: $run_id, id: mem_id})
      MERGE (e)-[:RETRIEVED]->(m)
    )
"""

_WRITE_MEMORIES = """
UNWIND $memories AS mem
  MERGE (m:Memory {run_id: $run_id, id: mem.id})
    SET m += mem.props
  WITH m, mem
  UNWIND (CASE WHEN mem.sources = [] THEN [null] ELSE mem.sources END) AS uid
    FOREACH (_ IN CASE WHEN uid IS NULL THEN [] ELSE [1] END |
      MERGE (e:Episode {uid: uid})
      MERGE (e)-[:INDUCED]->(m)
    )
"""

_WRITE_PRUNED = """
MATCH (g:Generation {run_id: $run_id, index: $generation})
UNWIND $ids AS mem_id
  MERGE (m:Memory {run_id: $run_id, id: mem_id})
    SET m.pruned_at = $generation
  MERGE (g)-[:PRUNED]->(m)
"""

_WRITE_CANDIDATES = """
MATCH (g:Generation {run_id: $run_id, index: $generation})
UNWIND $candidates AS cand
  MERGE (c:Prompt {run_id: $run_id, version: cand.version})
    SET c.text = cand.text,
        c.validation_score = cand.score,
        c.adopted = cand.adopted,
        c.created_generation = $generation
  MERGE (parent:Prompt {run_id: $run_id, version: cand.parent})
  MERGE (c)-[:PROPOSED_FROM]->(parent)
  FOREACH (_ IN CASE WHEN cand.adopted THEN [1] ELSE [] END |
    MERGE (g)-[:ADOPTED]->(c)
  )
"""


def episode_uid(run_id: str, trajectory: Trajectory) -> str:
    """Stable identity for one episode across re-records."""
    return (
        f"{run_id}:{trajectory.generation}:{trajectory.task_id}:{trajectory.rollout_index}"
    )


class CausalGraphRecorder:
    """Streams a run's causal graph into Neo4j."""

    def __init__(
        self,
        url: str | None = None,
        run_id: str = "run",
        benchmark: str = "",
        database: str | None = None,
        setup: bool = True,
    ) -> None:
        self.url = url or os.environ.get(GRAPH_URL_ENV, "")
        self.run_id = run_id
        self.benchmark = benchmark
        self.database = database
        self.driver: Any = None
        self.errors: list[str] = []
        self.n_writes = 0

        if not self.url:
            self.describe = "disabled"
            return

        try:
            from neo4j import GraphDatabase
        except ImportError:
            self.describe = "unavailable (pip install 'meta-evolver[graph]')"
            self.errors.append("neo4j driver not installed")
            return

        parsed = urlparse(self.url)
        auth = (
            (parsed.username, parsed.password)
            if parsed.username and parsed.password
            else None
        )
        # Credentials in the URL are convenient but the driver wants them
        # separately, and a URL carrying them must never reach a log.
        endpoint = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 7687}"

        try:
            self.driver = GraphDatabase.driver(endpoint, auth=auth)
            self.driver.verify_connectivity()
            if setup:
                self._apply_schema()
            self.describe = f"neo4j:{redact(self.url)}"
        except Exception as exc:
            self.driver = None
            self.describe = f"unreachable ({type(exc).__name__})"
            self.errors.append(f"connect: {exc}")

    @property
    def enabled(self) -> bool:
        return self.driver is not None

    def _apply_schema(self) -> None:
        with self.driver.session(database=self.database) as session:
            for statement in SCHEMA_STATEMENTS:
                session.run(statement)

    def _run(self, query: str, **params: Any) -> None:
        """Execute a write, recording rather than raising on failure.

        An observability backend must not be able to fail a run. The counter
        exists so "recorded nothing" is visible instead of silent.
        """
        if not self.enabled:
            return
        try:
            with self.driver.session(database=self.database) as session:
                session.run(query, **params)
            self.n_writes += 1
        except Exception as exc:
            if len(self.errors) < 10:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    # -- recording ---------------------------------------------------------

    def record_generation(
        self,
        report: GenerationReport,
        trajectories: Sequence[Trajectory],
        prompt_text: str,
        curriculum_name: str = "",
    ) -> None:
        """One generation: its episodes, tasks, prompt, level and retrievals."""
        if not self.enabled:
            return

        current = [t for t in trajectories if t.generation == report.generation]
        episodes = [
            {
                "uid": episode_uid(self.run_id, t),
                "run_id": self.run_id,
                "generation": t.generation,
                "task_id": t.task_id,
                "benchmark": t.benchmark or self.benchmark,
                "instruction": t.instruction,
                "rollout_index": t.rollout_index,
                "success": bool(t.success),
                "score": float(t.score),
                "steps": t.n_steps,
                "tokens": int(t.tokens),
                "duration_ms": float(t.duration_ms),
                "error": t.error,
                "memory_evicted_at": t.memory_evicted_at,
                "prompt_version": t.prompt_version,
                "retrieved": list(t.retrieved_memory_ids),
                # The label the browser shows on the node.
                "name": f"{t.task_id} g{t.generation}"
                + (f".{t.rollout_index}" if t.rollout_index else ""),
            }
            for t in current
        ]

        self._run(
            _WRITE_GENERATION,
            run_id=self.run_id,
            benchmark=self.benchmark,
            generation=report.generation,
            generation_props={
                "pass_rate": float(report.pass_rate),
                "avg_steps": float(report.avg_steps),
                "avg_score": float(report.avg_score),
                "n_tasks": report.n_tasks,
                "n_errors": report.n_errors,
                "regressions": report.regressions,
                "recoveries": report.recoveries,
                "tokens": report.tokens,
                "rollouts_per_task": report.rollouts_per_task,
                "memories_added": report.memories_added,
                "memories_pruned": report.memories_pruned,
                "prompt_version": report.prompt_version,
                "curriculum_level": float(report.curriculum_level),
                "duration_s": float(report.duration_s),
                "name": f"gen {report.generation}",
            },
            prompt_version=report.prompt_version,
            prompt_text=prompt_text[:4000],
            curriculum_level=float(report.curriculum_level),
            curriculum_name=curriculum_name,
            episodes=episodes,
        )

    def record_memories(
        self, memories: Sequence[MemoryItem], generation: int, induced: bool = True
    ) -> None:
        """Memory nodes, linked back to the episodes that produced them."""
        if not self.enabled or not memories:
            return
        payload = [
            {
                "id": m.id,
                "sources": (
                    [f"{self.run_id}:{generation}:{task}:0" for task in m.source_task_ids]
                    if induced
                    else []
                ),
                "props": {
                    "title": m.title,
                    "scenario": m.scenario,
                    "lesson": m.lesson,
                    "procedure": m.procedure,
                    "polarity": m.polarity,
                    "uses": m.uses,
                    "wins": m.wins,
                    "utility": round(m.utility, 4),
                    "created_generation": m.created_generation,
                    "name": m.title or m.scenario,
                },
            }
            for m in memories
        ]
        self._run(_WRITE_MEMORIES, run_id=self.run_id, memories=payload)

    def record_pruned(self, memories: Sequence[MemoryItem], generation: int) -> None:
        if not self.enabled or not memories:
            return
        self._run(
            _WRITE_PRUNED,
            run_id=self.run_id,
            generation=generation,
            ids=[m.id for m in memories],
        )

    def record_prompt_candidates(
        self, candidates: Sequence[Any], generation: int, adopted_version: str
    ) -> None:
        """Candidate prompts and their lineage, including the rejected ones.

        Rejections are the more interesting half: they are the record of what
        was tried and measured, which is what stops a later reader assuming
        the adopted prompt was the only idea anyone had.
        """
        if not self.enabled or not candidates:
            return
        payload = [
            {
                "version": c.version,
                "parent": c.parent,
                "text": (c.text or "")[:4000],
                "score": float(c.score) if c.score is not None else None,
                "adopted": c.version == adopted_version,
            }
            for c in candidates
        ]
        self._run(
            _WRITE_CANDIDATES, run_id=self.run_id, generation=generation, candidates=payload
        )

    # -- reading -----------------------------------------------------------

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Run a read query and return plain dicts. Empty when disabled."""
        if not self.enabled:
            return []
        params.setdefault("run_id", self.run_id)
        try:
            with self.driver.session(database=self.database) as session:
                return [record.data() for record in session.run(cypher, **params)]
        except Exception as exc:
            self.errors.append(f"query: {exc}")
            return []

    def browser_url(self) -> str:
        """Where to watch this run, derived from the bolt endpoint."""
        parsed = urlparse(self.url)
        host = parsed.hostname or "localhost"
        # The compose file maps browser to bolt-port minus 213 (7475 / 7688);
        # for anything else, fall back to the conventional 7474.
        port = 7475 if parsed.port == 7688 else 7474
        return f"http://{host}:{port}"

    def close(self) -> None:
        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:
                pass
            self.driver = None
