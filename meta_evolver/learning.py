"""The per-generation learning step for the graph and skill layers.

One object, two hooks, so the evolution graph needs exactly two call sites and
the rest of the engine stays ignorant of either layer:

* :meth:`GenerationLearner.credit` -- right after the bank's credit pass.
  Charges each skill that was in a prompt with the episode's outcome.
* :meth:`GenerationLearner.learn` -- after prune. Writes the generation's
  episodes into the knowledge graph, mines new skill candidates, runs the
  skill lifecycle, and every ``repair_every`` generations validates and
  repairs the graph.

Every method swallows its own exceptions into ``self.errors``: these layers
are additive, and a bug in one must never cost a user a generation of paid
rollouts.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from meta_evolver.core.types import Trajectory
from meta_evolver.knowledge import (
    GraphIngestor,
    GraphMemory,
    KnowledgeGraph,
    Ontology,
    agent_ontology,
    extract_concepts,
    health,
    repair,
)
from meta_evolver.skills import SkillLibrary


class GenerationLearner:
    def __init__(
        self,
        graph: KnowledgeGraph | None = None,
        skills: SkillLibrary | None = None,
        ontology: Ontology | None = None,
        repair_every: int = 1,
        mine_skills: bool = True,
        lift: bool = False,
    ) -> None:
        self.graph = graph
        self.skills = skills
        self.ontology = ontology or agent_ontology()
        self.ingestor = GraphIngestor(graph, self.ontology) if graph is not None else None
        self.lift = lift
        """Opt-in: induce ``similar_to``/``is_a`` links between concepts each
        generation and let retrieval transfer evidence across them. Off by
        default -- it is only worth enabling where an OOD recall measurement
        says it helps."""
        self.graph_memory = GraphMemory(graph, transfer=lift) if graph is not None else None
        self.repair_every = repair_every
        self.mine_skills = mine_skills
        self.errors: list[str] = []
        self.history: list[dict[str, Any]] = []

    @classmethod
    def default(
        cls,
        directory: str | Path | None = None,
        embed: Any = None,
        graph: bool = True,
        skills: bool = True,
        lift: bool = False,
    ) -> "GenerationLearner":
        """Graph and skill library persisted under ``directory`` (in memory if
        ``None``), ready to hand to :class:`~meta_evolver.core.evolver.MetaEvolver`."""
        d = Path(directory) if directory else None
        return cls(
            graph=KnowledgeGraph(d / "knowledge.sqlite" if d else None) if graph else None,
            skills=SkillLibrary(d / "skills.json" if d else None, embed=embed) if skills else None,
            lift=lift,
        )

    # -- hooks ---------------------------------------------------------------

    def credit(self, trajectories: Sequence[Trajectory], generation: int, gamma: float = 1.0) -> None:
        if self.skills is None:
            return
        try:
            self.skills.credit(
                [
                    (t.retrieved_skill_ids, t.success, t.benchmark or None, t.task_id,
                     extract_concepts(t.instruction))
                    for t in trajectories
                    if t.generation == generation and t.usable and t.retrieved_skill_ids
                ],
                generation=generation,
                gamma=gamma,
            )
        except Exception as exc:  # noqa: BLE001 - additive layer, never fatal
            self.errors.append(f"skill credit: {exc!r}")

    def learn(self, trajectories: Sequence[Trajectory], generation: int) -> dict[str, Any]:
        this_gen = [t for t in trajectories if t.generation == generation]
        out: dict[str, Any] = {"generation": generation}
        if self.ingestor is not None:
            try:
                out["graph"] = self.ingestor.ingest(this_gen, generation)
                if self.lift:
                    from meta_evolver.knowledge import induce_concept_links

                    rep = induce_concept_links(self.graph, generation)
                    out["lift"] = {k: v for k, v in vars(rep).items() if isinstance(v, int)}
                if self.repair_every > 0 and (generation + 1) % self.repair_every == 0:
                    report = repair(self.graph, self.ontology, generation=generation)
                    out["repair"] = {"fixed": report.fixed, "unresolved": len(report.unresolved)}
                out["health"] = health(self.graph, self.ontology)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"graph: {exc!r}")
        if self.skills is not None:
            try:
                created = self.skills.induce(this_gen, generation) if self.mine_skills else []
                out["skills_created"] = len(created)
                out["skills"] = {**self.skills.lifecycle(generation), **self.skills.stats()}
                out["skills_refined"] = len(self.skills.refine(generation))
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"skills: {exc!r}")
        self.history.append(out)
        return out
