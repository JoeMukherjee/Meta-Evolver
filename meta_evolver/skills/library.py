"""The skill library: creation, retrieval, credit, and lifecycle management.

Lifecycle, and why each gate exists::

    candidate --(enough uses, win rate, >=2 distinct winning tasks, lcb)--> active
    active    --(enough uses and the *upper* bound is still poor)-------> deprecated
    candidate --(never earned a trial within ``stale_after`` generations)-> deprecated
    active    --(a promoted child version supersedes it)----------------> deprecated

* A fresh candidate is **not** trusted. It is offered in a small number of
  *trial slots* per retrieval so it can gather evidence, and is otherwise
  invisible to the agent. Without trial slots a candidate could never earn
  the uses that promote it (the cold-start trap); with unlimited trust a
  wrong skill poisons every prompt.
* Promotion needs wins on **distinct tasks**. A procedure that succeeded
  three times on one task has shown it can repeat itself, not that it
  generalises -- and generalisation is the whole point.
* Deprecation keys on the *upper* credible bound, so one unlucky episode
  cannot retire a good skill: it takes enough evidence that even the
  optimistic reading is poor.
* Nothing is deleted. A deprecated skill stays on disk with its evidence,
  so a bad deprecation is recoverable and the history is auditable.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path

from meta_evolver.core.types import Trajectory
from meta_evolver.knowledge.ingest import extract_concepts
from meta_evolver.llm.embeddings import cosine
from meta_evolver.skills.distill import mine_skills
from meta_evolver.skills.models import Skill


@dataclass
class SkillConfig:
    promote_min_uses: int = 3
    promote_min_rate: float = 0.6
    promote_min_win_tasks: int = 2
    promote_min_lcb: float = 0.35
    deprecate_min_uses: int = 4
    deprecate_max_ucb: float = 0.4
    stale_after: int = 8
    merge_similarity: float = 0.9
    min_relevance: float = 0.15
    trial_slots: int = 1
    max_active: int = 40


class SkillLibrary:
    def __init__(
        self,
        path: str | Path | None = None,
        embed: Callable[[str], list[float]] | None = None,
        config: SkillConfig | None = None,
        refiner: Callable[[Skill, list[Trajectory]], Skill] | None = None,
        ledger_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.ledger_path = Path(ledger_path) if ledger_path else (
            self.path.with_suffix(".ledger.jsonl") if self.path else None
        )
        self.embed = embed
        self.cfg = config or SkillConfig()
        self.refiner = refiner
        self.skills: dict[str, Skill] = {}
        self._dry = False
        self._viewed: set[tuple[str, str]] = set()
        if self.path and self.path.exists():
            for row in json.loads(self.path.read_text(encoding="utf-8")):
                s = Skill(**row)
                self.skills[s.id] = s

    # -- audit ledger ---------------------------------------------------------
    #
    # Every change to a skill is one append-only JSONL row holding the skill
    # before and after. Auto-created skills that supersede and merge each other
    # must be *reversible*: a bad merge is otherwise unrecoverable, and "we can
    # see exactly what the loop did to the library" is what makes it safe to let
    # it run unattended. (The idea is Hermes's curator ledger, minus its
    # content-addressed blob store -- skills here are small records.)

    @staticmethod
    def _snap(skill: Skill | None) -> dict | None:
        if skill is None:
            return None
        d = skill.model_dump()
        d["embedding"] = None
        return d

    def _audit(self, op: str, before: dict | None, after: Skill | None, generation: int, actor: str = "lifecycle") -> None:
        if self.ledger_path is None or self._dry:
            return
        a = self._snap(after)

        def h(x):
            return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16] if x else None

        row = {
            "generation": generation, "op": op, "actor": actor,
            "skill_id": (a or before or {}).get("id"),
            "before_hash": h(before), "after_hash": h(a), "before": before, "after": a,
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def history(self, skill_id: str | None = None) -> list[dict]:
        if self.ledger_path is None or not self.ledger_path.exists():
            return []
        rows = [json.loads(x) for x in self.ledger_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return [r for r in rows if skill_id is None or r["skill_id"] == skill_id]

    def rollback(self, skill_id: str, generation: int = 0) -> bool:
        """Undo the most recent recorded change to ``skill_id`` (the rollback is
        itself recorded). False when there is nothing to undo."""
        rows = [r for r in self.history(skill_id) if r["op"] != "rollback"]
        if not rows:
            return False
        last, current = rows[-1], self.skills.get(skill_id)
        before = self._snap(current)
        if last["before"] is None:  # the change was a creation: undo it by retiring
            if current is None:
                return False
            current.status = "deprecated"
        else:
            self.skills[skill_id] = Skill(**last["before"])
        self._audit("rollback", before, self.skills.get(skill_id), generation, actor="rollback")
        self.save()
        return True

    def _transition(self, skill: Skill, status: str, op: str, generation: int, **fields) -> None:
        before = self._snap(skill)
        skill.status = status  # type: ignore[assignment]
        for k, v in fields.items():
            setattr(skill, k, v)
        self._audit(op, before, skill, generation)

    # -- creation ------------------------------------------------------------

    def induce(self, trajectories: Sequence[Trajectory], generation: int = 0, **mine_kw) -> list[Skill]:
        """Mine ``trajectories`` and register any new candidates."""
        created: list[Skill] = []
        for skill in mine_skills(trajectories, generation=generation, **mine_kw):
            if self.refiner is not None:
                try:
                    skill = self.refiner(skill, list(trajectories))
                except Exception:  # a refiner may never fail a run
                    pass
            stored, is_new = self.add_candidate(skill, generation)
            if is_new:
                created.append(stored)
        return created

    def add_candidate(self, skill: Skill, generation: int = 0) -> tuple[Skill, bool]:
        twin = self._find_twin(skill)
        if twin is not None:
            # Same procedure re-discovered: pool its provenance, keep its record.
            twin.origin_task_ids = sorted({*twin.origin_task_ids, *skill.origin_task_ids})[:40]
            twin.when_to_use = list(dict.fromkeys([*twin.when_to_use, *skill.when_to_use]))[:8]
            return twin, False
        parent = self._lineage_parent(skill)
        if parent is not None:
            skill.parent_id, skill.version = parent.id, parent.version + 1
            skill.id = ""
            skill.model_post_init(None)
        if self.embed is not None and skill.embedding is None:
            skill.embedding = self._safe_embed(skill.embed_text())
        skill.created_generation = generation
        self.skills[skill.id] = skill
        self._audit("create", None, skill, generation, actor="induce")
        return skill, True

    def _find_twin(self, new: Skill) -> Skill | None:
        for s in self.skills.values():
            if s.benchmark != new.benchmark or s.status == "deprecated":
                continue
            if s.signature == new.signature:
                return s
            if set(s.tools) == set(new.tools) and self._similar(s, new) >= self.cfg.merge_similarity:
                return s
        return None

    def _lineage_parent(self, new: Skill) -> Skill | None:
        """An active skill over the same tools with a different procedure is
        the thing this new candidate would replace."""
        for s in self.skills.values():
            if (s.benchmark == new.benchmark and s.status == "active"
                    and set(s.tools) == set(new.tools) and s.signature != new.signature):
                return s
        return None

    def _similar(self, a: Skill, b: Skill) -> float:
        if a.embedding and b.embedding:
            return cosine(a.embedding, b.embedding)
        # Order-sensitive: the same verbs in a different order are a different
        # procedure (``take>go`` is not ``go>take``), so a set overlap would
        # wrongly call a reordered variant a duplicate and block versioning.
        return SequenceMatcher(None, a.signature.split(">"), b.signature.split(">")).ratio()

    def _safe_embed(self, text: str) -> list[float] | None:
        try:
            return self.embed(text) if self.embed else None
        except Exception:
            return None

    # -- guarded edits ----------------------------------------------------------

    def view(self, skill_id: str, actor: str = "agent") -> Skill | None:
        """Read a skill *as* ``actor``. A later :meth:`patch` by the same actor
        is only allowed after this -- an autonomous editor must have seen what
        it is about to change (Hermes's read-before-write rule), or it
        overwrites a version it never read."""
        skill = self.skills.get(skill_id)
        if skill is not None:
            self._viewed.add((actor, skill_id))
        return skill

    def patch(
        self,
        skill_id: str,
        actor: str = "agent",
        generation: int = 0,
        description: str | None = None,
        steps: list[str] | None = None,
        when_to_use: list[str] | None = None,
    ) -> Skill:
        """Edit a skill in place, behind three guards: it must exist, must not
        be ``locked``, and ``actor`` must have :meth:`view`-ed it first. Every
        edit lands in the ledger, so it can be rolled back."""
        skill = self.skills.get(skill_id)
        if skill is None:
            raise KeyError(f"unknown skill {skill_id!r}")
        if skill.locked:
            raise PermissionError(f"skill {skill.name!r} is locked against autonomous edits")
        if (actor, skill_id) not in self._viewed:
            raise PermissionError(f"{actor!r} must view skill {skill.name!r} before patching it")
        before = self._snap(skill)
        if description is not None:
            skill.description = description
        if steps is not None:
            skill.steps = list(steps)
        if when_to_use is not None:
            skill.when_to_use = list(when_to_use)
        skill.patch_count += 1
        skill.patched_at_uses = skill.uses
        self._audit("patch", before, skill, generation, actor=actor)
        self.save()
        return skill

    # -- precondition refinement --------------------------------------------------

    def refine(
        self,
        generation: int = 0,
        actor: str = "refiner",
        min_outcomes: int = 5,
        min_failures: int = 2,
        avoid_rate: float = 0.6,
        require_rate: float = 0.6,
        max_new: int = 3,
    ) -> list[dict]:
        """Learn *when* a skill works by contrasting its wins with its failures.

        A concept present in most of a skill's failures and in almost none of
        its wins becomes a negative precondition (``avoid_when``); a concept in
        most wins and almost no failures becomes a positive one
        (``when_to_use``). Both need real counts -- ``min_failures`` failures
        and ``min_outcomes`` episodes -- so one bad episode never teaches a
        rule. Edits go through the guarded, audited :meth:`patch` path, so each
        refinement is attributable and reversible, and the pass is idempotent:
        a skill whose preconditions already match its evidence is left alone.
        """
        changes: list[dict] = []
        for s in self.skills.values():
            if s.status == "deprecated" or s.locked or len(s.outcomes) < min_outcomes:
                continue  # a locked skill is skipped, never allowed to abort the pass
            wins = [set(c) for c, ok in s.outcomes if ok]
            fails = [set(c) for c, ok in s.outcomes if not ok]
            if len(fails) < min_failures:
                continue
            seen = set().union(*wins, *fails) if (wins or fails) else set()
            avoid, require = [], []
            for c in sorted(seen):
                w = sum(c in x for x in wins) / len(wins) if wins else 0.0
                f = sum(c in x for x in fails) / len(fails)
                n_fail = sum(c in x for x in fails)
                if f >= avoid_rate and w <= 0.1 and n_fail >= min_failures and c not in s.avoid_when:
                    avoid.append(c)
                elif wins and w >= require_rate and f <= 0.2 and c not in s.when_to_use:
                    require.append(c)
            avoid, require = avoid[:max_new], require[:max_new]
            if not (avoid or require):
                continue
            self.view(s.id, actor)
            self.patch(
                s.id,
                actor=actor,
                generation=generation,
                when_to_use=[*s.when_to_use, *require] if require else None,
            )
            if avoid:  # patch() has no avoid_when argument by design: preconditions are set here
                s.avoid_when = [*s.avoid_when, *avoid]
                self.save()
            changes.append({"skill": s.id, "avoid_added": avoid, "require_added": require})
        return changes

    # -- retrieval -----------------------------------------------------------

    def _relevance(self, skill: Skill, query: str, q_vec: list[float] | None) -> float:
        if q_vec is not None and skill.embedding:
            return max(0.0, cosine(q_vec, skill.embedding))
        want = set(extract_concepts(query, limit=30))
        if skill.avoid_when and want & set(skill.avoid_when):
            return 0.0  # a learned negative precondition: this skill has failed here
        if skill.when_to_use:
            # Precondition coverage: what fraction of the skill's own trigger
            # concepts does this task meet? Jaccard would punish a long
            # observation for containing words the skill never mentions.
            return len(want & set(skill.when_to_use)) / len(skill.when_to_use)
        have = set(extract_concepts(skill.description, limit=20))
        return len(want & have) / max(1, len(want | have)) if want and have else 0.0

    def retrieve(self, query: str, k: int = 2, domain: str | None = None) -> list[Skill]:
        """Up to ``k`` skills for ``query``: proven ones ranked by relevance
        weighted by evidence, plus ``trial_slots`` candidates to be tested."""
        q_vec = self._safe_embed(query)
        scored: list[tuple[float, Skill]] = []
        trials: list[tuple[float, Skill]] = []
        for s in self.skills.values():
            rel = self._relevance(s, query, q_vec)
            if rel < self.cfg.min_relevance or s.status == "deprecated":
                continue
            if s.status == "active":
                util = s.lcb()
                if domain and domain in s.domain_stats:
                    w, u = s.domain_stats[domain]
                    util = max(0.0, (w + 1.0) / (u + 2.0) - 0.15)  # domain-specific record
                scored.append((rel * (0.25 + util), s))
            else:
                trials.append((rel, s))
        scored.sort(key=lambda x: -x[0])
        trials.sort(key=lambda x: (x[1].uses, -x[0]))  # least-tested first
        n_trial = min(self.cfg.trial_slots, len(trials), max(0, k - 1) if scored else k)
        picked = [s for _, s in scored[: max(0, k - n_trial)]] + [s for _, s in trials[:n_trial]]
        return picked[:k]

    # -- evidence + lifecycle ---------------------------------------------------

    def credit(
        self,
        episodes: Iterable[tuple],
        generation: int = 0,
        gamma: float = 1.0,
    ) -> int:
        """``(skill_ids, success, domain, task_id[, concepts])`` per episode.

        The optional fifth element is the task's concept words; supplying it is
        what lets :meth:`refine` learn *when* a skill fails, not just that it does.
        """
        n = 0
        for ep in episodes:
            ids, success, domain, task_id = ep[:4]
            concepts = ep[4] if len(ep) > 4 else None
            for sid in ids:
                s = self.skills.get(sid)
                if s is not None:
                    s.record(success, task_id=task_id, domain=domain, generation=generation,
                             gamma=gamma, concepts=concepts)
                    n += 1
        return n

    def lifecycle(self, generation: int = 0) -> dict[str, int]:
        c = self.cfg
        out = {"promoted": 0, "deprecated": 0, "superseded": 0, "merged": 0}
        for s in list(self.skills.values()):
            if s.status == "candidate":
                rate = s.wins / s.uses if s.uses else 0.0
                if (s.uses >= c.promote_min_uses and rate >= c.promote_min_rate
                        and len(s.win_tasks) >= c.promote_min_win_tasks and s.lcb() >= c.promote_min_lcb):
                    self._transition(s, "active", "promote", generation, promoted_generation=generation)
                    out["promoted"] += 1
                    parent = self.skills.get(s.parent_id or "")
                    if parent is not None and parent.status == "active":
                        self._transition(parent, "deprecated", "supersede", generation, superseded_by=s.id)
                        out["superseded"] += 1
                elif s.uses >= c.deprecate_min_uses and s.ucb() < c.deprecate_max_ucb:
                    self._transition(s, "deprecated", "deprecate", generation)
                    out["deprecated"] += 1
                elif s.uses == 0 and generation - s.created_generation >= c.stale_after:
                    # Zero uses is absence of evidence, not evidence of failure: a
                    # skill is judged on time alone only after `stale_after`
                    # generations of trial slots in which it never got used.
                    self._transition(s, "deprecated", "expire", generation)
                    out["deprecated"] += 1
            elif s.status == "active":
                if s.uses >= c.deprecate_min_uses and s.ucb() < c.deprecate_max_ucb:
                    self._transition(s, "deprecated", "deprecate", generation)
                    out["deprecated"] += 1
        out["merged"] = self.merge_duplicates(generation)
        self._cap_active(generation)
        self.save()
        return out

    def preview_lifecycle(self, generation: int = 0) -> dict[str, object]:
        """What :meth:`lifecycle` *would* do, without doing it: the counts and
        each skill whose status would change. Nothing is written -- not the
        library file, not the ledger."""
        saved, path = self.skills, self.path
        self.skills, self.path, self._dry = copy.deepcopy(saved), None, True
        try:
            counts = self.lifecycle(generation)
            changes = {sid: (saved[sid].status, s.status) for sid, s in self.skills.items()
                       if sid in saved and saved[sid].status != s.status}
        finally:
            self.skills, self.path, self._dry = saved, path, False
        return {"counts": counts, "changes": changes}

    def merge_duplicates(self, generation: int = 0) -> int:
        """Fold active skills that are the same procedure, pooling evidence."""
        merged = 0
        live = sorted((s for s in self.skills.values() if s.status != "deprecated"), key=lambda s: -s.uses)
        for i, a in enumerate(live):
            if a.status == "deprecated":
                continue
            for b in live[i + 1 :]:
                if b.status == "deprecated" or a.benchmark != b.benchmark or set(a.tools) != set(b.tools):
                    continue
                if self._similar(a, b) >= self.cfg.merge_similarity:
                    before_a = self._snap(a)
                    a.uses += b.uses
                    a.wins += b.wins
                    a.win_tasks = list(dict.fromkeys([*a.win_tasks, *b.win_tasks]))[:50]
                    self._audit("merge-absorb", before_a, a, generation)
                    self._transition(b, "deprecated", "merge", generation, merged_into=a.id)
                    merged += 1
        return merged

    def _cap_active(self, generation: int = 0) -> None:
        active = sorted((s for s in self.skills.values() if s.status == "active"), key=lambda s: s.lcb())
        for s in active[: max(0, len(active) - self.cfg.max_active)]:
            self._transition(s, "deprecated", "evict", generation)

    # -- views / persistence ----------------------------------------------------

    def by_status(self, status: str) -> list[Skill]:
        return [s for s in self.skills.values() if s.status == status]

    def stats(self) -> dict[str, int]:
        return {st: len(self.by_status(st)) for st in ("candidate", "active", "deprecated")}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [s.model_dump() for s in self.skills.values()]
        for r in rows:
            r["embedding"] = None  # recomputed on demand; keeps the file small
        self.path.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    def export_markdown(self, directory: str | Path, statuses: Sequence[str] = ("active",)) -> list[Path]:
        """Write each skill as ``<dir>/<name>/SKILL.md`` (Agent Skills layout)."""
        written = []
        for s in self.skills.values():
            if s.status in statuses:
                p = Path(directory) / s.name / "SKILL.md"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(s.to_markdown(), encoding="utf-8")
                written.append(p)
        return written
