"""ReasoningMemoryBank -- a memory that gets *better*, not just bigger.

An append-only store of distilled lessons plateaus quickly and then degrades:
duplicates crowd the retrieval slots, and a confidently-wrong lesson gets
re-injected into every similar task forever. Three mechanisms here prevent
that, and they are the reason the outer loop keeps improving past generation
two or three.

**Dedup on write.** Items are keyed by normalized ``scenario + lesson`` text
and by embedding proximity. A near-duplicate merges into the existing item --
inheriting its usage record -- rather than taking a second retrieval slot.

**Credit assignment.** Every episode records which memory ids were in its
prompt. After the generation, ``credit_assign`` increments ``uses`` for each
and ``wins`` where the episode succeeded. That turns each item's ``utility``
into a Beta posterior over "episodes citing this succeed".

**Utility-weighted retrieval and pruning.** Retrieval ranks by similarity
*modulated by* utility, so an item that keeps appearing alongside failures
loses its slot before it is ever deleted. ``prune`` then removes items that
have had a fair trial and lost.

Retrieval offers MMR alongside plain cosine. With a bank of near-synonymous
lessons, pure cosine returns five paraphrases of one idea; MMR spends the
slots on genuinely different strategies, which matters most exactly when the
task is unfamiliar.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from meta_evolver.core.types import MemoryItem
from meta_evolver.llm.embeddings import Embedder, cosine


class ReasoningMemoryBank:
    """Persistent, self-curating store of reusable strategies."""

    def __init__(
        self,
        items: Iterable[MemoryItem | dict] | None = None,
        embedder: Embedder | None = None,
        path: str | Path | None = None,
        dedup_similarity: float = 0.93,
        max_items: int = 500,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.path = Path(path) if path else None
        self.dedup_similarity = float(dedup_similarity)
        self.max_items = int(max_items)
        self.items: list[MemoryItem] = []
        for raw in items or []:
            self.items.append(raw if isinstance(raw, MemoryItem) else MemoryItem(**raw))
        self._reindex()

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(
        cls, path: str | Path, embedder: Embedder | None = None, **kwargs
    ) -> ReasoningMemoryBank:
        p = Path(path)
        items: list[MemoryItem] = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(MemoryItem(**json.loads(line)))
                except (json.JSONDecodeError, ValueError):
                    # A single corrupt line should cost one memory, not the run.
                    continue
        return cls(items=items, embedder=embedder, path=p, **kwargs)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path configured for this bank")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for item in self.items:
                fh.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
        self.path = target
        return target

    # -- indexing ----------------------------------------------------------

    def _reindex(self) -> None:
        """Fill in missing ids and embeddings. Cheap and idempotent."""
        missing = [it for it in self.items if not it.embedding]
        if missing:
            vectors = self.embedder.embed([it.embed_text() for it in missing])
            for item, vec in zip(missing, vectors, strict=True):
                item.embedding = vec
        for i, item in enumerate(self.items):
            if not item.id:
                item.id = f"mem-{item.key()}"
            if not item.title:
                item.title = (item.lesson or item.scenario or f"memory {i}")[:70]

    # -- writes ------------------------------------------------------------

    def add(self, item: MemoryItem, generation: int = 0) -> tuple[MemoryItem, bool]:
        """Insert ``item``, merging into a near-duplicate if one exists.

        Returns ``(stored_item, was_new)``.
        """
        item.created_generation = generation
        if not item.embedding:
            item.embedding = self.embedder.embed_one(item.embed_text())
        if not item.id:
            item.id = f"mem-{item.key()}"

        existing = self._find_duplicate(item)
        if existing is not None:
            # Keep the longer procedure -- later inductions tend to be more
            # specific -- but preserve the incumbent's earned track record.
            if len(item.procedure) > len(existing.procedure):
                existing.procedure = item.procedure
            if len(item.lesson) > len(existing.lesson):
                existing.lesson = item.lesson
            existing.triggers = sorted(set(existing.triggers) | set(item.triggers))
            existing.source_task_ids = sorted(
                set(existing.source_task_ids) | set(item.source_task_ids)
            )
            return existing, False

        self.items.append(item)
        self._enforce_capacity()
        return item, True

    def extend(self, items: Iterable[MemoryItem], generation: int = 0) -> int:
        return sum(1 for it in items if self.add(it, generation=generation)[1])

    def _find_duplicate(self, item: MemoryItem) -> MemoryItem | None:
        key = item.key()
        for existing in self.items:
            if existing.key() == key:
                return existing
        if item.embedding:
            for existing in self.items:
                if (
                    existing.polarity == item.polarity
                    and existing.embedding
                    and cosine(item.embedding, existing.embedding) >= self.dedup_similarity
                ):
                    return existing
        return None

    def _enforce_capacity(self) -> None:
        """Bound the bank by dropping the least useful items first."""
        if len(self.items) <= self.max_items:
            return
        self.items.sort(key=lambda it: (it.utility, it.uses, it.last_used_generation))
        del self.items[: len(self.items) - self.max_items]

    # -- retrieval ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: str = "mmr",
        mmr_lambda: float = 0.6,
        utility_weight: float = 0.35,
        min_similarity: float = 0.0,
    ) -> list[MemoryItem]:
        """Top-``k`` memories for ``query``.

        ``utility_weight`` blends each item's track record into its ranking
        score. At 0 this is pure semantic similarity; at 1 a well-performing
        but marginally relevant memory can outrank a relevant but unproven
        one. The default leans on similarity while still letting repeated
        failure cost an item its slot.
        """
        if not self.items or k <= 0:
            return []
        self._reindex()

        q_vec = self.embedder.embed_one(query)
        scored: list[tuple[float, float, MemoryItem]] = []
        for item in self.items:
            sim = cosine(q_vec, item.embedding or [])
            if sim < min_similarity:
                continue
            # Utility is centred on 0.5 (the untested prior) so an unproven
            # item is neither promoted nor demoted.
            score = sim * (1.0 - utility_weight) + (item.utility - 0.5) * 2 * utility_weight
            scored.append((score, sim, item))

        if not scored:
            return []
        scored.sort(key=lambda t: t[0], reverse=True)

        if mode == "cosine":
            return [item for _, _, item in scored[:k]]

        # Maximal Marginal Relevance: pay for each additional slot in
        # information, not in paraphrase.
        selected: list[tuple[float, float, MemoryItem]] = []
        pool = list(scored)
        while pool and len(selected) < k:
            best_idx, best_val = 0, -float("inf")
            for i, (score, _sim, item) in enumerate(pool):
                if selected:
                    redundancy = max(
                        cosine(item.embedding or [], chosen.embedding or [])
                        for _, _, chosen in selected
                    )
                else:
                    redundancy = 0.0
                val = mmr_lambda * score - (1.0 - mmr_lambda) * redundancy
                if val > best_val:
                    best_idx, best_val = i, val
            selected.append(pool.pop(best_idx))
        return [item for _, _, item in selected]

    # -- learning ----------------------------------------------------------

    def credit_assign(
        self, episodes: Sequence[tuple[Sequence[str], bool]], generation: int = 0
    ) -> None:
        """Update usage statistics from ``(retrieved_ids, success)`` pairs.

        This is the feedback edge that makes the bank a learner rather than a
        cache. Episodes that ended in an infrastructure error must be filtered
        out *before* this call -- charging a memory for a rate-limit would
        eventually prune the bank's best strategies.
        """
        by_id = {item.id: item for item in self.items}
        for ids, success in episodes:
            for mid in ids:
                item = by_id.get(mid)
                if item is None:
                    continue
                item.uses += 1
                item.last_used_generation = generation
                if success:
                    item.wins += 1

    def prune(self, min_uses: int = 4, min_utility: float = 0.34) -> list[MemoryItem]:
        """Drop memories that have had a fair trial and underperformed.

        ``min_uses`` is the fairness guard: an item retrieved twice into two
        hard tasks has not been disproven. Returns the removed items.
        """
        keep, dropped = [], []
        for item in self.items:
            if item.uses >= min_uses and item.utility < min_utility:
                dropped.append(item)
            else:
                keep.append(item)
        self.items = keep
        return dropped

    # -- prompt rendering --------------------------------------------------

    def render_prompt_block(
        self, items: Sequence[MemoryItem], soft_prior: bool = True
    ) -> str:
        """Format memories for injection into a system prompt.

        The framing is load-bearing. Presenting retrieved memories as
        instructions produces confirmation bias: on an out-of-distribution
        task the agent follows a strategy that does not apply and keeps
        following it. Framing them as priors that may be wrong -- and saying
        out loud what to do when they are -- is what makes retrieval safe to
        use on tasks the bank has never seen.
        """
        if not items:
            return ""
        header = (
            "## Retrieved experience (priors, not instructions)\n"
            "These come from earlier episodes and may not fit this task. Use them\n"
            "to skip known dead ends. If two steps of following one produce no\n"
            "progress, abandon it and reason from the observations instead.\n"
            if soft_prior
            else "## Retrieved experience\n"
        )
        body = "\n\n".join(item.render(i) for i, item in enumerate(items, 1))
        return header + "\n" + body + "\n"

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, float | int]:
        if not self.items:
            return {"n": 0, "n_success": 0, "n_failure": 0, "mean_utility": 0.0, "n_tested": 0}
        tested = [it for it in self.items if it.uses > 0]
        return {
            "n": len(self.items),
            "n_success": sum(1 for it in self.items if it.polarity == "success"),
            "n_failure": sum(1 for it in self.items if it.polarity == "failure"),
            "n_tested": len(tested),
            "mean_utility": (
                sum(it.utility for it in tested) / len(tested) if tested else 0.0
            ),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
