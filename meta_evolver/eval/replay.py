"""Replay screening -- cheap pre-selection of prompt candidates before they
pay for a real held-out validation rollout.

Dream-RSI's own idea is a learned simulator built from past search trees, so
a candidate change can be scored off-policy at zero runtime cost. Building an
actual simulator would itself require training something, which is not zero
cost the first time -- what this module does instead is the honest, bounded
approximation of the same *purpose*: every candidate this project's own
prompt optimizer proposes gets a real validation score eventually (that is
what ``_select_prompt`` does), so that history is a free, growing dataset of
"prompt text -> how well it actually did". A new candidate's predicted score
is its nearest past neighbours' real scores, by embedding similarity --
reusing whichever :class:`~meta_evolver.llm.embeddings.Embedder` the run
already has, so this adds no new model dependency and, on the local fallback
embedder, no new API cost at all.

The mechanism only has one job worth trusting it for: deciding which
candidates are worth a real rollout when there are more candidates than
budget. It is explicitly *not* trusted to replace validation -- every
candidate that survives screening still goes through the exact same
``_select_prompt`` margin-and-rehearsal gate as if screening had never run.
And it does not trust itself either: :meth:`ReplayScreener.update_agreement`
checks its own predictions against what real validation actually found each
generation, and turns screening off the moment they stop correlating rather
than letting a stale predictor keep silently discarding good candidates.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from meta_evolver.llm.embeddings import Embedder, cosine


class ReplayScreener:
    """Predicts a prompt candidate's validation score from embedding
    similarity to past candidates whose real score is already known.

    ``top_k=None`` (the default) never trims anything -- every candidate
    still goes to real validation, and the screener only accumulates history
    in the background. Set ``top_k`` to actually start screening.
    """

    def __init__(
        self,
        embedder: Embedder,
        top_k: int | None = None,
        min_agreement: float = 0.6,
        min_history: int = 3,
        neighbors: int = 3,
    ) -> None:
        self.embedder = embedder
        self.top_k = top_k
        self.min_agreement = float(min_agreement)
        self.min_history = int(min_history)
        self.neighbors = int(neighbors)
        self.history: list[tuple[list[float], float]] = []
        self.disabled = False
        """Set once :meth:`update_agreement` finds screening's own
        predictions no longer track real validation results. Sticky for the
        rest of the run -- a predictor that has already been shown wrong
        once this generation is not more trustworthy next generation just
        because the data changed."""
        self.last_agreement: float | None = None
        self.n_screened_out = 0

    def record(self, text: str, real_score: float) -> None:
        """Feed one real validation result back in. Called for every
        candidate that actually gets validated, screened or not -- the
        predictor should learn from everything it has real evidence for."""
        self.history.append((self.embedder.embed_one(text), float(real_score)))

    def predict(self, text: str) -> float | None:
        """Predicted score for ``text``, or ``None`` with too little history
        to predict from -- not zero, which would look like a confident bad
        prediction rather than an honest "cannot tell yet"."""
        if len(self.history) < self.min_history:
            return None
        vec = self.embedder.embed_one(text)
        ranked = sorted(self.history, key=lambda h: cosine(vec, h[0]), reverse=True)
        top = ranked[: min(self.neighbors, len(ranked))]
        return sum(score for _, score in top) / len(top)

    def screen(self, candidates: Sequence[Any]) -> list[Any]:
        """The candidates worth a real validation rollout this generation.

        A candidate screening cannot yet predict anything for (too little
        history) is always kept -- screening only trims what it can actually
        judge, never what it is uncertain about. Disabled, or with fewer
        candidates than the budget, this is the identity function.
        """
        candidates = list(candidates)
        if self.disabled or self.top_k is None or len(candidates) <= self.top_k:
            return candidates

        scored = [(c, self.predict(_text_of(c))) for c in candidates]
        certain = [(c, p) for c, p in scored if p is not None]
        uncertain = [c for c, p in scored if p is None]
        certain.sort(key=lambda cp: cp[1], reverse=True)

        budget = max(0, self.top_k - len(uncertain))
        kept = [c for c, _ in certain[:budget]]
        dropped = len(candidates) - len(uncertain) - len(kept)
        self.n_screened_out += max(0, dropped)
        return uncertain + kept

    def update_agreement(self, validated: Sequence[tuple[Any, float]]) -> float | None:
        """Compare this generation's predicted ranking against the real
        scores ``validated`` came back with, for whichever candidates were
        actually validated. Returns the agreement fraction, or ``None`` when
        there were fewer than two candidates to rank against each other.

        Pairwise concordance: of every pair, the fraction where the
        predicted order and the real order agree. 1.0 is perfect agreement,
        0.5 is what a coin flip gets, and this project treats anything below
        ``min_agreement`` as screening no longer earning its keep.
        """
        pairs_with_prediction = [
            (self.predict(_text_of(c)), real) for c, real in validated
        ]
        pairs_with_prediction = [(p, r) for p, r in pairs_with_prediction if p is not None]
        if len(pairs_with_prediction) < 2:
            self.last_agreement = None
            return None

        concordant = 0
        total = 0
        for i in range(len(pairs_with_prediction)):
            for j in range(i + 1, len(pairs_with_prediction)):
                p_i, r_i = pairs_with_prediction[i]
                p_j, r_j = pairs_with_prediction[j]
                if p_i == p_j or r_i == r_j:
                    continue  # a tie has no order to agree or disagree on
                total += 1
                if (p_i > p_j) == (r_i > r_j):
                    concordant += 1

        if total == 0:
            self.last_agreement = None
            return None

        agreement = concordant / total
        self.last_agreement = agreement
        if agreement < self.min_agreement:
            self.disabled = True
        return agreement


def _text_of(candidate: Any) -> str:
    """A ``PromptCandidate`` has ``.text``; accept a bare string too, so this
    module has no hard import dependency on ``prompts.optimizer``."""
    return candidate.text if hasattr(candidate, "text") else str(candidate)


__all__ = ["ReplayScreener"]
