"""ReplayScreener: prediction, screening, and the self-disabling agreement
check. All offline -- uses the fallback hashed embedder, no API calls."""
from __future__ import annotations

from dataclasses import dataclass

from meta_evolver.eval.replay import ReplayScreener
from meta_evolver.llm.embeddings import Embedder


@dataclass
class _Cand:
    text: str


def _screener(**kwargs) -> ReplayScreener:
    return ReplayScreener(embedder=Embedder(dim=32), min_history=2, **kwargs)


# -- record / predict -----------------------------------------------------------


def test_predict_returns_none_below_min_history():
    s = _screener()
    s.record("candidate about restarting the pool", 0.8)
    assert s.predict("another candidate about the pool") is None


def test_predict_uses_nearest_neighbours_once_there_is_enough_history():
    s = _screener()
    s.record("restart the connection pool and raise the limit", 0.9)
    s.record("restart the connection pool after raising limits", 0.85)
    s.record("check the weather forecast for tomorrow", 0.1)
    # Text similar to the pool-restart candidates should predict near their
    # scores, not the unrelated weather one.
    predicted = s.predict("restart the pool and increase its connection limit")
    assert predicted is not None
    assert predicted > 0.5


def test_record_does_not_mutate_top_k_or_disabled_state():
    s = _screener(top_k=2)
    s.record("a", 0.5)
    assert s.top_k == 2
    assert not s.disabled


# -- screen -----------------------------------------------------------------


def test_screen_is_identity_with_no_top_k():
    s = _screener()
    cands = [_Cand("a"), _Cand("b"), _Cand("c")]
    assert s.screen(cands) == cands


def test_screen_is_identity_when_candidates_fit_the_budget():
    s = _screener(top_k=5)
    cands = [_Cand("a"), _Cand("b")]
    assert s.screen(cands) == cands


def test_screen_keeps_everything_uncertain_when_history_is_too_thin():
    """No history yet -- screening cannot judge anything, so it must not
    discard candidates it has no basis to discard."""
    s = _screener(top_k=1)
    cands = [_Cand("a"), _Cand("b"), _Cand("c")]
    kept = s.screen(cands)
    assert len(kept) == 3
    assert all(c in cands for c in kept)


def test_screen_keeps_the_highest_predicted_candidates():
    s = _screener(top_k=1)
    s.record("restart the connection pool", 0.9)
    s.record("check disk pressure on the log shipper", 0.9)
    s.record("read logs and do nothing else", 0.1)
    strong = _Cand("restart the connection pool immediately")
    weak = _Cand("read logs and do nothing else, repeatedly")
    kept = s.screen([strong, weak])
    assert kept == [strong]
    assert s.n_screened_out == 1


def test_screen_when_disabled_is_identity_regardless_of_top_k():
    s = _screener(top_k=1)
    s.record("a", 0.5)
    s.record("b", 0.5)
    s.disabled = True
    cands = [_Cand("a"), _Cand("b"), _Cand("c")]
    assert s.screen(cands) == cands


# -- update_agreement ---------------------------------------------------------


def test_agreement_is_none_with_fewer_than_two_predictable_candidates():
    s = _screener()
    assert s.update_agreement([(_Cand("only one"), 0.5)]) is None


def test_agreement_is_high_when_predictions_track_real_scores():
    s = _screener(min_agreement=0.6)
    s.record("restart the pool and raise the limit", 0.9)
    s.record("restart the pool with a higher limit", 0.85)
    s.record("look at the weather", 0.1)
    s.record("check tomorrow's forecast", 0.15)

    validated = [
        (_Cand("restart the connection pool, raise its limit"), 0.88),
        (_Cand("look at tomorrow's weather forecast"), 0.12),
    ]
    agreement = s.update_agreement(validated)
    assert agreement is not None
    assert agreement >= 0.6
    assert not s.disabled


def test_agreement_below_threshold_disables_screening():
    s = _screener(min_agreement=0.6, neighbors=1)
    # Deliberately mislead the predictor: text similarity says one thing,
    # the real validated scores say the opposite.
    s.record("alpha alpha alpha", 0.9)
    s.record("beta beta beta", 0.1)

    validated = [
        (_Cand("alpha alpha alpha"), 0.05),  # predicted high, actually low
        (_Cand("beta beta beta"), 0.95),  # predicted low, actually high
    ]
    agreement = s.update_agreement(validated)
    assert agreement is not None
    assert agreement < 0.6
    assert s.disabled is True


def test_disabled_is_sticky_across_generations():
    s = _screener(top_k=1, min_agreement=0.6, neighbors=1)
    s.record("alpha alpha alpha", 0.9)
    s.record("beta beta beta", 0.1)
    s.update_agreement(
        [(_Cand("alpha alpha alpha"), 0.05), (_Cand("beta beta beta"), 0.95)]
    )
    assert s.disabled

    # Even data that WOULD agree now must not silently re-enable screening.
    agreement = s.update_agreement(
        [(_Cand("alpha alpha alpha"), 0.9), (_Cand("beta beta beta"), 0.1)]
    )
    assert s.disabled
    cands = [_Cand("x"), _Cand("y"), _Cand("z")]
    assert s.screen(cands) == cands


def test_ties_are_excluded_from_the_agreement_count():
    """A tie has no order to agree or disagree on -- it must not be counted
    as either concordant or discordant."""
    s = _screener(neighbors=1)
    s.record("same text everywhere", 0.5)
    s.record("same text everywhere", 0.5)
    validated = [
        (_Cand("same text everywhere"), 0.5),
        (_Cand("same text everywhere"), 0.5),
    ]
    # Both predicted and real are tied -- no comparable pairs at all.
    assert s.update_agreement(validated) is None
