"""RedisEmbeddingCache and its wiring into Embedder.

Offline tests cover the "not configured" and "guarded against failure" paths,
which is what every process without Docker running actually exercises.
Integration tests skip (not fail) when nothing is listening, matching
tests/test_storage.py's pattern exactly -- reachability probed once per
session, not once per test, since a real connect attempt against a dead host
costs its full timeout.

    docker compose -f docker/docker-compose.yml --profile redis up -d
"""
from __future__ import annotations

import os
import uuid

import pytest

from meta_evolver.llm.embed_cache import REDIS_URL_ENV, RedisEmbeddingCache
from meta_evolver.llm.embeddings import Embedder

REDIS_URL = os.environ.get("META_EVOLVER_TEST_REDIS", "redis://localhost:6380/0")

_REACHABLE: dict[str, bool] = {}


def _unreachable() -> str | None:
    if "redis" not in _REACHABLE:
        cache = RedisEmbeddingCache.from_env(dim=8, url=REDIS_URL)
        _REACHABLE["redis"] = cache is not None
        if cache is not None:
            cache.close()
    return None if _REACHABLE["redis"] else f"no redis at {REDIS_URL}"


def _skip_if_unreachable() -> None:
    reason = _unreachable()
    if reason:
        pytest.skip(reason)


def _fresh_text() -> str:
    """A key guaranteed never to be in Redis already.

    Redis here has AOF persistence on (by design -- that is the point of a
    shared cache), so it survives across separate pytest invocations. A
    hardcoded string that was "never seen before" the first time this file
    ran stays cached forever after; every test that needs a true cache miss
    has to mint its own unique text instead.
    """
    return f"pytest-{uuid.uuid4().hex}"


# -- offline: no Redis configured, or configured but unreachable ------------


def test_from_env_with_nothing_set_returns_none(monkeypatch):
    monkeypatch.delenv(REDIS_URL_ENV, raising=False)
    assert RedisEmbeddingCache.from_env(dim=8) is None


def test_from_env_with_an_unreachable_url_returns_none_not_raises():
    # Port 1 is never a real Redis; the 1s connect timeout keeps this fast.
    assert RedisEmbeddingCache.from_env(dim=8, url="redis://localhost:1/0") is None


def test_embedder_with_no_redis_configured_behaves_exactly_as_before(monkeypatch):
    monkeypatch.delenv(REDIS_URL_ENV, raising=False)
    embedder = Embedder(dim=8)
    assert embedder.redis_cache is None
    # The ordinary in-memory-cache path still works with no Redis anywhere.
    vec = embedder.embed_one("connection pool exhausted")
    assert len(vec) == 8
    assert embedder.embed_one("connection pool exhausted") == vec


def test_get_and_set_swallow_a_client_failure_rather_than_raise():
    """Constructed successfully, then the client breaks mid-run -- a real
    failure mode (the container restarts, a network blip) that must cost a
    cache miss, not the episode."""

    class _BrokenClient:
        def get(self, key):
            raise ConnectionError("boom")

        def set(self, key, value, ex=None):
            raise ConnectionError("boom")

        def ping(self):
            return True

        def close(self):
            pass

    cache = RedisEmbeddingCache.__new__(RedisEmbeddingCache)
    cache._client = _BrokenClient()
    cache._prefix = "meta-evolver:embed:8:"
    cache.ttl_seconds = None
    cache.n_hits = 0
    cache.n_misses = 0

    assert cache.get("document", "text") is None
    cache.set("document", "text", [0.1, 0.2])  # must not raise


# -- integration: a real Redis, via docker compose --------------------------


def test_set_and_get_round_trip_through_a_real_redis():
    _skip_if_unreachable()
    text = _fresh_text()
    cache = RedisEmbeddingCache.from_env(dim=8, url=REDIS_URL)
    try:
        assert cache.get("document", text) is None
        cache.set("document", text, [0.1, 0.2, 0.3])
        assert cache.get("document", text) == [0.1, 0.2, 0.3]
    finally:
        cache.close()


def test_query_and_document_kinds_are_cached_separately():
    _skip_if_unreachable()
    text = _fresh_text()
    cache = RedisEmbeddingCache.from_env(dim=8, url=REDIS_URL)
    try:
        cache.set("document", text, [1.0])
        cache.set("query", text, [2.0])
        assert cache.get("document", text) == [1.0]
        assert cache.get("query", text) == [2.0]
    finally:
        cache.close()


def test_a_second_embedder_gets_a_cache_hit_without_recomputing():
    """The actual point of this whole module: two Embedder instances --
    standing in for two processes -- share vectors instead of each computing
    its own, using the local hashed fallback (no real provider needed here,
    so this proves the cache layer itself, independent of any API key)."""
    _skip_if_unreachable()

    first = Embedder(dim=8, redis_url=REDIS_URL)
    text = _fresh_text()
    vec = first.embed_one(text)
    assert first.redis_cache.n_misses >= 1
    first.redis_cache.close()

    second = Embedder(dim=8, redis_url=REDIS_URL)
    try:
        assert second.embed_one(text) == vec
        assert second.redis_cache.n_hits >= 1
        assert second.redis_cache.n_misses == 0
    finally:
        second.redis_cache.close()


def test_bank_close_reaches_the_redis_client():
    """redis.Redis is connection-pool-backed and transparently reconnects
    after close() -- there is no "refuses further use" to observe -- so this
    checks the one real contract: bank.close() must actually call through to
    the cache's close(), not silently skip it."""
    from meta_evolver.memory.bank import ReasoningMemoryBank

    _skip_if_unreachable()
    embedder = Embedder(dim=8, redis_url=REDIS_URL)
    calls = []
    embedder.redis_cache.close = lambda: calls.append(True)

    bank = ReasoningMemoryBank(embedder=embedder)
    bank.close()

    assert calls == [True]
