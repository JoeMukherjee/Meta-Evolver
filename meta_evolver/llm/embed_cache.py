"""Redis-backed embedding cache -- optional, and shared across processes.

:class:`~meta_evolver.llm.embeddings.Embedder` already caches by
``(kind, text)`` in a plain dict, which is free and correct within one
process but starts cold every time: a fresh CLI invocation, or one of
``scripts/baseline.py``'s per-arm runs, re-pays every embedding call the
previous run already made. This is the same in-memory-first, remote-second
shape every other optional backend in this project follows (see
:mod:`meta_evolver.storage`): nothing here is required, and a bank with no
``$META_EVOLVER_REDIS_URL`` configured runs exactly as it always has.

Values are stored as JSON arrays of floats under keys namespaced by the
configured embedding width, so a width change (a different
``$DEFAULT_EMBED_DIMENSIONS``) cannot return a vector of the wrong
dimensionality from a stale entry -- it just misses and recomputes, the same
failure mode a fresh cache has.
"""
from __future__ import annotations

import json
import os

#: Env var this module checks, matching $META_EVOLVER_DB_URL / _GRAPH_URL.
REDIS_URL_ENV = "META_EVOLVER_REDIS_URL"


class RedisEmbeddingCache:
    """Thin wrapper around a ``redis`` client. Never raises past construction.

    Every call is guarded: a Redis hiccup mid-run should cost one cache miss
    (a recomputed embedding), not the run -- the same discipline
    :meth:`~meta_evolver.memory.bank.ReasoningMemoryBank.retrieve` already
    applies to its own store-backed candidate search.
    """

    def __init__(self, url: str, dim: int, ttl_seconds: int | None = None) -> None:
        import redis

        self._client = redis.Redis.from_url(
            url,
            decode_responses=True,
            # Short and deliberate: every Embedder() construction probes
            # this once (see from_env), and an unreachable host must fail
            # fast rather than stall every episode behind a TCP timeout.
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        self._prefix = f"meta-evolver:embed:{int(dim)}:"
        self.ttl_seconds = ttl_seconds
        self.n_hits = 0
        self.n_misses = 0

    @classmethod
    def from_env(cls, dim: int, url: str | None = None) -> "RedisEmbeddingCache | None":
        """A cache if ``$META_EVOLVER_REDIS_URL`` (or ``url``) is set and
        reachable, else ``None`` -- the signal :class:`Embedder` uses to fall
        back to its in-memory-only cache."""
        target = url or os.environ.get(REDIS_URL_ENV)
        if not target:
            return None
        try:
            cache = cls(target, dim)
            cache._client.ping()
        except Exception:
            return None
        return cache

    def _key(self, kind: str, text: str) -> str:
        import hashlib

        # The text itself, not just its hash, would make keys unboundedly
        # long for a big trajectory excerpt; Redis keys are meant to be short.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self._prefix}{kind}:{digest}"

    def get(self, kind: str, text: str) -> list[float] | None:
        try:
            raw = self._client.get(self._key(kind, text))
        except Exception:
            return None
        if raw is None:
            self.n_misses += 1
            return None
        self.n_hits += 1
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

    def set(self, kind: str, text: str, vector: list[float]) -> None:
        try:
            self._client.set(
                self._key(kind, text), json.dumps(vector), ex=self.ttl_seconds
            )
        except Exception:
            return None

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


__all__ = ["REDIS_URL_ENV", "RedisEmbeddingCache"]
