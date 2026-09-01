"""Embeddings, with a deterministic local fallback.

Retrieval must keep working without an embedding API -- offline, in CI, and
when a provider is down -- so every embedder degrades to a hashed bag-of-words
encoder rather than failing.

The fallback uses ``zlib.crc32``, not the built-in ``hash()``. Python
randomizes string hashing per process unless ``PYTHONHASHSEED`` is pinned, so
a bank persisted by one process and queried by the next would compare vectors
drawn from two different random projections: retrieval would silently return
near-noise, and the failure looks like "memory just doesn't help much" rather
than like a bug. CRC32 is stable across processes, machines and releases.
"""
from __future__ import annotations

import math
import re
import zlib
from collections.abc import Sequence

from meta_evolver.llm.client import BaseLLMClient

FALLBACK_DIM = 256

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

#: Tokens that appear in nearly every trajectory and carry no retrieval signal.
_STOPWORDS = frozenset(
    """a an the and or but if then than to of in on at for with without from by is are was
    were be been being do does did doing this that these those it its as not no you your i
    we they he she them us me my our their there here what which who when where how""".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def hashed_embedding(text: str, dim: int = FALLBACK_DIM) -> list[float]:
    """Stable, L2-normalized hashed bag-of-words with sub-linear term weighting.

    Uses the signed hashing trick (one bucket, one sign bit) so unrelated
    tokens colliding in a bucket cancel on average instead of compounding.
    """
    vec = [0.0] * dim
    counts: dict[str, int] = {}
    for tok in _tokens(text):
        counts[tok] = counts.get(tok, 0) + 1
    for tok, count in counts.items():
        digest = zlib.crc32(tok.encode("utf-8"))
        bucket = digest % dim
        sign = 1.0 if (digest >> 16) & 1 else -1.0
        vec[bucket] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class Embedder:
    """Embeds text via an LLM client, falling back to the local encoder.

    Results are cached by text, because the memory bank re-embeds the same
    items on every retrieval round and the calls are neither free nor fast.
    """

    def __init__(self, client: BaseLLMClient | None = None, dim: int = FALLBACK_DIM) -> None:
        self.client = client
        self.dim = dim
        self._cache: dict[str, list[float]] = {}
        self.remote_available = client is not None
        self.n_remote = 0
        self.n_fallback = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        missing = [t for t in texts if t not in self._cache]

        if missing and self.remote_available and self.client is not None:
            vectors = self.client.embed(missing)
            if vectors and len(vectors) == len(missing):
                for text, vec in zip(missing, vectors, strict=True):
                    self._cache[text] = vec
                self.n_remote += len(missing)
                missing = []
            else:
                # One failure disables the remote path for the rest of the run.
                # Retrying per item would turn a provider outage into a very
                # slow run instead of a slightly worse one.
                self.remote_available = False

        for text in missing:
            self._cache[text] = hashed_embedding(text, self.dim)
            self.n_fallback += 1

        return [self._cache[t] for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, tolerant of dimension mismatch.

    Vectors can genuinely differ in width within one bank: items embedded
    remotely before an outage sit alongside items embedded locally after it.
    Comparing over the shared prefix keeps retrieval running; the alternative
    is an exception in the middle of a long evolution run.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
