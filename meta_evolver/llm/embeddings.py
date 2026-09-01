"""Embeddings, with a deterministic local fallback.

Backed by LangChain's ``Embeddings`` interface, so any provider integration
works and the memory bank never sees a vendor SDK.

**Model.** ``gemini-embedding-2`` at 768 of its 3072 dimensions. Both Gemini
embedding models are Matryoshka-trained -- the most significant structure is
packed into the leading dimensions, so a truncated vector keeps nearly all of
its retrieval signal at a quarter of the storage and a quarter of the
dot-product cost. Both costs are real here: a bank persists every vector to
JSONL, and MMR retrieval is O(k*n) dot products per episode.

The reason for ``-2`` specifically is normalization. It **renormalizes
truncated output automatically**; ``gemini-embedding-001`` does not, so a
768-dim vector from ``-001`` is no longer unit-norm and every consumer has to
renormalize it or silently start comparing by magnitude as well as direction.
This module normalizes on arrival anyway, so a bank stays coherent across a
model switch -- but with ``-2`` that guard is a belt, not the braces.

**Asymmetry.** A stored memory and a search query are different kinds of
text, and Gemini exposes that: ``RETRIEVAL_DOCUMENT`` and ``RETRIEVAL_QUERY``
project into the same space but from different sides, which measurably beats
embedding both identically. :class:`Embedder` therefore has two entry points --
:meth:`~Embedder.embed_documents` for what goes into the bank and
:meth:`~Embedder.embed_query` for what searches it -- and the cache is keyed by
both text *and* kind, because the same string legitimately has two vectors.

**Fallback.** Retrieval must keep working with no embedding API -- offline, in
CI, when a provider is down -- so this degrades to a hashed bag-of-words
encoder rather than failing.

That fallback uses ``zlib.crc32``, not the built-in ``hash()``. Python
randomizes string hashing per process unless ``PYTHONHASHSEED`` is pinned, so
a bank persisted by one process and queried by the next would compare vectors
drawn from two different random projections: retrieval would return near-noise
while looking like it worked. CRC32 is stable across processes and platforms.
"""
from __future__ import annotations

import math
import re
import zlib
from collections.abc import Sequence
from typing import Any

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


def _default_width() -> int:
    """The configured embedding width, imported lazily to avoid a cycle."""
    from meta_evolver.llm.client import DEFAULT_EMBED_DIMENSIONS

    return DEFAULT_EMBED_DIMENSIONS or FALLBACK_DIM


def l2_normalize(vec: Sequence[float]) -> list[float]:
    """Scale to unit length. A zero vector is returned unchanged."""
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else list(vec)


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
    return l2_normalize(vec)


def build_embeddings(
    model: str | None = None, dimensions: int | None = None, **kwargs: Any
) -> Any:
    """A LangChain ``Embeddings`` for ``model``, or ``None`` if unavailable.

    Returns ``None`` rather than raising when the provider package is missing
    or no credentials are configured. :class:`Embedder` then runs on the local
    encoder, which is the difference between "retrieval is a bit worse" and
    "the run does not start".
    """
    from meta_evolver.llm.client import (
        DEFAULT_EMBED_DIMENSIONS,
        DEFAULT_EMBED_MODEL,
        load_dotenv_once,
        split_model,
    )

    load_dotenv_once()
    spec = model or DEFAULT_EMBED_MODEL
    width = DEFAULT_EMBED_DIMENSIONS if dimensions is None else dimensions
    provider, name = split_model(spec)

    try:
        if provider == "google_genai":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            # `output_dimensionality` is a first-class parameter on this
            # integration. Routing through a generic gateway instead risks it
            # being dropped: the width silently reverts to 3072, storage
            # quadruples, and nothing errors.
            if width:
                kwargs.setdefault("output_dimensionality", int(width))
            return GoogleGenerativeAIEmbeddings(model=name, **kwargs)

        from langchain.embeddings import init_embeddings

        if width:
            kwargs.setdefault("dimensions", int(width))
        return init_embeddings(name, provider=provider or None, **kwargs)
    except Exception:
        return None


class Embedder:
    """Embeds text via a LangChain ``Embeddings``, falling back locally.

    Results are cached by text, because the memory bank re-embeds the same
    items on every retrieval round and the calls are neither free nor fast.
    """

    def __init__(
        self,
        embeddings: Any | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        dim: int | None = None,
    ) -> None:
        if embeddings is None and model is not None:
            embeddings = build_embeddings(model, dimensions=dimensions)
        self.embeddings = embeddings
        # The fallback encoder emits the same width as the remote model by
        # default. A narrower fallback would produce vectors that no vector
        # index can hold, so a provider outage mid-run would quietly leave
        # part of the bank unsearchable rather than merely less accurate.
        self.dim = int(dim if dim is not None else (dimensions or _default_width()))
        self._cache: dict[tuple[str, str], list[float]] = {}
        self.remote_available = embeddings is not None
        self.n_remote = 0
        self.n_fallback = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text that will be *stored* and later searched over."""
        return self._embed(list(texts), kind="document")

    def embed_query(self, text: str) -> list[float]:
        """Embed text that is *searching* the bank."""
        return self._embed([text], kind="query")[0]

    # `embed` / `embed_one` keep the document side, which is what every
    # existing caller meant by them.
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        keys = [(kind, t) for t in texts]
        missing = [t for key, t in zip(keys, texts, strict=True) if key not in self._cache]

        if missing and self.remote_available and self.embeddings is not None:
            try:
                if kind == "query":
                    # `embed_query` carries RETRIEVAL_QUERY; `embed_documents`
                    # carries RETRIEVAL_DOCUMENT. Calling the document method
                    # for a query -- as this did before -- silently discards
                    # the asymmetry the model was trained to exploit.
                    vectors = [self.embeddings.embed_query(t) for t in missing]
                else:
                    vectors = self.embeddings.embed_documents(missing)
            except Exception:
                vectors = None
            if vectors and len(vectors) == len(missing):
                for text, vec in zip(missing, vectors, strict=True):
                    # Normalized on arrival so the bank stays coherent even
                    # across an embedding-model switch. See the module note.
                    self._cache[(kind, text)] = l2_normalize(vec)
                self.n_remote += len(missing)
                missing = []
            else:
                # One failure disables the remote path for the rest of the run.
                # Retrying per item would turn a provider outage into a very
                # slow run instead of a slightly worse one.
                self.remote_available = False

        for text in missing:
            # The local encoder has no notion of task type, so both kinds map
            # to the same vector -- correct, since it is a symmetric encoder.
            self._cache[(kind, text)] = hashed_embedding(text, self.dim)
            self.n_fallback += 1

        return [self._cache[key] for key in keys]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, tolerant of dimension mismatch.

    Vectors can genuinely differ in width within one bank: items embedded
    remotely before an outage sit alongside items embedded locally after it,
    or a bank predates a change in the configured embedding width.

    Comparing over the shared prefix keeps retrieval running rather than
    raising mid-run -- and for a Matryoshka-trained model it is the *correct*
    comparison, not merely a survivable one: the leading dimensions of a
    3072-wide vector are exactly what the model would have returned at 768.
    Renormalizing over the prefix, as this does, is what makes that hold.
    Across genuinely different embedding spaces the number is meaningless but
    bounded, which is the intended failure mode.
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
