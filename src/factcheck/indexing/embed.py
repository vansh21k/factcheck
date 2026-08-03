"""Dense embeddings, behind a protocol narrow enough to swap the backend.

`get_embedder` is the swap point named in the spec ("swapping the semantic backend
requires no change above the retrieval interface"): a `sentence-transformers` (or
hosted) backend implements `Embedder` -- `model_id`, `dim`, `encode` -- and gets
registered here under its own `EmbeddingConfig.model` string. Nothing in `dense.py`
or above it would change.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..config import EmbeddingConfig

FloatArray = npt.NDArray[np.float32]

# Matches `EmbeddingConfig.model`'s default. Recorded in the manifest and checked at
# query time against the configured model -- see `manifest.check_compatible`.
HASH_CHARNGRAM_V1 = "hash-charngram-v1"

_WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_NGRAM_N = 3


@runtime_checkable
class Embedder(Protocol):
    model_id: str
    dim: int

    def encode(self, texts: Sequence[str]) -> FloatArray:
        """Return float32 `[len(texts), dim]`, L2-normalized rows.

        Normalized rows mean cosine similarity between any two vectors is a plain
        dot product -- callers never need to know a vector came from an embedder to
        score it correctly.
        """


def _stable_hash(token: str, dim: int) -> int:
    """A hash stable across processes and interpreter runs.

    Python's built-in `hash()` is salted per-process for strings (`PYTHONHASHSEED`),
    which would make two builds of the same corpus land features in different
    buckets -- silently breaking the determinism the spec requires. blake2b has no
    such salt.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _features(text: str) -> list[str]:
    """Hashed word unigrams, word bigrams, and character trigrams.

    Character n-grams give partial credit for morphological/typo variants (plurals,
    "-ing" forms); word bigrams give a little word-order signal a bag-of-words
    lacks. All three are cheap, dependency-free, and enough to distinguish a
    paraphrase from an unrelated sentence, which is all the default backend needs to
    do -- it is not competing with a trained sentence encoder.
    """
    words = _WORD_RE.findall(text.lower())
    feats = [f"w:{w}" for w in words]
    feats.extend(f"b:{a}_{b}" for a, b in zip(words, words[1:]))
    for w in words:
        padded = f"#{w}#"
        feats.extend(f"c:{padded[i:i + _NGRAM_N]}" for i in range(len(padded) - _NGRAM_N + 1))
    return feats


class HashingEmbedder:
    """Deterministic hashed bag-of-features embedder. No training, no network.

    Sublinear (log) tf weighting keeps a term that appears 20 times from swamping
    one that appears twice; L2 normalization at the end is what makes cosine
    similarity a dot product downstream.
    """

    model_id = HASH_CHARNGRAM_V1

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> FloatArray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            for feat in _features(text):
                idx = _stable_hash(feat, self.dim)
                counts[idx] = counts.get(idx, 0) + 1
            for idx, tf in counts.items():
                out[row, idx] = 1.0 + math.log(tf)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # an all-zero row (empty text) stays zero, not NaN
        normalized: FloatArray = (out / norms).astype(np.float32)
        return normalized


def get_embedder(cfg: EmbeddingConfig) -> Embedder:
    """Registry from `EmbeddingConfig.model` to an `Embedder`.

    Raises rather than silently falling back, because an unrecognized model id at
    build time producing *some* embedder is exactly the same-dimension-different-model
    failure mode the manifest's identity check exists to catch -- better to fail here
    than to build an index no query-time config can legitimately match.
    """
    if cfg.model == HASH_CHARNGRAM_V1:
        return HashingEmbedder(dim=cfg.dim)
    # Swap point: e.g.
    #   if cfg.model == "sentence-transformers/all-MiniLM-L6-v2":
    #       return SentenceTransformerEmbedder(cfg.model, cfg.dim)
    # implementing the same three-attribute Embedder protocol, no dependency added
    # here until that backend is actually needed.
    raise ValueError(
        f"unknown embedding model {cfg.model!r}. Known models: {HASH_CHARNGRAM_V1!r}"
    )
