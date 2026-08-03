"""Dense (semantic) retrieval, split across the offline/online boundary.

Row `i` of `dense.npy` corresponds to line `i` of `chunks.jsonl`, by construction:
both are written from (and read back into) the same chunk-ordered sequence. That
positional contract is the whole interface between `build` and `load` here -- no
chunk id is stored in the matrix itself.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DenseConfig
from ..types import Chunk, Query, ScoredChunk
from .embed import Embedder, FloatArray

DENSE_FILENAME = "dense.npy"
CHUNKS_FILENAME = "chunks.jsonl"


class DenseBuilder:
    """Offline: embed every chunk once, write the matrix. No query-time cost here."""

    name = "dense"

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def build(self, chunks: Sequence[Chunk], out_dir: Path) -> dict[str, Any]:
        if chunks:
            vectors = self._embedder.encode([c.text for c in chunks])
        else:
            vectors = np.zeros((0, self._embedder.dim), dtype=np.float32)
        np.save(out_dir / DENSE_FILENAME, vectors)
        return {
            "path": DENSE_FILENAME,
            "model": self._embedder.model_id,
            "dim": self._embedder.dim,
            "n_chunks": len(chunks),
        }


class DenseLoader:
    """Online: memory-map the matrix, embed only the query, floor by `min_score`.

    The floor exists so a claim unrelated to the corpus retrieves nothing rather
    than the least-irrelevant chunk available -- cosine similarity always returns
    *something*, ranked, even between unrelated vectors, and "ranked" is not the
    same guarantee as "relevant".
    """

    name = "dense"

    def __init__(self, embedder: Embedder, cfg: DenseConfig) -> None:
        self._embedder = embedder
        self._cfg = cfg

    def load(self, out_dir: Path, manifest: dict[str, Any]) -> _DenseRetriever:
        vectors = np.load(out_dir / DENSE_FILENAME, mmap_mode="r")
        chunks = _read_chunks(out_dir)
        return _DenseRetriever(vectors, chunks, self._embedder, self._cfg.min_score)


class _DenseRetriever:
    def __init__(
        self,
        vectors: FloatArray,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        min_score: float,
    ) -> None:
        self._vectors = vectors
        self._chunks = list(chunks)
        self._embedder = embedder
        self._min_score = min_score

    def search(self, query: Query, k: int) -> list[ScoredChunk]:
        if self._vectors.shape[0] == 0:
            return []
        qvec = self._embedder.encode([query.text])[0]
        scores = np.asarray(self._vectors) @ qvec
        order = np.argsort(-scores)
        results: list[ScoredChunk] = []
        for idx in order:
            score = float(scores[idx])
            if score < self._min_score:
                break  # descending order: everything after this is lower too
            results.append(ScoredChunk(chunk=self._chunks[idx], score=score, source="dense"))
            if len(results) >= k:
                break
        return results


def _read_chunks(out_dir: Path) -> list[Chunk]:
    """`chunks.jsonl` is this package's own artifact (written by `build.py`), not
    the ingest side's `documents.jsonl` -- reading it here creates no dependency on
    the ingest package.
    """
    chunks: list[Chunk] = []
    with (out_dir / CHUNKS_FILENAME).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(Chunk(**obj))
    return chunks
