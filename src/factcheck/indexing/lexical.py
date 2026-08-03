"""BM25 lexical retrieval, split across the offline/online boundary.

Claims quote exact tokens -- benchmark names ("GPT-4"), dataset/paper identifiers
("2309.01431"), acronyms ("BM25") -- that a semantic embedding can blur together but
exact lexical overlap catches perfectly. That is the entire reason this module
exists alongside dense retrieval rather than instead of it.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from ..config import BM25Config
from ..types import Chunk, Query, ScoredChunk

LEXICAL_FILENAME = "lexical.pkl"

# Lowercase alphanumeric runs, tolerating an internal hyphen or dot so "GPT-4" and
# "2309.01431" survive as single tokens instead of splitting into meaningless
# fragments. A trailing dot (end of sentence) is not swallowed, since it must be
# followed by another alphanumeric run to extend the token.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Builder:
    """Offline: tokenize every chunk, fit BM25Okapi, pickle it alongside the chunks
    it was built from so query time can recover them without touching chunks.jsonl.
    """

    name = "bm25"

    def __init__(self, cfg: BM25Config) -> None:
        self._cfg = cfg

    def build(self, chunks: Sequence[Chunk], out_dir: Path) -> dict[str, Any]:
        tokenized = [tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized, k1=self._cfg.k1, b=self._cfg.b)
        records = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for c in chunks
        ]
        payload = {"bm25": bm25, "records": records}
        with (out_dir / LEXICAL_FILENAME).open("wb") as f:
            pickle.dump(payload, f)
        return {
            "path": LEXICAL_FILENAME,
            "k1": self._cfg.k1,
            "b": self._cfg.b,
            "n_chunks": len(chunks),
        }


class BM25Loader:
    """Online: unpickle and hand back a `Retriever`. No corpus recomputation."""

    name = "bm25"

    def __init__(self, cfg: BM25Config | None = None) -> None:
        self._cfg = cfg or BM25Config()

    def load(self, out_dir: Path, manifest: dict[str, Any]) -> _BM25Retriever:
        with (out_dir / LEXICAL_FILENAME).open("rb") as f:
            payload = pickle.load(f)
        chunks = [Chunk(**r) for r in payload["records"]]
        return _BM25Retriever(payload["bm25"], chunks, self._cfg.min_score)


class _BM25Retriever:
    def __init__(
        self, bm25: BM25Okapi, chunks: Sequence[Chunk], min_score: float = 0.0
    ) -> None:
        self._bm25 = bm25
        self._chunks = list(chunks)
        self._min_score = min_score

    def search(self, query: Query, k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query.text))
        order = scores.argsort()[::-1]
        results: list[ScoredChunk] = []
        for idx in order:
            score = float(scores[idx])
            # A score at or below the floor is not weak evidence -- it is incidental
            # overlap on common words. Ranking it anyway is how a claim the corpus
            # has nothing to say about still fills every slot the verifier has.
            if score <= 0.0 or score < self._min_score:
                break
            results.append(ScoredChunk(chunk=self._chunks[idx], score=score, source="bm25"))
            if len(results) >= k:
                break
        return results
