"""Combine multiple ranked lists into one.

Fusion is its own port -- not a private detail of the hybrid retriever -- so that
reciprocal rank fusion can be measured against an alternative rather than assumed
correct. A single hardcoded strategy could never earn an ablation row; two
strategies behind one interface can.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..config import FusionConfig
from ..errors import ConfigError
from ..ports import Fusion
from ..types import Chunk, ScoredChunk


class ReciprocalRankFusion:
    """Fuses by rank, never by score.

    BM25 scores and cosine similarities live on incomparable scales -- a BM25 score
    of 12 and a cosine similarity of 0.4 cannot be added meaningfully, and any
    weighting between them is a per-corpus tuning parameter in disguise. Rank
    fusion sidesteps the whole problem: a chunk's contribution is ``1 / (k +
    rank)``, so only its ordinal position in each list matters, never its raw
    score. This is also why ``fuse`` never receives or reads ``ScoredChunk.score``.
    """

    def __init__(self, k: int = 60) -> None:
        self._k = k

    def fuse(self, ranked_lists: Sequence[Sequence[ScoredChunk]], k: int) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for ranked in ranked_lists:
            for rank, scored in enumerate(ranked, start=1):
                chunk_id = scored.chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self._k + rank)
                chunks.setdefault(chunk_id, scored.chunk)

        # Sort by score descending, then by chunk_id ascending so a tie always
        # resolves the same way -- the whole point of moving off raw scores is a
        # reproducible order, and an unstable tiebreak would quietly undo that.
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            ScoredChunk(chunk=chunks[chunk_id], score=score, source="rrf")
            for chunk_id, score in ordered[:k]
        ]


class WeightedScoreFusion:
    """Fuses by a weighted sum of raw scores.

    Exists to give ``ReciprocalRankFusion`` something to be compared against --
    the score-blending alternative RRF was chosen over. Genuinely functional, not
    a stub: per-list weights default to 1.0 when unspecified or when a list
    outnumbers the configured weights.
    """

    def __init__(self, weights: Sequence[float] = ()) -> None:
        self._weights = tuple(weights)

    def fuse(self, ranked_lists: Sequence[Sequence[ScoredChunk]], k: int) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for i, ranked in enumerate(ranked_lists):
            weight = self._weights[i] if i < len(self._weights) else 1.0
            for scored in ranked:
                chunk_id = scored.chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * scored.score
                chunks.setdefault(chunk_id, scored.chunk)

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            ScoredChunk(chunk=chunks[chunk_id], score=score, source="weighted_score")
            for chunk_id, score in ordered[:k]
        ]


def get_fusion(cfg: FusionConfig) -> Fusion:
    """Registry lookup, so adding a strategy is a config value, not a call-site edit."""
    if cfg.strategy == "rrf":
        return ReciprocalRankFusion(k=cfg.rrf_k)
    if cfg.strategy == "weighted_score":
        return WeightedScoreFusion()
    raise ConfigError(
        f"unknown fusion strategy '{cfg.strategy}'. Known strategies: 'rrf', 'weighted_score'."
    )
