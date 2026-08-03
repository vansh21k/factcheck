"""Precision after fusion: cut candidates down to what the verifier actually sees.

The spec calls for a reranking stage "present in the pipeline even where it changes
little today" -- at 50 abstracts, fusion order is already close to correct, so a
reranker mostly just confirms it and truncates. The value of the stage is at the
scale this design does not (yet) run at, where fusion order is much noisier and
reordering the top-k before verification is where most of the precision gain lives.
Retrofitting that stage later -- once the pipeline, tests, and config all assume a
two-stage retrieve-then-verify shape -- costs more than shipping the interface now
and letting it be near-inert until the corpus grows into it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..config import AUDITOR_MODEL, RerankConfig
from ..ports import LLMClient, Reranker
from ..types import ScoredChunk

# Reranking is a narrow "which of these already-retrieved passages matters" call,
# the same shape as the auditor's isolated entailment check -- so it reuses the
# auditor's small model rather than the (larger, pricier) verifier model.
# RerankConfig carries no model field of its own; this is the one place that
# choice had to be made without a config knob backing it.
_RERANK_MODEL = AUDITOR_MODEL
_RERANK_TEMPERATURE = 0.0

_RERANK_TOOL: dict[str, Any] = {
    "name": "rerank_passages",
    "description": (
        "Order the candidate passages from most to least relevant to the claim."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Candidate indices (0-based), most relevant first.",
            },
        },
        "required": ["order"],
    },
}

_SYSTEM_PROMPT = (
    "You rank retrieved passages by relevance to a claim, for a fact-checking "
    "system. Call rerank_passages exactly once with every candidate index, "
    "ordered most to least relevant."
)


class PassThroughReranker:
    """Truncate to ``k`` and drop anything below ``rerank.min_score``.

    This is the reranker used when ``rerank.enabled`` is false, and it is also
    the ``LLMListwiseReranker`` fallback: filtering and truncation are exactly
    what a broken reranking call should degrade to, since they never invent an
    ordering the retriever didn't already produce.
    """

    def __init__(self, cfg: RerankConfig) -> None:
        self._min_score = cfg.min_score

    def rerank(self, claim: str, chunks: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        filtered = [c for c in chunks if c.score >= self._min_score]
        return list(filtered[:k])


class LLMListwiseReranker:
    """One LLM call, one tool, an ordering of candidate indices.

    Malformed or unusable output degrades to ``PassThroughReranker`` rather than
    failing the claim -- a bad reordering is a precision loss, not a correctness
    one, so it does not deserve the same "must always produce something" ceremony
    as query expansion. It still must not crash the request.
    """

    def __init__(self, llm: LLMClient, cfg: RerankConfig) -> None:
        self._llm = llm
        self._fallback = PassThroughReranker(cfg)

    def rerank(self, claim: str, chunks: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        try:
            listing = "\n".join(
                f"[{i}] {scored.chunk.text}" for i, scored in enumerate(chunks)
            )
            result = self._llm.complete(
                system=_SYSTEM_PROMPT,
                user=f"Claim: {claim}\n\nCandidates:\n{listing}",
                model=_RERANK_MODEL,
                temperature=_RERANK_TEMPERATURE,
                tool=_RERANK_TOOL,
                max_tokens=256,
            )
            order = _parse_order(result, len(chunks))
        except Exception:
            return self._fallback.rerank(claim, chunks, k)

        return [chunks[i] for i in order][:k]


def _parse_order(result: Any, n: int) -> list[int]:
    """Validate and de-noise the tool output, raising on anything unusable."""
    if not isinstance(result, dict):
        raise ValueError("rerank result must be a dict")
    order_raw = result.get("order")
    if not isinstance(order_raw, list):
        raise ValueError("order must be a list")

    seen: set[int] = set()
    order: list[int] = []
    for idx in order_raw:
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        if idx < 0 or idx >= n or idx in seen:
            continue
        seen.add(idx)
        order.append(idx)

    if not order:
        raise ValueError("no valid candidate indices in order")
    return order


def get_reranker(cfg: RerankConfig, llm: LLMClient | None = None) -> Reranker:
    """Registry lookup: which reranker backs the pipeline for this config."""
    if not cfg.enabled:
        return PassThroughReranker(cfg)
    if llm is None:
        raise ValueError("rerank.enabled is true but no LLMClient was provided")
    return LLMListwiseReranker(llm, cfg)
