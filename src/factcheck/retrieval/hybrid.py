"""Hybrid retrieval as composition, and the pipeline that wires retrieval together.

``HybridRetriever`` is the concrete instance of the design decision: a multi-source
retriever is itself a ``Retriever``, so "swap the retrieval mechanism" and "go from
one retriever to three" are the same operation and nothing upstream has to know
which one happened. Adding a SPLADE or graph retriever later is a list entry here,
not a change to ``RetrievalPipeline`` or to ``FactChecker`` above it.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..ports import Fusion, QueryExpander, Reranker, Retriever
from ..types import Query, RetrievalTrace, ScoredChunk


class HybridRetriever:
    """A retriever composed of a list of retrievers plus a fusion strategy.

    A sub-retriever that raises is allowed to take the whole search down with it.
    This system's house style is to fail loudly at boundaries rather than degrade
    silently -- a stale index, a mismatched embedding model, and a short corpus are
    all treated as errors rather than as "answer with what's left" -- and a
    retriever miss is exactly that kind of failure: swallowing it would produce a
    plausible-looking but silently incomplete result (e.g. BM25 broke and only
    dense results came back) that is invisible until an eval catches it. Letting
    it propagate keeps that failure visible at the boundary where it happened.
    """

    def __init__(self, retrievers: Sequence[Retriever], fusion: Fusion) -> None:
        self._retrievers = tuple(retrievers)
        self._fusion = fusion

    def search(self, query: Query, k: int) -> list[ScoredChunk]:
        ranked_lists = [retriever.search(query, k) for retriever in self._retrievers]
        return self._fusion.fuse(ranked_lists, k)


class RetrievalPipeline:
    """Composes expander -> retriever -> fusion-across-queries -> reranker.

    ``retrieve`` is the method ``FactChecker`` calls, and its signature is the
    contract: a claim in, the passages the verifier will see plus a
    ``RetrievalTrace`` out. The trace exists so the interactive ``:why`` command
    can attribute a bad answer to a specific stage instead of re-rolling the claim
    and hoping.
    """

    def __init__(
        self,
        expander: QueryExpander,
        retriever: Retriever,
        fusion: Fusion,
        reranker: Reranker,
        *,
        k_per_query: int = 50,
        top_n: int = 8,
    ) -> None:
        self._expander = expander
        self._retriever = retriever
        self._fusion = fusion
        self._reranker = reranker
        self._k_per_query = k_per_query
        self._top_n = top_n

    def retrieve(self, claim: str) -> tuple[list[ScoredChunk], RetrievalTrace]:
        queries = self._expander.expand(claim)

        per_query_results: list[list[ScoredChunk]] = []
        per_query_ranks: list[tuple[str, tuple[str, ...]]] = []
        for query in queries:
            results = self._retriever.search(query, self._k_per_query)
            per_query_results.append(results)
            per_query_ranks.append(
                (query.text, tuple(scored.chunk.chunk_id for scored in results))
            )

        # One claim can expand into several queries (claim, negation, sub-claims);
        # each is retrieved independently and their ranked lists are fused the
        # same way a hybrid retriever fuses across retrievers -- rank fusion
        # doesn't care whether the lists being combined came from different
        # backends or different queries against the same backend.
        fused = self._fusion.fuse(per_query_results, self._k_per_query)
        fused_ranks = tuple(scored.chunk.chunk_id for scored in fused)

        reranked = self._reranker.rerank(claim, fused, self._top_n)
        # Cap explicitly rather than trusting the reranker to have done it --
        # PassThroughReranker already truncates to top_n, but a future reranker
        # implementation must not be relied on to enforce the bound itself.
        capped = reranked[: self._top_n]
        reranked_ranks = tuple(scored.chunk.chunk_id for scored in capped)

        trace = RetrievalTrace(
            queries=tuple(queries),
            per_query_ranks=tuple(per_query_ranks),
            fused_ranks=fused_ranks,
            reranked_ranks=reranked_ranks,
        )
        return capped, trace
