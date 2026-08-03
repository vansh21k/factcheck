"""Tests for the retrieval stage: expansion, fusion, hybrid composition, reranking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from factcheck.config import ExpanderConfig, RerankConfig
from factcheck.errors import ModelCallError
from factcheck.retrieval.expand import IdentityExpander, NegationAwareExpander
from factcheck.retrieval.fusion import ReciprocalRankFusion, WeightedScoreFusion, get_fusion
from factcheck.retrieval.hybrid import HybridRetriever, RetrievalPipeline
from factcheck.retrieval.rerank import LLMListwiseReranker, PassThroughReranker
from factcheck.types import Chunk, Query, ScoredChunk

# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


def make_chunk(chunk_id: str, *, doc_id: str = "doc1", text: str = "some text") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, char_start=0, char_end=len(text))


def sc(chunk_id: str, score: float, *, doc_id: str = "doc1", source: str = "test") -> ScoredChunk:
    return ScoredChunk(chunk=make_chunk(chunk_id, doc_id=doc_id), score=score, source=source)


class FakeRetriever:
    """Returns a scripted list of ScoredChunk, ignoring the query text."""

    def __init__(self, results: list[ScoredChunk]) -> None:
        self._results = results

    def search(self, query: Query, k: int) -> list[ScoredChunk]:
        return self._results[:k]


class RaisingRetriever:
    def search(self, query: Query, k: int) -> list[ScoredChunk]:
        raise RuntimeError("retriever exploded")


class FakeLLM:
    """A scripted LLMClient. Feed one fixed response, or a queue of them."""

    def __init__(
        self,
        response: Any = None,
        *,
        responses: list[Any] | None = None,
    ) -> None:
        self._response = response
        self._responses = responses
        self._calls = 0

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        tool: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        self._calls += 1
        result = self._responses[self._calls - 1] if self._responses is not None else self._response
        if isinstance(result, BaseException):
            raise result
        return result  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# ReciprocalRankFusion
# --------------------------------------------------------------------------- #


def test_rrf_discriminates_rank_from_score() -> None:
    # c3 is ranked 3rd in both lists but has a tiny raw score; c1 is ranked 1st
    # in only one list but has a huge raw score. Score blending would favor c1;
    # rank fusion must favor c3, because it never looks at .score at all.
    list_a = [sc("c1", 100.0), sc("c2", 50.0), sc("c3", 1.0)]
    list_b = [sc("c4", 100.0), sc("c5", 50.0), sc("c3", 1.0)]

    fusion = ReciprocalRankFusion(k=2)
    fused = fusion.fuse([list_a, list_b], k=10)

    assert fused[0].chunk.chunk_id == "c3"


def test_rrf_is_score_scale_invariant() -> None:
    list_a = [sc("c1", 5.0), sc("c2", 3.0), sc("c3", 1.0)]
    list_b = [sc("c3", 5.0), sc("c1", 3.0), sc("c2", 1.0)]

    fusion = ReciprocalRankFusion(k=60)
    order_before = [s.chunk.chunk_id for s in fusion.fuse([list_a, list_b], k=10)]

    list_b_scaled = [sc(s.chunk.chunk_id, s.score * 1000) for s in list_b]
    order_after = [s.chunk.chunk_id for s in fusion.fuse([list_a, list_b_scaled], k=10)]

    assert order_before == order_after


def test_rrf_dedupes_by_chunk_id() -> None:
    list_a = [sc("c1", 10.0), sc("c2", 5.0)]
    list_b = [sc("c1", 1.0), sc("c3", 8.0)]

    fusion = ReciprocalRankFusion(k=60)
    fused = fusion.fuse([list_a, list_b], k=10)

    ids = [s.chunk.chunk_id for s in fused]
    assert len(ids) == len(set(ids))
    assert set(ids) == {"c1", "c2", "c3"}
    assert all(s.source == "rrf" for s in fused)


def test_rrf_ties_broken_deterministically() -> None:
    # c1 and c2 each appear only once, both at rank 1 in separate lists -- equal
    # fused score. The tiebreak (chunk_id ascending) must be stable across runs.
    list_a = [sc("c2", 1.0)]
    list_b = [sc("c1", 1.0)]

    fusion = ReciprocalRankFusion(k=60)
    order1 = [s.chunk.chunk_id for s in fusion.fuse([list_a, list_b], k=10)]
    order2 = [s.chunk.chunk_id for s in fusion.fuse([list_a, list_b], k=10)]

    assert order1 == order2 == ["c1", "c2"]


def test_weighted_score_fusion_is_a_working_second_strategy() -> None:
    list_a = [sc("c1", 10.0)]
    list_b = [sc("c2", 1.0)]

    fusion = WeightedScoreFusion(weights=(1.0, 100.0))
    fused = fusion.fuse([list_a, list_b], k=10)

    assert fused[0].chunk.chunk_id == "c2"
    assert fused[0].source == "weighted_score"


def test_get_fusion_registry() -> None:
    from factcheck.config import FusionConfig
    from factcheck.errors import ConfigError

    assert isinstance(get_fusion(FusionConfig(strategy="rrf", rrf_k=10)), ReciprocalRankFusion)
    assert isinstance(get_fusion(FusionConfig(strategy="weighted_score")), WeightedScoreFusion)
    with pytest.raises(ConfigError):
        get_fusion(FusionConfig(strategy="nonexistent"))


# --------------------------------------------------------------------------- #
# HybridRetriever
# --------------------------------------------------------------------------- #


def test_hybrid_retriever_satisfies_retriever_and_composes() -> None:
    from factcheck.ports import Retriever

    r1 = FakeRetriever([sc("a", 1.0), sc("b", 0.5)])
    r2 = FakeRetriever([sc("b", 1.0), sc("c", 0.5)])
    fusion = ReciprocalRankFusion(k=60)

    hybrid = HybridRetriever([r1, r2], fusion)
    assert isinstance(hybrid, Retriever)

    # A hybrid of hybrids: nothing upstream can tell the difference.
    outer = HybridRetriever([hybrid, FakeRetriever([sc("d", 1.0)])], fusion)
    assert isinstance(outer, Retriever)

    results = outer.search(Query("claim"), k=10)
    ids = {s.chunk.chunk_id for s in results}
    assert ids == {"a", "b", "c", "d"}


def test_hybrid_retriever_propagates_a_failing_retriever() -> None:
    # House style is fail-loud: a sub-retriever error is not swallowed, even if
    # other sub-retrievers would have succeeded.
    good = FakeRetriever([sc("a", 1.0)])
    bad = RaisingRetriever()
    hybrid = HybridRetriever([good, bad], ReciprocalRankFusion())

    with pytest.raises(RuntimeError):
        hybrid.search(Query("claim"), k=10)


# --------------------------------------------------------------------------- #
# NegationAwareExpander / IdentityExpander
# --------------------------------------------------------------------------- #


def test_identity_expander_produces_exactly_one_query() -> None:
    queries = IdentityExpander().expand("Paris is the capital of France.")
    assert queries == [Query("Paris is the capital of France.", kind="claim")]


def test_negation_expander_includes_claim_and_negation() -> None:
    claim = "Vaccines cause autism."
    llm = FakeLLM({"negation": "Vaccines do not cause autism.", "subclaims": []})
    expander = NegationAwareExpander(llm, ExpanderConfig(include_negation=True, n_queries=3))

    queries = expander.expand(claim)

    assert queries[0] == Query(claim, kind="claim")
    assert any(q.kind == "negation" and q.text == "Vaccines do not cause autism." for q in queries)


def test_negation_expander_skips_negation_when_disabled() -> None:
    claim = "The sky is blue."
    llm = FakeLLM({"negation": "The sky is not blue.", "subclaims": []})
    expander = NegationAwareExpander(llm, ExpanderConfig(include_negation=False))

    queries = expander.expand(claim)

    assert all(q.kind != "negation" for q in queries)


@pytest.mark.parametrize(
    "bad_response",
    ["not a dict at all", []],
    ids=["malformed-string", "empty-list"],
)
def test_expander_degrades_to_identity_on_malformed_output(bad_response: Any) -> None:
    """A call that *succeeds* with an unusable shape degrades gracefully."""
    claim = "Water boils at 100 degrees Celsius at sea level."
    llm = FakeLLM(bad_response)
    expander = NegationAwareExpander(llm, ExpanderConfig())

    queries = expander.expand(claim)

    assert queries == [Query(claim, kind="claim")]


def test_expander_propagates_model_call_error() -> None:
    """A call that *fails* (rate limit, network, quota) must not be swallowed.

    The negation query is a correctness requirement, not a recall optimization
    (see NegationAwareExpander's docstring): silently degrading to identity on a
    transient API failure would silently disable the one query path
    `contradicted` depends on, with no signal anywhere in the result. Every
    other stage in this pipeline lets `ModelCallError` propagate to the CLI
    boundary instead of swallowing it, and this stage now matches.
    """
    claim = "Water boils at 100 degrees Celsius at sea level."
    llm = FakeLLM(ModelCallError("model call failed"))
    expander = NegationAwareExpander(llm, ExpanderConfig())

    with pytest.raises(ModelCallError):
        expander.expand(claim)


def test_expander_respects_n_queries() -> None:
    claim = "Claim text."
    llm = FakeLLM(
        {
            "negation": "Negated claim text.",
            "subclaims": ["Sub-claim one.", "Sub-claim two.", "Sub-claim three."],
        }
    )
    expander = NegationAwareExpander(llm, ExpanderConfig(n_queries=3, include_negation=True))

    queries = expander.expand(claim)

    assert len(queries) == 3
    assert [q.kind for q in queries] == ["claim", "negation", "subclaim"]


def test_expander_dedupes_case_insensitively_keeping_first_kind() -> None:
    claim = "The Earth is round."
    # The model echoes the claim back verbatim (mixed case) as its "negation" --
    # a bad model output, but one that must not produce a duplicate query.
    llm = FakeLLM({"negation": "THE EARTH IS ROUND.", "subclaims": ["the earth is round."]})
    expander = NegationAwareExpander(llm, ExpanderConfig(n_queries=5, include_negation=True))

    queries = expander.expand(claim)

    assert len(queries) == 1
    assert queries[0].kind == "claim"
    assert queries[0].text == claim


# --------------------------------------------------------------------------- #
# Rerankers
# --------------------------------------------------------------------------- #


def test_pass_through_reranker_applies_min_score() -> None:
    chunks = [sc("a", 0.9), sc("b", 0.4), sc("c", 0.6)]
    reranker = PassThroughReranker(RerankConfig(min_score=0.5))

    result = reranker.rerank("claim", chunks, k=10)

    assert [s.chunk.chunk_id for s in result] == ["a", "c"]


def test_pass_through_reranker_truncates_to_k() -> None:
    chunks = [sc(str(i), 1.0) for i in range(5)]
    reranker = PassThroughReranker(RerankConfig(min_score=0.0))

    result = reranker.rerank("claim", chunks, k=2)

    assert len(result) == 2


def test_llm_listwise_reranker_reorders() -> None:
    chunks = [sc("a", 0.5), sc("b", 0.9), sc("c", 0.1)]
    llm = FakeLLM({"order": [2, 0, 1]})
    reranker = LLMListwiseReranker(llm, RerankConfig())

    result = reranker.rerank("claim", chunks, k=10)

    assert [s.chunk.chunk_id for s in result] == ["c", "a", "b"]


def test_llm_listwise_reranker_degrades_to_pass_through_on_malformed_output() -> None:
    chunks = [sc("a", 0.9), sc("b", 0.1)]
    llm = FakeLLM({"order": "not a list"})
    reranker = LLMListwiseReranker(llm, RerankConfig(min_score=0.0))

    result = reranker.rerank("claim", chunks, k=10)

    assert [s.chunk.chunk_id for s in result] == ["a", "b"]


def test_llm_listwise_reranker_degrades_on_exception() -> None:
    chunks = [sc("a", 0.9), sc("b", 0.1)]
    llm = FakeLLM(RuntimeError("boom"))
    reranker = LLMListwiseReranker(llm, RerankConfig(min_score=0.0))

    result = reranker.rerank("claim", chunks, k=10)

    assert [s.chunk.chunk_id for s in result] == ["a", "b"]


# --------------------------------------------------------------------------- #
# RetrievalPipeline
# --------------------------------------------------------------------------- #


def _make_pipeline(
    *,
    n_chunks: int = 10,
    top_n: int = 3,
    reranker: Any = None,
) -> tuple[RetrievalPipeline, Sequence[ScoredChunk]]:
    results = [sc(str(i), score=float(n_chunks - i)) for i in range(n_chunks)]
    retriever = FakeRetriever(results)
    fusion = ReciprocalRankFusion(k=60)
    rr = reranker if reranker is not None else PassThroughReranker(RerankConfig(top_n=top_n))
    pipeline = RetrievalPipeline(
        IdentityExpander(), retriever, fusion, rr, k_per_query=50, top_n=top_n
    )
    return pipeline, results


def test_pipeline_caps_passages_at_rerank_top_n() -> None:
    pipeline, _ = _make_pipeline(n_chunks=10, top_n=3)

    passages, _trace = pipeline.retrieve("some claim")

    assert len(passages) == 3


def test_pipeline_populates_trace() -> None:
    pipeline, _ = _make_pipeline(n_chunks=5, top_n=4)

    _passages, trace = pipeline.retrieve("some claim")

    assert trace.queries == (Query("some claim", kind="claim"),)
    assert len(trace.per_query_ranks) == 1
    query_text, chunk_ids = trace.per_query_ranks[0]
    assert query_text == "some claim"
    assert chunk_ids == tuple(str(i) for i in range(5))
    assert trace.fused_ranks == tuple(str(i) for i in range(5))
    assert trace.reranked_ranks == tuple(str(i) for i in range(4))
