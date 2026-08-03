"""One shared contract per port, run against every shipped implementation.

The point is that a new retrieval or verification backend is validated against the
same checks as the existing ones rather than against its own bespoke tests -- which
is the only way "swap the implementation" stays a real claim rather than an
aspiration. Kept deliberately small: these assert the contract, not the behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from conftest import CORPUS, RAG_BENCH, ScriptedLLM, chunk_of
from factcheck.config import Config
from factcheck.ports import AggregationPolicy, Fusion, QueryExpander, Reranker
from factcheck.retrieval.expand import IdentityExpander, NegationAwareExpander
from factcheck.retrieval.fusion import ReciprocalRankFusion, WeightedScoreFusion
from factcheck.retrieval.rerank import PassThroughReranker
from factcheck.types import Evidence, Query, ScoredChunk, Stance
from factcheck.verification.aggregate import ContradictionWins, MajorityStance

CFG = Config().query
CANDIDATES = [
    ScoredChunk(chunk_of(doc), score=1.0 - i * 0.1, source="t")
    for i, doc in enumerate(CORPUS)
]


def expanders() -> list[QueryExpander]:
    return [IdentityExpander(), NegationAwareExpander(ScriptedLLM([{}]), CFG.expander)]


def fusions() -> list[Fusion]:
    return [ReciprocalRankFusion(k=60), WeightedScoreFusion()]


def policies() -> list[AggregationPolicy]:
    return [ContradictionWins(CFG.aggregation), MajorityStance(CFG.aggregation)]


@pytest.mark.parametrize("expander", expanders(), ids=lambda e: type(e).__name__)
def test_every_expander_returns_at_least_one_query(expander: QueryExpander) -> None:
    """A claim that produces no queries retrieves nothing and is silently unknown."""
    queries = expander.expand("RAG reduces hallucination")

    assert len(queries) >= 1
    assert all(isinstance(q, Query) and q.text.strip() for q in queries)


@pytest.mark.parametrize("fusion", fusions(), ids=lambda f: type(f).__name__)
def test_every_fusion_respects_k_and_dedupes_by_chunk(fusion: Fusion) -> None:
    fused = fusion.fuse([CANDIDATES, list(reversed(CANDIDATES))], k=2)

    assert len(fused) == 2
    assert len({s.chunk.chunk_id for s in fused}) == 2


@pytest.mark.parametrize("fusion", fusions(), ids=lambda f: type(f).__name__)
def test_every_fusion_tolerates_empty_input(fusion: Fusion) -> None:
    assert fusion.fuse([], k=5) == []
    assert fusion.fuse([[]], k=5) == []


@pytest.mark.parametrize("reranker", [PassThroughReranker(CFG.rerank)], ids=["PassThrough"])
def test_every_reranker_caps_at_k(reranker: Reranker) -> None:
    """The cap is what keeps context size and cost bounded as the corpus grows."""
    assert len(reranker.rerank("claim", CANDIDATES, k=2)) <= 2


@pytest.mark.parametrize("policy", policies(), ids=lambda p: type(p).__name__)
def test_every_policy_abstains_without_a_supporting_stance(
    policy: AggregationPolicy,
) -> None:
    """A policy may choose among verdicts; it may not invent one from nothing."""
    neither: Sequence[Evidence] = [
        Evidence(RAG_BENCH.doc_id, "some verbatim span from the abstract", Stance.NEITHER, 0, 10)
    ]

    assert policy.decide(neither).value == "unknown"
