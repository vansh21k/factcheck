"""Assembly: the one place that knows which implementation each port gets.

Everything above this module depends on protocols. Everything below is a concrete
class. Keeping the wiring here is what lets the CLI and the evaluation harness be
thin callers of the same construction, so the harness measures the path interactive
use actually drives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .corpus import corpus_hash
from .indexing.build import load_retrievers
from .indexing.manifest import Manifest
from .ingest.store import JsonlDocumentStore
from .llm.cache import CachingLLM, StageCache
from .llm.client import CallCounter, GeminiClient
from .ports import LLMClient
from .retrieval.expand import IdentityExpander, NegationAwareExpander
from .retrieval.fusion import get_fusion
from .retrieval.hybrid import HybridRetriever, RetrievalPipeline
from .retrieval.rerank import get_reranker
from .verification.checker import FactChecker


@dataclass
class Session:
    """A loaded index plus everything needed to answer claims against it.

    Carries the paths and the client it was built from so an in-session config change
    can re-wire the pipeline without the REPL having to remember how to construct one.
    """

    checker: FactChecker
    manifest: Manifest
    store: JsonlDocumentStore
    config: Config
    warnings: tuple[str, ...]
    counter: CallCounter
    index_dir: Path
    docs_dir: Path
    llm: LLMClient
    cache_dir: Path | None = None


def build_session(
    cfg: Config,
    index_dir: Path,
    docs_dir: Path,
    *,
    llm: LLMClient | None = None,
    cache_dir: Path | None = None,
) -> Session:
    """Load a prebuilt index and wire the pipeline over it.

    Never builds. ``Manifest.read`` raises ``MissingIndexError`` naming the index
    command if there is nothing there, because an implicit rebuild hides the cost
    that dominates at scale.
    """
    manifest = Manifest.read(index_dir)
    store = JsonlDocumentStore.load(docs_dir)

    # Raises on an embedding-model or index-time config mismatch; returns staleness
    # as a warning, because a stale index should be reported rather than refused.
    warnings = manifest.check_compatible(
        cfg, corpus_hash(list(store)) if len(store) else None
    )

    client = llm or GeminiClient()
    verifier_llm = _cached(client, cache_dir, "verifier")
    auditor_llm = _cached(client, cache_dir, "auditor")
    expander_llm = _cached(client, cache_dir, "expander")

    counter = CallCounter()
    retrievers = load_retrievers(index_dir, manifest, cfg)
    query = cfg.query
    pipeline = RetrievalPipeline(
        expander=(
            NegationAwareExpander(expander_llm, query.expander)
            if query.expander.enabled
            else IdentityExpander()
        ),
        retriever=HybridRetriever(list(retrievers.values()), get_fusion(query.fusion)),
        fusion=get_fusion(query.fusion),
        reranker=get_reranker(query.rerank, expander_llm),
        k_per_query=max(query.bm25.top_k, query.dense.top_k),
        top_n=query.rerank.top_n,
    )

    checker = FactChecker.from_config(
        cfg,
        retrieval=pipeline,
        document_store=store,
        verifier_llm=verifier_llm,
        auditor_llm=auditor_llm,
        call_counter=counter,
    )
    return Session(
        checker=checker,
        manifest=manifest,
        store=store,
        config=cfg,
        warnings=tuple(warnings),
        counter=counter,
        index_dir=index_dir,
        docs_dir=docs_dir,
        llm=client,
        cache_dir=cache_dir,
    )


def _cached(client: LLMClient, cache_dir: Path | None, stage: str) -> LLMClient:
    return client if cache_dir is None else CachingLLM(client, StageCache(cache_dir), stage)
