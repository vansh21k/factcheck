"""A small fixed corpus and a scripted model.

The suite is hermetic: no network, no API key, no wall-clock dependence. Every test
that exercises grounding drives the single public seam, ``FactChecker.check``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from factcheck.types import Chunk, Document, Query, RetrievalTrace, ScoredChunk

# Two versions of one paper with textually different abstracts under an identical
# title and published date. This is the real corpus finding the conflict flag exists
# for, reproduced small enough to test.
LOST_V1 = Document(
    doc_id="2307.03172v1",
    title="Lost in the Middle: How Language Models Use Long Contexts",
    authors=("Nelson F. Liu", "Kevin Lin"),
    published="2023-07-06",
    updated="2023-07-06",
    abstract=(
        "We analyze the performance of language models on tasks that require identifying "
        "relevant information in their input contexts. We find that performance degrades "
        "significantly when models must access relevant information in the middle of long "
        "contexts, and that performance decreases as the input context grows longer."
    ),
)

LOST_V3 = Document(
    doc_id="2307.03172v3",
    title="Lost in the Middle: How Language Models Use Long Contexts",
    authors=("Nelson F. Liu", "Kevin Lin"),
    published="2023-07-06",
    updated="2024-01-22",
    abstract=(
        "We analyze the performance of language models on tasks that require identifying "
        "relevant information in their input contexts. Extended experiments show that "
        "performance is stable when models must access relevant information in the middle "
        "of long contexts, and that context length alone does not predict accuracy."
    ),
)

RAG_BENCH = Document(
    doc_id="2309.01431v2",
    title="Benchmarking Large Language Models in Retrieval-Augmented Generation",
    authors=("Jiawei Chen", "Hongyu Lin"),
    published="2023-09-04",
    updated="2023-10-11",
    abstract=(
        "Retrieval-Augmented Generation (RAG) is a promising approach for mitigating the "
        "hallucination of large language models. We systematically evaluate the impact of "
        "retrieval on generation quality across four fundamental abilities, and find that "
        "noise robustness remains a substantial weakness of current systems."
    ),
)

RAG_ORIGINAL = Document(
    doc_id="2005.11401v4",
    title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    authors=("Patrick Lewis", "Ethan Perez"),
    published="2020-05-22",
    updated="2021-04-12",
    abstract=(
        "We explore a general-purpose fine-tuning recipe for retrieval-augmented "
        "generation, models which combine pre-trained parametric and non-parametric "
        "memory for language generation. We set the state of the art on three open domain "
        "question answering tasks, outperforming parametric seq2seq models."
    ),
)

CORPUS = (LOST_V1, LOST_V3, RAG_BENCH, RAG_ORIGINAL)


class InMemoryDocumentStore:
    """Satisfies the ``DocumentStore`` protocol structurally, as a test double should."""

    def __init__(self, documents: Sequence[Document] = CORPUS) -> None:
        self._docs = {d.doc_id: d for d in documents}

    def get(self, doc_id: str) -> Document:
        return self._docs[doc_id]

    def iter_chunks(self) -> list[Chunk]:
        return [chunk_of(d) for d in self._docs.values()]


def chunk_of(doc: Document) -> Chunk:
    """One chunk per document, which is what the default 400-token setting produces."""
    return Chunk(
        chunk_id=f"{doc.doc_id}::0",
        doc_id=doc.doc_id,
        text=doc.abstract,
        char_start=0,
        char_end=len(doc.abstract),
    )


class FixedRetrieval:
    """A retrieval pipeline that always returns the documents it was told to."""

    def __init__(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)

    def retrieve(self, claim: str) -> tuple[list[ScoredChunk], RetrievalTrace]:
        passages = [
            ScoredChunk(chunk=chunk_of(d), score=1.0 - i * 0.1, source="rrf")
            for i, d in enumerate(self._documents)
        ]
        trace = RetrievalTrace(
            queries=(Query(claim, "claim"),),
            fused_ranks=tuple(p.chunk.chunk_id for p in passages),
            reranked_ranks=tuple(p.chunk.chunk_id for p in passages),
        )
        return passages, trace


class ScriptedLLM:
    """Replays predetermined responses in order, recording what it was asked.

    Records the prompts too, because two grounding requirements are about what the
    model was *not* shown: the verifier must see only retrieved passages, and the
    auditor must see neither the retrieval context nor the first pass's reasoning.
    """

    def __init__(self, responses: Sequence[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {"system": system, "user": user, "model": model, "temperature": temperature}
        )
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        response = self._responses.pop(0)
        return response(self.calls[-1]) if callable(response) else response


@pytest.fixture
def store() -> InMemoryDocumentStore:
    return InMemoryDocumentStore()
