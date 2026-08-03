"""Value types shared by every stage.

Everything here is frozen. A result that can be mutated after the fact is a result
that cannot be trusted to correspond to the run that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Stance(str, Enum):
    """What a single quote does to a claim."""

    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    NEITHER = "neither"


class Verdict(str, Enum):
    """The decision returned for a claim."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class Flag(str, Enum):
    """Non-fatal observations attached to a result."""

    CONFLICTING_EVIDENCE = "conflicting_evidence"
    LOW_COVERAGE = "low_coverage"


@dataclass(frozen=True)
class Document:
    """One arXiv abstract, at one specific version.

    ``doc_id`` carries the version suffix (``2307.03172v3``) because two versions of a
    paper are two documents: a revision that changes a finding must not be collapsed
    into its predecessor.
    """

    doc_id: str
    title: str
    authors: tuple[str, ...]
    published: str
    updated: str
    abstract: str

    @property
    def base_id(self) -> str:
        """The arXiv ID without its version suffix, e.g. ``2307.03172``."""
        head, sep, tail = self.doc_id.rpartition("v")
        return head if sep and tail.isdigit() else self.doc_id


@dataclass(frozen=True)
class Chunk:
    """A retrievable span of a document, with offsets into ``Document.abstract``."""

    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Query:
    """One retrieval query derived from a claim.

    ``kind`` is retained for the explain command: knowing that the only hit came from
    the negation query is the difference between a tuned expander and a lucky one.
    """

    text: str
    kind: str = "claim"


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with the score a retriever gave it, plus where the score came from."""

    chunk: Chunk
    score: float
    source: str = "unknown"


@dataclass(frozen=True)
class RawEvidence:
    """An evidence item straight from the model, before span validation.

    This type exists so that unvalidated model output is never mistakable for
    validated evidence: the pipeline cannot accidentally return one where the other
    is expected, because they are different types.
    """

    doc_id: str
    quote: str
    stance: Stance


@dataclass(frozen=True)
class RawVerdict:
    """Pass-1 output: a proposed verdict plus the spans it claims to rest on."""

    verdict: Verdict
    evidence: tuple[RawEvidence, ...] = ()
    reasoning: str = ""


@dataclass(frozen=True)
class Evidence:
    """A quote proven to be literally present in the document it cites."""

    doc_id: str
    quote: str
    stance: Stance
    char_start: int
    char_end: int


@dataclass(frozen=True)
class RejectedSpan:
    """A span the validator dropped, and why.

    Kept on the trace rather than discarded: the rejection rate over time is how
    prompt drift is detected before it reaches verdict quality.
    """

    doc_id: str
    quote: str
    reason: str


@dataclass(frozen=True)
class RetrievalTrace:
    """Per-stage retrieval detail for the explain command."""

    queries: tuple[Query, ...] = ()
    per_query_ranks: tuple[tuple[str, tuple[str, ...]], ...] = ()
    fused_ranks: tuple[str, ...] = ()
    reranked_ranks: tuple[str, ...] = ()


@dataclass(frozen=True)
class Trace:
    """Everything needed to attribute a bad verdict to a stage."""

    retrieval: RetrievalTrace = field(default_factory=RetrievalTrace)
    passages: tuple[Chunk, ...] = ()
    raw_verdict: RawVerdict | None = None
    accepted: tuple[Evidence, ...] = ()
    rejected: tuple[RejectedSpan, ...] = ()
    audit_stances: tuple[tuple[str, Stance], ...] = ()


@dataclass(frozen=True)
class Stats:
    """Per-claim counters, surfaced during ordinary interactive use."""

    passages_retrieved: int = 0
    quotes_proposed: int = 0
    quotes_accepted: int = 0
    quotes_rejected: int = 0
    llm_calls: int = 0
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class VerificationResult:
    """The single public result shape.

    ``retrieved`` looks like a debugging leak and is the most important field for
    evaluation: without it, a retriever miss and a verifier misread are
    indistinguishable.
    """

    claim: str
    verdict: Verdict
    evidence: tuple[Evidence, ...] = ()
    flags: tuple[Flag, ...] = ()
    retrieved: tuple[str, ...] = ()
    stats: Stats = field(default_factory=Stats)
    trace: Trace | None = None
