"""``FactChecker.check`` -- the one public entry point.

The CLI and the evaluation harness are thin callers of this method, so the harness
measures the exact path interactive use drives. Everything else in the package is
internal and expected to change.

Two invariants live here rather than behind a port, because a swappable invariant is
not one:

* no surviving evidence -> unknown, checked above ``AggregationPolicy``
* every returned quote has passed ``SpanValidator``
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

from ..config import Config
from ..llm.client import CallCounter, CountingLLM
from ..ports import (
    AggregationPolicy,
    Auditor,
    DocumentStore,
    LLMClient,
    Verifier,
)
from ..types import (
    Chunk,
    Document,
    Evidence,
    Flag,
    RawEvidence,
    RetrievalTrace,
    ScoredChunk,
    Stance,
    Stats,
    Trace,
    Verdict,
    VerificationResult,
)
from .aggregate import get_policy
from .auditor import LLMAuditor, NoopAuditor
from .validator import SpanValidator
from .verifier import LLMVerifier


class RetrievalPipeline(Protocol):
    def retrieve(self, claim: str) -> tuple[list[ScoredChunk], RetrievalTrace]:
        """Return the passages the verifier will see, plus how they were chosen."""


class FactChecker:
    def __init__(
        self,
        *,
        retrieval: RetrievalPipeline,
        verifier: Verifier,
        auditor: Auditor,
        validator: SpanValidator,
        policy: AggregationPolicy,
        document_store: DocumentStore,
        call_counter: CallCounter | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._verifier = verifier
        self._auditor = auditor
        self._validator = validator
        self._policy = policy
        self._store = document_store
        self._counter = call_counter

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        retrieval: RetrievalPipeline,
        document_store: DocumentStore,
        verifier_llm: LLMClient,
        auditor_llm: LLMClient,
        call_counter: CallCounter | None = None,
    ) -> FactChecker:
        """Assemble the default implementations named by ``cfg``."""
        counter = call_counter or CallCounter()
        query = cfg.query
        return cls(
            retrieval=retrieval,
            verifier=LLMVerifier(CountingLLM(verifier_llm, counter), query.verifier),
            auditor=(
                LLMAuditor(CountingLLM(auditor_llm, counter), query.auditor)
                if query.auditor.enabled
                else NoopAuditor()
            ),
            validator=SpanValidator(query.validator),
            policy=get_policy(query.aggregation),
            document_store=document_store,
            call_counter=counter,
        )

    # ---------------------------------------------------------------- #

    def check(self, claim: str) -> VerificationResult:
        started = time.monotonic()
        calls_before = self._counter.calls if self._counter else 0

        passages, retrieval_trace = self._retrieval.retrieve(claim)
        chunks = [scored.chunk for scored in passages]
        retrieved = _distinct_doc_ids(chunks)

        # Nothing retrieved means nothing to say. Short-circuiting here also avoids
        # paying for a model call that could only ever answer from its own weights.
        if not chunks:
            return self._result(
                claim, Verdict.UNKNOWN, (), (), retrieved, started, calls_before,
                Trace(retrieval=retrieval_trace),
            )

        raw = self._verifier.adjudicate(claim, chunks)
        documents = self._documents_for(retrieved)
        outcome = self._validator.validate(raw.evidence, documents, retrieved)

        audited, stances = self._audit(claim, outcome.accepted)
        trace = Trace(
            retrieval=retrieval_trace,
            passages=tuple(chunks),
            raw_verdict=raw,
            accepted=outcome.accepted,
            rejected=outcome.rejected,
            audit_stances=stances,
        )

        # The short-circuit sits above the policy: the policy chooses among verdicts,
        # it does not choose whether evidence was required.
        if not audited:
            return self._result(
                claim, Verdict.UNKNOWN, (), outcome.rejected, retrieved, started,
                calls_before, trace,
            )

        verdict = self._policy.decide(audited)
        flags = _flags(audited, chunks)
        return self._result(
            claim, verdict, audited, outcome.rejected, retrieved, started, calls_before,
            trace, flags,
        )

    # ---------------------------------------------------------------- #

    def _audit(
        self, claim: str, accepted: Sequence[Evidence]
    ) -> tuple[tuple[Evidence, ...], tuple[tuple[str, Stance], ...]]:
        """Re-stance every surviving quote, dropping the ones that settle nothing."""
        kept: list[Evidence] = []
        stances: list[tuple[str, Stance]] = []
        for item in accepted:
            stance = self._auditor.audit(claim, RawEvidence(item.doc_id, item.quote, item.stance))
            stances.append((item.quote, stance))
            if stance is not Stance.NEITHER:
                kept.append(
                    item if stance is item.stance else Evidence(
                        doc_id=item.doc_id,
                        quote=item.quote,
                        stance=stance,
                        char_start=item.char_start,
                        char_end=item.char_end,
                    )
                )
        return tuple(kept), tuple(stances)

    def _documents_for(self, doc_ids: Sequence[str]) -> dict[str, Document]:
        documents: dict[str, Document] = {}
        for doc_id in doc_ids:
            try:
                documents[doc_id] = self._store.get(doc_id)
            except KeyError:
                # A retrieved chunk whose document is missing from the store means the
                # index is out of step with the corpus. Every quote citing it fails
                # validation, which is the conservative outcome.
                continue
        return documents

    def _result(
        self,
        claim: str,
        verdict: Verdict,
        evidence: Sequence[Evidence],
        rejected: Sequence[object],
        retrieved: Sequence[str],
        started: float,
        calls_before: int,
        trace: Trace,
        flags: Sequence[Flag] = (),
    ) -> VerificationResult:
        proposed = len(trace.accepted) + len(trace.rejected)
        return VerificationResult(
            claim=claim,
            verdict=verdict,
            evidence=tuple(evidence),
            flags=tuple(flags),
            retrieved=tuple(retrieved),
            stats=Stats(
                passages_retrieved=len(trace.passages),
                quotes_proposed=proposed,
                quotes_accepted=len(trace.accepted),
                quotes_rejected=len(rejected),
                llm_calls=(self._counter.calls - calls_before) if self._counter else 0,
                elapsed_s=round(time.monotonic() - started, 4),
            ),
            trace=trace,
        )


def _distinct_doc_ids(chunks: Sequence[Chunk]) -> tuple[str, ...]:
    """Retrieval order, deduplicated. Order matters: recall@k reads off this field."""
    seen: dict[str, None] = {}
    for chunk in chunks:
        seen.setdefault(chunk.doc_id, None)
    return tuple(seen)


def _flags(evidence: Sequence[Evidence], chunks: Sequence[Chunk]) -> tuple[Flag, ...]:
    flags: list[Flag] = []
    stances = {e.stance for e in evidence}
    if Stance.ENTAILS in stances and Stance.CONTRADICTS in stances:
        # Surfaced rather than resolved: two versions of one paper can legitimately
        # disagree, and silently picking a side misreports the corpus.
        flags.append(Flag.CONFLICTING_EVIDENCE)
    if len({e.doc_id for e in evidence}) == 1 and len(chunks) > 1:
        # Several documents were on the table and the verdict rests on one of them.
        # Not wrong, but worth knowing before acting on it.
        flags.append(Flag.LOW_COVERAGE)
    return tuple(flags)
