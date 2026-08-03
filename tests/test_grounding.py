"""The heart of the suite: behaviour that cannot be verified by inspection.

Every test here drives ``FactChecker.check`` and asserts on the returned result --
verdict, evidence, flags. None asserts on which stage produced what, how many
internal calls occurred, or the shape of intermediate structures. Internal stage
boundaries are expected to change as retrieval and verification implementations are
swapped, and a test that freezes them converts the extensibility the design asks for
into a liability.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from conftest import (
    CORPUS,
    LOST_V1,
    LOST_V3,
    RAG_BENCH,
    RAG_ORIGINAL,
    FixedRetrieval,
    InMemoryDocumentStore,
    ScriptedLLM,
)
from factcheck.config import Config
from factcheck.types import Evidence, Flag, Stance, Verdict
from factcheck.verification.auditor import NoopAuditor
from factcheck.verification.checker import FactChecker
from factcheck.verification.validator import SpanValidator
from factcheck.verification.verifier import LLMVerifier

# A verbatim span of the RAG benchmarking abstract, comfortably over the 40-char floor.
GROUNDED_QUOTE = (
    "Retrieval-Augmented Generation (RAG) is a promising approach for mitigating the "
    "hallucination of large language models"
)
CLAIM = "RAG is a promising approach for mitigating hallucination in LLMs"


def verdict_response(
    verdict: str, evidence: Sequence[dict[str, str]], reasoning: str = ""
) -> dict[str, Any]:
    return {"verdict": verdict, "evidence": list(evidence), "reasoning": reasoning}


def audit(stance: str) -> dict[str, Any]:
    return {"stance": stance}


def build(
    *,
    verifier_response: dict[str, Any],
    audit_stances: Sequence[str] = (),
    documents: Sequence[Any] = CORPUS,
    config: Config | None = None,
) -> tuple[FactChecker, ScriptedLLM, ScriptedLLM]:
    """Wire a checker over a scripted model. No network, no key, deterministic."""
    cfg = config or Config()
    verifier_llm = ScriptedLLM([verifier_response])
    audit_llm = ScriptedLLM([audit(s) for s in audit_stances])
    checker = FactChecker.from_config(
        cfg,
        retrieval=FixedRetrieval(documents),
        document_store=InMemoryDocumentStore(CORPUS),
        verifier_llm=verifier_llm,
        auditor_llm=audit_llm,
    )
    return checker, verifier_llm, audit_llm


# --------------------------------------------------------------------------- #
# span validation is the guarantee
# --------------------------------------------------------------------------- #


def test_fabricated_quote_yields_unknown_not_the_models_verdict() -> None:
    """A quote absent from the cited document cannot carry a verdict.

    This is the single most important test in the suite: the model asserts
    'supported' with confident, well-formed, entirely invented evidence.
    """
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported",
            [
                {
                    "doc_id": RAG_BENCH.doc_id,
                    "quote": "RAG eliminates hallucination entirely in all benchmarks tested",
                    "stance": "entails",
                }
            ],
        )
    )
    result = checker.check(CLAIM)

    assert result.verdict is Verdict.UNKNOWN
    assert result.evidence == ()


def test_quote_attributed_to_the_wrong_document_yields_unknown() -> None:
    """Verbatim in the corpus is not enough; it must be verbatim in the doc it cites."""
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported",
            [{"doc_id": RAG_ORIGINAL.doc_id, "quote": GROUNDED_QUOTE, "stance": "entails"}],
        )
    )
    result = checker.check(CLAIM)

    assert result.verdict is Verdict.UNKNOWN
    assert result.evidence == ()


def test_quote_citing_a_document_that_was_never_retrieved_yields_unknown() -> None:
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported",
            [{"doc_id": LOST_V1.doc_id, "quote": GROUNDED_QUOTE, "stance": "entails"}],
        ),
        documents=[RAG_BENCH],
    )
    result = checker.check(CLAIM)

    assert result.verdict is Verdict.UNKNOWN


@pytest.mark.parametrize(
    "mangled",
    [
        GROUNDED_QUOTE.replace(" ", "  "),
        GROUNDED_QUOTE.replace(" ", "\n  "),
        f"  {GROUNDED_QUOTE}  ",
    ],
    ids=["double-spaced", "wrapped", "padded"],
)
def test_whitespace_variation_is_accepted(mangled: str) -> None:
    """Trivial formatting variation is not fabrication.

    Models reflow quotes out of wrapped source text constantly; rejecting that would
    make the gate fire on the honest case and teach nobody anything.
    """
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported", [{"doc_id": RAG_BENCH.doc_id, "quote": mangled, "stance": "entails"}]
        ),
        audit_stances=["entails"],
    )
    result = checker.check(CLAIM)

    assert result.verdict is Verdict.SUPPORTED
    assert len(result.evidence) == 1


def test_typographic_quote_characters_are_accepted() -> None:
    doc = RAG_ORIGINAL
    verbatim = "models which combine pre-trained parametric and non-parametric memory"
    curly = verbatim.replace("pre-trained", "pre‐trained")
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported", [{"doc_id": doc.doc_id, "quote": curly, "stance": "entails"}]
        ),
        audit_stances=["entails"],
        documents=[doc],
    )
    result = checker.check("RAG combines parametric and non-parametric memory")

    assert result.verdict is Verdict.SUPPORTED


def test_valid_quote_survives_alongside_a_fabricated_one() -> None:
    """Partial fabrication degrades the answer; it does not poison it."""
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported",
            [
                {"doc_id": RAG_BENCH.doc_id, "quote": GROUNDED_QUOTE, "stance": "entails"},
                {
                    "doc_id": RAG_BENCH.doc_id,
                    "quote": "RAG was shown to remove all factual errors on medical QA",
                    "stance": "entails",
                },
            ],
        ),
        audit_stances=["entails"],
    )
    result = checker.check(CLAIM)

    assert result.verdict is Verdict.SUPPORTED
    assert len(result.evidence) == 1
    assert result.evidence[0].quote == GROUNDED_QUOTE
    assert result.stats.quotes_accepted == 1
    assert result.stats.quotes_rejected == 1


def test_quote_below_the_minimum_length_is_rejected_despite_being_present() -> None:
    """Exact-substring checking has a hole at the short end.

    'is a promising approach' is perfectly verbatim and proves nothing. Without a
    floor, citation validity rate would read 100% while grounding quietly failed.
    """
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported",
            [{"doc_id": RAG_BENCH.doc_id, "quote": "is a promising", "stance": "entails"}],
        )
    )
    result = checker.check(CLAIM)

    assert result.verdict is Verdict.UNKNOWN


def test_quote_above_the_maximum_length_is_rejected() -> None:
    """The mirror-image hole: an entire abstract is verbatim and pinpoints nothing."""
    cfg = Config().set("validator.max_quote_chars", 120)
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported",
            [{"doc_id": RAG_BENCH.doc_id, "quote": RAG_BENCH.abstract, "stance": "entails"}],
        ),
        config=cfg,
    )
    result = checker.check(CLAIM)

    assert result.verdict is Verdict.UNKNOWN


def test_accepted_evidence_carries_exact_offsets_into_the_source_document() -> None:
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported",
            [{"doc_id": RAG_BENCH.doc_id, "quote": GROUNDED_QUOTE, "stance": "entails"}],
        ),
        audit_stances=["entails"],
    )
    result = checker.check(CLAIM)

    item = result.evidence[0]
    assert RAG_BENCH.abstract[item.char_start : item.char_end] == GROUNDED_QUOTE


def test_offsets_are_exact_even_when_the_quote_arrived_reflowed() -> None:
    """Offsets must index the source, not the normalized comparison string."""
    reflowed = GROUNDED_QUOTE.replace(" ", "\n   ")
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported", [{"doc_id": RAG_BENCH.doc_id, "quote": reflowed, "stance": "entails"}]
        ),
        audit_stances=["entails"],
    )
    result = checker.check(CLAIM)

    item = result.evidence[0]
    assert RAG_BENCH.abstract[item.char_start : item.char_end] == GROUNDED_QUOTE


# --------------------------------------------------------------------------- #
# abstention is the default; answering is what must be earned
# --------------------------------------------------------------------------- #


def test_claim_on_a_topic_absent_from_the_corpus_returns_unknown() -> None:
    """Nothing retrieved means nothing to say -- not a nearest-neighbour verdict."""
    checker, verifier_llm, _ = build(
        verifier_response=verdict_response("unknown", []), documents=[]
    )
    result = checker.check("The Baltic dry index fell sharply in March 1997")

    assert result.verdict is Verdict.UNKNOWN
    assert result.retrieved == ()
    assert verifier_llm.calls == [], "no model call should be paid for on an empty retrieval"


def test_claim_the_corpus_is_merely_silent_about_is_unknown_never_contradicted() -> None:
    """Absence is not contradiction. Collapsing these is the standard naive failure."""
    checker, _, _ = build(
        verifier_response=verdict_response("unknown", []),
        documents=[RAG_BENCH, RAG_ORIGINAL],
    )
    result = checker.check("RAG reduces hallucination specifically on paediatric oncology QA")

    assert result.verdict is Verdict.UNKNOWN
    assert Flag.CONFLICTING_EVIDENCE not in result.flags


def test_verdict_with_no_evidence_at_all_cannot_be_supported() -> None:
    checker, _, _ = build(verifier_response=verdict_response("supported", []))
    result = checker.check(CLAIM)

    assert result.verdict is Verdict.UNKNOWN


class AlwaysSupported:
    """The policy this design must be safe against."""

    def decide(self, evidence: Sequence[Evidence]) -> Verdict:
        return Verdict.SUPPORTED


@pytest.mark.parametrize(
    "response",
    [
        verdict_response("supported", []),
        verdict_response(
            "supported",
            [{"doc_id": RAG_BENCH.doc_id, "quote": "wholly invented span", "stance": "entails"}],
        ),
    ],
    ids=["model-cited-nothing", "every-quote-fabricated"],
)
def test_no_policy_can_answer_on_zero_surviving_evidence(response: dict[str, Any]) -> None:
    """The short-circuit lives above the policy, not inside it.

    With the default policy this is unobservable -- ``contradiction_wins`` abstains on
    an empty list anyway -- so the guarantee has to be tested against a policy that
    *would* answer. That is the whole point of placing it above the port: the policy
    chooses among verdicts, it does not choose whether evidence was required.
    """
    cfg = Config()
    checker = FactChecker(
        retrieval=FixedRetrieval(CORPUS),
        verifier=LLMVerifier(ScriptedLLM([response]), cfg.query.verifier),
        auditor=NoopAuditor(),
        validator=SpanValidator(cfg.query.validator),
        policy=AlwaysSupported(),
        document_store=InMemoryDocumentStore(CORPUS),
    )
    result = checker.check(CLAIM)

    assert result.verdict is Verdict.UNKNOWN
    assert result.evidence == ()


def test_min_supporting_quotes_of_zero_still_cannot_manufacture_a_verdict() -> None:
    """The same hole reached through config rather than through a swapped policy."""
    cfg = Config().set("aggregation.min_supporting_quotes", 0)
    checker, _, _ = build(verifier_response=verdict_response("supported", []), config=cfg)

    assert checker.check(CLAIM).verdict is Verdict.UNKNOWN


# --------------------------------------------------------------------------- #
# the second pass catches what the first cannot
# --------------------------------------------------------------------------- #


def test_verbatim_but_irrelevant_quote_is_caught_by_the_audit() -> None:
    """Span validation cannot see relevance; that is exactly the auditor's job."""
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported",
            [
                {
                    "doc_id": RAG_ORIGINAL.doc_id,
                    "quote": "We set the state of the art on three open domain question answering",
                    "stance": "entails",
                }
            ],
        ),
        audit_stances=["neither"],
        documents=[RAG_ORIGINAL],
    )
    result = checker.check("RAG eliminates hallucination in clinical summarisation")

    assert result.verdict is Verdict.UNKNOWN


def test_auditor_sees_neither_the_retrieval_context_nor_the_first_pass_reasoning() -> None:
    """An auditor shown the first pass's chain largely ratifies it. The isolation is
    the entire value of the stage, so it is asserted rather than assumed."""
    checker, _, audit_llm = build(
        verifier_response=verdict_response(
            "supported",
            [{"doc_id": RAG_BENCH.doc_id, "quote": GROUNDED_QUOTE, "stance": "entails"}],
            reasoning="The abstract opens by calling RAG promising for hallucination.",
        ),
        audit_stances=["entails"],
    )
    checker.check(CLAIM)

    prompt = audit_llm.calls[0]["system"] + audit_llm.calls[0]["user"]
    assert GROUNDED_QUOTE in prompt, "the auditor must see the quote it is judging"
    assert CLAIM in prompt, "and the claim"
    assert "The abstract opens by calling RAG" not in prompt
    assert LOST_V1.abstract not in prompt
    assert RAG_ORIGINAL.abstract not in prompt


def test_audit_can_overturn_the_first_pass_to_contradicted() -> None:
    quote = "performance degrades significantly when models must access relevant information"
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported", [{"doc_id": LOST_V1.doc_id, "quote": quote, "stance": "entails"}]
        ),
        audit_stances=["contradicts"],
        documents=[LOST_V1],
    )
    result = checker.check("Language models handle mid-context information reliably")

    assert result.verdict is Verdict.CONTRADICTED


def test_disabling_the_second_pass_is_a_working_pipeline() -> None:
    """An ablation path the evaluation depends on, so it must actually run."""
    cfg = Config().set("auditor.enabled", False)
    checker, _, audit_llm = build(
        verifier_response=verdict_response(
            "supported",
            [{"doc_id": RAG_BENCH.doc_id, "quote": GROUNDED_QUOTE, "stance": "entails"}],
        ),
        config=cfg,
    )
    result = checker.check(CLAIM)

    assert result.verdict is Verdict.SUPPORTED
    assert audit_llm.calls == []


# --------------------------------------------------------------------------- #
# conflicts are surfaced, not resolved
# --------------------------------------------------------------------------- #


def test_opposing_stances_across_two_versions_produce_contradicted_plus_the_flag() -> None:
    """A revision that changes a finding is a real disagreement in the corpus.

    Deduplicating hides it; letting contradiction silently win misreports a
    superseded finding. Both sides are returned, with version identifiers.
    """
    v1_quote = "performance degrades significantly when models must access relevant information"
    v3_quote = "performance is stable when models must access relevant information"
    checker, _, _ = build(
        verifier_response=verdict_response(
            "contradicted",
            [
                {"doc_id": LOST_V1.doc_id, "quote": v1_quote, "stance": "entails"},
                {"doc_id": LOST_V3.doc_id, "quote": v3_quote, "stance": "contradicts"},
            ],
        ),
        audit_stances=["entails", "contradicts"],
        documents=[LOST_V1, LOST_V3],
    )
    result = checker.check("Language models lose information in the middle of long contexts")

    assert result.verdict is Verdict.CONTRADICTED
    assert Flag.CONFLICTING_EVIDENCE in result.flags
    cited = {e.doc_id for e in result.evidence}
    assert cited == {LOST_V1.doc_id, LOST_V3.doc_id}, "both sides must be shown"
    assert {e.stance for e in result.evidence} == {Stance.ENTAILS, Stance.CONTRADICTS}


def test_agreement_across_versions_raises_no_conflict_flag() -> None:
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported",
            [{"doc_id": RAG_BENCH.doc_id, "quote": GROUNDED_QUOTE, "stance": "entails"}],
        ),
        audit_stances=["entails"],
    )
    result = checker.check(CLAIM)

    assert result.flags == () or Flag.CONFLICTING_EVIDENCE not in result.flags


# --------------------------------------------------------------------------- #
# the result carries what evaluation needs
# --------------------------------------------------------------------------- #


def test_retrieved_documents_are_reported_so_misses_are_attributable() -> None:
    """Without this field a retriever miss and a verifier misread are the same event."""
    checker, _, _ = build(
        verifier_response=verdict_response("unknown", []), documents=[RAG_BENCH, RAG_ORIGINAL]
    )
    result = checker.check(CLAIM)

    assert result.retrieved == (RAG_BENCH.doc_id, RAG_ORIGINAL.doc_id)


def test_counters_report_retrieval_quotes_and_model_calls() -> None:
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported",
            [
                {"doc_id": RAG_BENCH.doc_id, "quote": GROUNDED_QUOTE, "stance": "entails"},
                {
                    "doc_id": RAG_BENCH.doc_id,
                    "quote": "invented text not in any abstract here",
                    "stance": "entails",
                },
            ],
        ),
        audit_stances=["entails"],
    )
    result = checker.check(CLAIM)

    assert result.stats.passages_retrieved == len(CORPUS)
    assert result.stats.quotes_proposed == 2
    assert result.stats.quotes_accepted == 1
    assert result.stats.quotes_rejected == 1
    assert result.stats.llm_calls == 2, "one verifier call plus one audit of the survivor"


def test_trace_explains_why_each_rejected_span_failed() -> None:
    checker, _, _ = build(
        verifier_response=verdict_response(
            "supported",
            [
                {"doc_id": RAG_BENCH.doc_id, "quote": "is a promising", "stance": "entails"},
                {"doc_id": "9999.99999v1", "quote": GROUNDED_QUOTE, "stance": "entails"},
            ],
        )
    )
    result = checker.check(CLAIM)

    assert result.trace is not None
    reasons = {r.reason for r in result.trace.rejected}
    assert len(result.trace.rejected) == 2
    assert reasons == {"too_short", "not_retrieved"}


def test_verifier_is_shown_only_retrieved_passages() -> None:
    """Its own training knowledge must not be able to leak into a verdict."""
    checker, verifier_llm, _ = build(
        verifier_response=verdict_response("unknown", []), documents=[RAG_BENCH]
    )
    checker.check(CLAIM)

    prompt = verifier_llm.calls[0]["user"]
    assert RAG_BENCH.abstract in prompt
    assert LOST_V1.abstract not in prompt
    assert RAG_ORIGINAL.abstract not in prompt
