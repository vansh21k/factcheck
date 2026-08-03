"""Rendering a verdict so it can be judged without opening another tool."""

from __future__ import annotations

import json
from typing import Any

from ..ingest.store import JsonlDocumentStore
from ..types import Trace, Verdict, VerificationResult

_RULE = "─" * 65

_VERDICT_LABEL = {
    Verdict.SUPPORTED: "SUPPORTED",
    Verdict.CONTRADICTED: "CONTRADICTED",
    Verdict.UNKNOWN: "UNKNOWN",
}


def render_result(result: VerificationResult, store: JsonlDocumentStore) -> str:
    """Verdict, evidence, source titles and dates, then the per-claim counters."""
    lines = ["", f"  {_VERDICT_LABEL[result.verdict]}"]
    if result.flags:
        lines.append(f"  flags: {', '.join(f.value for f in result.flags)}")
    lines.append(f"  {_RULE}")

    if not result.evidence:
        lines.append("  no surviving evidence")
    for item in result.evidence:
        lines.append(f"  [{item.doc_id}] {item.stance.value}")
        for wrapped in _wrap(item.quote, width=58):
            lines.append(f'    "{wrapped}"')
        source = _source_line(item.doc_id, store)
        if source:
            lines.append(f"    {source}")
        lines.append("")

    stats = result.stats
    lines.append(
        f"  retrieved {stats.passages_retrieved} · quotes {stats.quotes_accepted} · "
        f"span-rejected {stats.quotes_rejected} · {stats.llm_calls} LLM calls · "
        f"{stats.elapsed_s:.1f}s"
    )
    return "\n".join(lines)


def render_why(result: VerificationResult) -> str:
    """Attribute a bad answer to a stage: what was retrieved, how it ranked, what
    each quote's audit said, and why every rejected span failed."""
    trace: Trace | None = result.trace
    if trace is None:
        return "  no trace recorded for the last claim"

    lines = ["", f"  claim: {result.claim}", f"  {_RULE}"]

    lines.append("  queries")
    for query in trace.retrieval.queries:
        lines.append(f"    [{query.kind}] {query.text}")

    if trace.retrieval.per_query_ranks:
        lines.append("  per-query ranks")
        for text, chunk_ids in trace.retrieval.per_query_ranks:
            lines.append(f"    {text[:44]!r}")
            lines.append(f"      {' > '.join(chunk_ids[:6]) or '(nothing)'}")

    lines.append("  fused ranks")
    lines.append(f"    {' > '.join(trace.retrieval.fused_ranks[:8]) or '(nothing)'}")
    lines.append(f"  passages shown to the verifier: {len(trace.passages)}")

    if trace.raw_verdict is not None:
        lines.append(f"  pass-1 verdict: {trace.raw_verdict.verdict.value}")

    lines.append("  quote audit")
    if not trace.audit_stances:
        lines.append("    (none survived span validation)")
    for quote, stance in trace.audit_stances:
        lines.append(f"    {stance.value:<11} {quote[:48]!r}")

    lines.append("  rejected spans")
    if not trace.rejected:
        lines.append("    (none)")
    for span in trace.rejected:
        lines.append(f"    {span.reason:<15} [{span.doc_id}] {span.quote[:40]!r}")

    return "\n".join(lines)


def result_to_dict(result: VerificationResult) -> dict[str, Any]:
    """Machine-readable output, so results can be post-processed without scraping."""
    return {
        "claim": result.claim,
        "verdict": result.verdict.value,
        "flags": [flag.value for flag in result.flags],
        "retrieved": list(result.retrieved),
        "evidence": [
            {
                "doc_id": item.doc_id,
                "quote": item.quote,
                "stance": item.stance.value,
                "char_start": item.char_start,
                "char_end": item.char_end,
            }
            for item in result.evidence
        ],
        "stats": {
            "passages_retrieved": result.stats.passages_retrieved,
            "quotes_proposed": result.stats.quotes_proposed,
            "quotes_accepted": result.stats.quotes_accepted,
            "quotes_rejected": result.stats.quotes_rejected,
            "llm_calls": result.stats.llm_calls,
            "elapsed_s": result.stats.elapsed_s,
        },
    }


def result_to_json(result: VerificationResult) -> str:
    return json.dumps(result_to_dict(result), ensure_ascii=False)


def _source_line(doc_id: str, store: JsonlDocumentStore) -> str:
    try:
        doc = store.get(doc_id)
    except KeyError:
        return ""
    return f"{doc.title} · {doc.published}"


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
