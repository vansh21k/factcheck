"""Turn a claim into the set of queries retrieval actually runs.

The two implementations here are not "real" vs "stub" -- both are load-bearing.
``IdentityExpander`` is the ablation path exercised whenever ``expander.enabled`` is
false, so it has to behave correctly on its own, not just compile.
"""

from __future__ import annotations

from typing import Any

from ..config import ExpanderConfig
from ..ports import LLMClient
from ..types import Query

_EXPAND_TOOL: dict[str, Any] = {
    "name": "expand_claim",
    "description": (
        "Expand a factual claim into supporting retrieval queries: a negation of "
        "the claim (for finding refuting evidence) and, for multi-part claims, "
        "paraphrases or sub-claims that isolate each part."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "negation": {
                "type": "string",
                "description": (
                    "The claim rewritten to assert the opposite. Must be a real "
                    "sentence, not a mere negation marker -- 'X does not improve Y' "
                    "rather than 'not: X improves Y'."
                ),
            },
            "subclaims": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Paraphrases or sub-claims of the original, one per distinct "
                    "factual assertion the claim makes. Empty if the claim is "
                    "already a single atomic assertion."
                ),
            },
        },
        "required": ["negation", "subclaims"],
    },
}

_SYSTEM_PROMPT = (
    "You expand a factual claim into retrieval queries for a semantic search "
    "index over research paper abstracts. Call the expand_claim tool exactly "
    "once with your expansion."
)


class IdentityExpander:
    """The no-op expander: the claim, verbatim, as the only query.

    This is what runs when ``expander.enabled`` is false. It is a real
    implementation and not a placeholder, because the evaluation harness needs a
    working, cost-free baseline to measure the expander's contribution against --
    an expander that merely raises `NotImplementedError` cannot serve as that
    baseline.
    """

    def expand(self, claim: str) -> list[Query]:
        return [Query(claim, kind="claim")]


class NegationAwareExpander:
    """Expands a claim into claim + negation + sub-claim queries via one LLM call.

    The negation query is a correctness requirement, not a recall optimization.
    Retrieval ranks passages by similarity to the query, and a passage that looks
    like the claim is exactly the passage that *confirms* it -- semantic search has
    no way to distinguish "similar and true" from "similar and false" without being
    asked the negated question directly. Without a negation-form query in the mix,
    refuting passages are rarely retrieved at all, and the verifier is then
    structurally incapable of ever returning ``contradicted``: it cannot cite
    evidence it never saw. This is why ``contradicted`` recall is the metric that
    would catch a regression here, not overall accuracy.
    """

    def __init__(self, llm: LLMClient, cfg: ExpanderConfig) -> None:
        self._llm = llm
        self._cfg = cfg

    def expand(self, claim: str) -> list[Query]:
        queries = [Query(claim, kind="claim")]

        try:
            result = self._llm.complete(
                system=_SYSTEM_PROMPT,
                user=claim,
                model=self._cfg.model,
                temperature=self._cfg.temperature,
                tool=_EXPAND_TOOL,
                max_tokens=512,
            )
            negation, subclaims = _parse_expansion(result)
        except Exception:
            # A claim must always produce at least one query. A flaky or
            # malformed-output model call degrades to the identity expansion
            # rather than failing the claim outright.
            return _dedupe(queries)

        if self._cfg.include_negation and negation:
            queries.append(Query(negation, kind="negation"))

        remaining = max(self._cfg.n_queries - len(queries), 0)
        for text in subclaims[:remaining]:
            queries.append(Query(text, kind="subclaim"))

        return _dedupe(queries)


def _parse_expansion(result: Any) -> tuple[str, list[str]]:
    """Validate the tool output shape, raising on anything unusable.

    Raising (rather than returning a partial result) lets ``expand`` handle both
    "the call failed" and "the call succeeded with garbage" through the same
    fallback path.
    """
    if not isinstance(result, dict):
        raise ValueError("expansion result must be a dict")
    negation = result.get("negation", "")
    if not isinstance(negation, str):
        raise ValueError("negation must be a string")
    subclaims_raw = result.get("subclaims", [])
    if not isinstance(subclaims_raw, list):
        raise ValueError("subclaims must be a list")
    subclaims = [s.strip() for s in subclaims_raw if isinstance(s, str) and s.strip()]
    return negation.strip(), subclaims


def _dedupe(queries: list[Query]) -> list[Query]:
    """Case-insensitive de-dup, preserving order and the first-seen ``kind``."""
    seen: set[str] = set()
    out: list[Query] = []
    for q in queries:
        key = q.text.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out
