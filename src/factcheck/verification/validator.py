"""Span validation: the guarantee that survives prompt drift.

This is deliberately not a port and its core requirement is deliberately not
configurable. A swappable guarantee is not a guarantee -- the first convenient
substitution is a permissive validator, and groundedness leaves with it.

The knobs here bound *evidentiary weight*, not presence. Normalization controls how
text is compared; nothing controls whether the quote must be there. There is no
fuzzy-match threshold, because that is the knob that converts the grounding
guarantee back into a tuning parameter. Paraphrase drift is the auditor's job.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..config import ValidatorConfig
from ..types import Document, Evidence, RawEvidence, RejectedSpan

# Rejection reasons, as constants because the rate of each is tracked over time:
# a rise in NOT_PRESENT is prompt drift, a rise in TOO_SHORT is a model learning to
# satisfy the gate with fragments.
NOT_RETRIEVED = "not_retrieved"
NOT_PRESENT = "not_present"
TOO_SHORT = "too_short"
TOO_LONG = "too_long"
EMPTY = "empty"

# Typographic substitutions a model makes when reflowing source text. Every entry is
# a 1:1 character replacement, so applying them cannot shift an offset.
_PUNCTUATION = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"', "´": "'", "`": "'",
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ",
}


@dataclass(frozen=True)
class ValidationOutcome:
    accepted: tuple[Evidence, ...]
    rejected: tuple[RejectedSpan, ...]


class SpanValidator:
    """Checks each quote is literally present in the document it cites."""

    def __init__(self, cfg: ValidatorConfig) -> None:
        self._cfg = cfg

    def validate(
        self,
        evidence: Sequence[RawEvidence],
        documents: Mapping[str, Document],
        retrieved: Sequence[str],
    ) -> ValidationOutcome:
        accepted: list[Evidence] = []
        rejected: list[RejectedSpan] = []
        allowed = set(retrieved)

        for item in evidence:
            reason = None
            quote = item.quote.strip()
            if not quote:
                reason = EMPTY
            elif item.doc_id not in allowed or item.doc_id not in documents:
                # Covers both a hallucinated identifier and a real document the
                # verifier was never shown. Either way the citation is unearned.
                reason = NOT_RETRIEVED

            if reason is not None:
                rejected.append(RejectedSpan(item.doc_id, item.quote, reason))
                continue

            located = self._locate(quote, documents[item.doc_id].abstract)
            if located is None:
                rejected.append(RejectedSpan(item.doc_id, item.quote, NOT_PRESENT))
                continue

            start, end, length = located
            if length < self._cfg.min_quote_chars:
                rejected.append(RejectedSpan(item.doc_id, item.quote, TOO_SHORT))
            elif length > self._cfg.max_quote_chars:
                rejected.append(RejectedSpan(item.doc_id, item.quote, TOO_LONG))
            else:
                # The stored quote is the *source* text, not the model's rendering.
                # What we return is then provably corpus text rather than something
                # that merely matched corpus text after normalization.
                accepted.append(
                    Evidence(
                        doc_id=item.doc_id,
                        quote=documents[item.doc_id].abstract[start:end],
                        stance=item.stance,
                        char_start=start,
                        char_end=end,
                    )
                )

        return ValidationOutcome(tuple(accepted), tuple(rejected))

    def _locate(self, quote: str, document: str) -> tuple[int, int, int] | None:
        """Find the quote in the document, returning offsets into the *original* text.

        Comparison happens on a normalized copy; the offsets returned index the
        source, so evidence can be located exactly in the document a reader opens.
        """
        norm_doc, index_map = self._normalize(document)
        norm_quote, _ = self._normalize(quote)
        norm_quote = norm_quote.strip()
        if not norm_quote:
            return None

        at = norm_doc.find(norm_quote)
        if at < 0:
            return None
        start = index_map[at]
        end = index_map[at + len(norm_quote) - 1] + 1
        return start, end, len(norm_quote)

    def _normalize(self, text: str) -> tuple[str, list[int]]:
        """Normalize while recording, for each output character, its source index."""
        cfg = self._cfg
        out: list[str] = []
        index_map: list[int] = []
        previous_was_space = False

        for i, raw in enumerate(text):
            char = _PUNCTUATION.get(raw, raw) if cfg.normalize_unicode_quotes else raw
            if cfg.normalize_whitespace and char.isspace():
                if previous_was_space:
                    continue
                char = " "
                previous_was_space = True
            else:
                previous_was_space = False
            if cfg.case_insensitive:
                lowered = char.lower()
                # Guard the handful of characters whose lowercase form is longer:
                # substituting one would desynchronize the offset map.
                char = lowered if len(lowered) == 1 else char
            out.append(char)
            index_map.append(i)

        return "".join(out), index_map
