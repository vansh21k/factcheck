"""Pass 2: re-check one surviving quote against the claim, in isolation.

Span validation cannot see relevance -- a quote can be perfectly verbatim and still
have nothing to do with the claim. That is the hole this stage covers.

The isolation is the entire value: the auditor is given the claim and one quote, and
neither the retrieved context nor the first pass's reasoning. An auditor shown the
first pass's chain largely ratifies it, which buys nothing at twice the cost.
"""

from __future__ import annotations

from typing import Any

from ..config import AuditorConfig
from ..ports import LLMClient
from ..types import RawEvidence, Stance
from .prompts import AUDITOR_TOOL, auditor_prompt


class LLMAuditor:
    """Runs on a smaller model: this is a narrow entailment task, not adjudication."""

    def __init__(self, llm: LLMClient, cfg: AuditorConfig) -> None:
        self._llm = llm
        self._cfg = cfg

    def audit(self, claim: str, evidence: RawEvidence) -> Stance:
        raw: dict[str, Any] = self._llm.complete(
            system=auditor_prompt(self._cfg.prompt_version),
            user=f"CLAIM\n\n{claim}\n\nQUOTED PASSAGE\n\n{evidence.quote}",
            model=self._cfg.model,
            temperature=self._cfg.temperature,
            tool=AUDITOR_TOOL,
            max_tokens=256,
        )
        try:
            return Stance(raw.get("stance"))
        except ValueError:
            # An unreadable stance must not manufacture support. The auditor is only
            # ever allowed to remove it.
            return Stance.NEITHER


class NoopAuditor:
    """The ablation path for ``auditor.enabled = false``.

    Keeps the first pass's stance, so disabling the second pass measures its
    contribution rather than changing the shape of the pipeline around it.
    """

    def audit(self, claim: str, evidence: RawEvidence) -> Stance:
        return evidence.stance
