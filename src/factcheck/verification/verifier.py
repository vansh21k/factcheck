"""Pass 1: adjudicate the claim against the retrieved passages, and nothing else."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..config import VerifierConfig
from ..ports import LLMClient
from ..types import Chunk, RawEvidence, RawVerdict, Stance, Verdict
from .prompts import VERIFIER_TOOL, verifier_prompt


class LLMVerifier:
    """A single tool-use call over the passages the retriever selected.

    The schema, not the prompt, is what makes an uncited verdict impossible to
    express. Parsing here is deliberately forgiving: a malformed field degrades to a
    dropped evidence item or an 'unknown' verdict, never an exception, because a
    model failure should cost recall rather than crash an interactive session.
    """

    def __init__(self, llm: LLMClient, cfg: VerifierConfig) -> None:
        self._llm = llm
        self._cfg = cfg

    def adjudicate(self, claim: str, passages: Sequence[Chunk]) -> RawVerdict:
        raw = self._llm.complete(
            system=verifier_prompt(self._cfg.prompt_version),
            user=_render(claim, passages),
            model=self._cfg.model,
            temperature=self._cfg.temperature,
            tool=VERIFIER_TOOL,
            max_tokens=self._cfg.max_tokens,
        )
        return _parse(raw)


def _render(claim: str, passages: Sequence[Chunk]) -> str:
    blocks = "\n\n".join(f"[{p.doc_id}]\n{p.text}" for p in passages)
    return f"PASSAGES\n\n{blocks}\n\nCLAIM\n\n{claim}"


def _parse(raw: dict[str, Any]) -> RawVerdict:
    evidence: list[RawEvidence] = []
    for item in raw.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        doc_id, quote = item.get("doc_id"), item.get("quote")
        if not isinstance(doc_id, str) or not isinstance(quote, str):
            continue
        evidence.append(RawEvidence(doc_id, quote, _stance(item.get("stance"))))

    return RawVerdict(
        verdict=_verdict(raw.get("verdict")),
        evidence=tuple(evidence),
        reasoning=str(raw.get("reasoning") or ""),
    )


def _stance(value: Any) -> Stance:
    try:
        return Stance(value)
    except ValueError:
        return Stance.NEITHER


def _verdict(value: Any) -> Verdict:
    try:
        return Verdict(value)
    except ValueError:
        return Verdict.UNKNOWN
