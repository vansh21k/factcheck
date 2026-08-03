"""Every pipeline stage, as a structural-typing protocol.

Protocols rather than base classes so a test double is any object with the right
method and never a subclass. ``FactChecker`` depends on these names and never on a
concrete class.

Two things are deliberately *not* here. ``SpanValidator`` is the grounding guarantee,
and a swappable guarantee is not one -- the first convenient substitution is a
permissive validator. The ``no surviving evidence -> unknown`` short-circuit lives in
``FactChecker``, above ``AggregationPolicy``, so no policy can be written that answers
on zero evidence. The policy chooses among verdicts; it does not choose whether
evidence was required.

``IndexBuilder`` and ``IndexLoader`` are separate on purpose. A single fat interface
carrying both would let query-time code call ``build``, which is the accidental
startup rebuild the three-program split exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .types import (
    Chunk,
    Document,
    Evidence,
    Query,
    RawEvidence,
    RawVerdict,
    ScoredChunk,
    Stance,
    Verdict,
)

# --------------------------------------------------------------------------- #
# retrieval side
# --------------------------------------------------------------------------- #


@runtime_checkable
class QueryExpander(Protocol):
    def expand(self, claim: str) -> list[Query]:
        """Claim -> retrieval queries, including a negation form.

        Without the negation query, refuting passages are rarely retrieved and the
        verifier is structurally incapable of returning *contradicted*. Must always
        return at least one query.
        """


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: Query, k: int) -> list[ScoredChunk]:
        """Return at most ``k`` chunks, best first."""


@runtime_checkable
class Fusion(Protocol):
    def fuse(self, ranked_lists: Sequence[Sequence[ScoredChunk]], k: int) -> list[ScoredChunk]:
        """Combine ranked lists into one, best first, at most ``k`` long."""


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, claim: str, chunks: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        """Reorder candidates for precision and cap the number reaching the verifier."""


# --------------------------------------------------------------------------- #
# verification side
# --------------------------------------------------------------------------- #


@runtime_checkable
class Verifier(Protocol):
    def adjudicate(self, claim: str, passages: Sequence[Chunk]) -> RawVerdict:
        """Judge the claim against ``passages`` only, returning cited spans."""


@runtime_checkable
class Auditor(Protocol):
    def audit(self, claim: str, evidence: RawEvidence) -> Stance:
        """Re-check one quote against the claim in isolation.

        Implementations must not be given the retrieved context or the first pass's
        reasoning: an auditor shown the first pass's chain largely ratifies it, and
        that independence is the entire value of the stage.
        """


@runtime_checkable
class AggregationPolicy(Protocol):
    def decide(self, evidence: Sequence[Evidence]) -> Verdict:
        """Choose a verdict from collected stances. Never called with empty evidence."""


# --------------------------------------------------------------------------- #
# corpus side
# --------------------------------------------------------------------------- #


@runtime_checkable
class DocumentStore(Protocol):
    def get(self, doc_id: str) -> Document:
        """Return the document, or raise ``KeyError`` if it is not in the corpus."""

    def iter_chunks(self) -> Iterable[Chunk]:
        """Yield every chunk in a stable order."""


# --------------------------------------------------------------------------- #
# the offline / online boundary
# --------------------------------------------------------------------------- #


@runtime_checkable
class IndexBuilder(Protocol):
    """Offline, batch, allowed to be slow."""

    name: str

    def build(self, chunks: Sequence[Chunk], out_dir: Path) -> dict[str, Any]:
        """Write index artifacts and return the manifest entry describing them."""


@runtime_checkable
class IndexLoader(Protocol):
    """Online, per keystroke, not allowed to be slow."""

    name: str

    def load(self, out_dir: Path, manifest: dict[str, Any]) -> Retriever:
        """Load prebuilt artifacts into a queryable retriever."""


# --------------------------------------------------------------------------- #
# model access
# --------------------------------------------------------------------------- #


@runtime_checkable
class LLMClient(Protocol):
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
        """Return the model's structured output.

        With ``tool`` supplied the return value is that tool's input object; without
        it, ``{"text": ...}``. Keeping the port this narrow is what lets tests
        substitute a scripted model with no network call and no API key.
        """
