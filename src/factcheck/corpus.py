"""The corpus content hash, defined once.

This lives in its own leaf module rather than in ``ingest`` or ``indexing`` because
both sides need it and neither may depend on the other: ``index`` must not import
``ingest`` (an index build works from an in-memory ``Sequence[Document]``), and
``verify`` must not import build-time code.

It has to be exactly one function. The hash recorded in the manifest at build time
and the hash recomputed from ``documents.jsonl`` at query time are compared to decide
whether an index is stale -- two implementations that disagree turn that check into
either a permanent false warning or a silent no-op, and both failure modes look like
the system working.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .types import Document


def corpus_hash(documents: Iterable[Document]) -> str:
    """Order-independent content hash over every field of every document.

    Sorted before hashing so the hash reflects corpus *content*, not fetch or
    iteration order: two fetches of the same 50 IDs in a different order must hash
    identically, or every rebuild would look stale for no real reason.

    Every field is included, not just the abstract. A corrected title or a new
    ``updated`` date does not change retrieval, but it does change what a verdict
    displays as its source, and "the index is older than the documents" is the thing
    being detected.
    """
    payload = json.dumps(
        [document_to_dict(doc) for doc in sorted(documents, key=lambda d: d.doc_id)],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def document_to_dict(doc: Document) -> dict[str, object]:
    """The canonical serialized form, shared by the hash and by ``documents.jsonl``.

    One function so a field can never be persisted without also being hashed.
    """
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "authors": list(doc.authors),
        "published": doc.published,
        "updated": doc.updated,
        "abstract": doc.abstract,
    }
