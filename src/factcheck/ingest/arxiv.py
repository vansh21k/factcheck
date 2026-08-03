"""``ArxivClient``: the only component in this system permitted network access.

Everything downstream reads ``documents.jsonl`` (see ``store.py``), which is the
whole point of splitting ``fetch`` from ``index`` -- re-chunking or swapping the
embedding model should never touch the network again.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import urlopen

from ..types import Document

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"

# Verified against the live API during design (see HANDOFF.md / spec "Further
# Notes"): the plain http:// endpoint answers 301 with an EMPTY body, so a client
# that doesn't follow redirects onto https gets zero bytes back, parses zero
# entries, and every claim downstream quietly returns "unknown" -- a failure that
# reads as appropriate conservatism rather than a broken corpus. Requesting https
# directly sidesteps the redirect entirely; urllib still follows any further
# redirect transparently for a GET, so this is strictly safer, never more fragile.
DEFAULT_BASE_URL = "https://export.arxiv.org/api/query"


def _default_fetch_url(url: str) -> bytes:
    # Stdlib only, deliberately -- no requests/httpx dependency for the one HTTP call
    # this project makes.
    with urlopen(url, timeout=30) as response:
        return bytes(response.read())


@dataclass
class FetchOutcome:
    """What one ``fetch`` call produced: documents, and requested IDs that didn't resolve.

    Kept as two lists rather than one combined structure so the caller is forced to
    look at ``unresolved`` explicitly -- a silently short corpus is exactly the bug
    this type exists to make visible.
    """

    documents: list[Document]
    unresolved: list[str]


class ArxivClient:
    """Fetches arXiv documents by ID, in as few requests as the service allows.

    The HTTP call and the sleep between batches are both constructor-injected seams
    (``fetch_url``, ``sleep``) so tests exercise real batching, parsing, and matching
    logic with zero network I/O and zero wall-clock delay.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        batch_size: int = 50,
        rate_limit_s: float = 3.0,
        fetch_url: Callable[[str], bytes] = _default_fetch_url,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url
        self._batch_size = batch_size
        self._rate_limit_s = rate_limit_s
        self._fetch_url = fetch_url
        self._sleep = sleep

    def fetch(
        self,
        ids: Sequence[str],
        *,
        on_batch: Callable[[list[Document]], None] | None = None,
    ) -> FetchOutcome:
        """Fetch every ID, batched and rate-limited, and report what didn't resolve.

        ``on_batch`` fires after each network round trip with just that batch's
        documents, before the next request (and before the rate-limit sleep for it).
        It exists so a caller (the ``fetch`` CLI) can persist progress incrementally
        -- a network failure on batch 4 of 5 must not discard batches 1-3.
        """
        documents: list[Document] = []
        for batch_index, batch in enumerate(_chunked(ids, self._batch_size)):
            if batch_index > 0:
                self._sleep(self._rate_limit_s)
            body = self._fetch_url(self._build_url(batch))
            batch_documents = _parse_feed(body)
            documents.extend(batch_documents)
            if on_batch is not None:
                on_batch(batch_documents)
        return FetchOutcome(documents=documents, unresolved=find_unresolved(ids, documents))

    def _build_url(self, batch: Sequence[str]) -> str:
        params = {"id_list": ",".join(batch), "max_results": str(self._batch_size)}
        return f"{self._base_url}?{urlencode(params, safe=',')}"


def find_unresolved(requested_ids: Sequence[str], documents: Iterable[Document]) -> list[str]:
    """Requested IDs with no matching resolved document.

    Matching is case-insensitive (hand-edited ID lists, arXiv IDs are not
    case-sensitive in practice) and tolerates the version-suffix asymmetry the API
    itself has: a version-suffixed request only matches that exact ``doc_id``,
    while an unsuffixed request matches any resolved document sharing its base ID,
    because the server resolves an unsuffixed ID to whatever is currently latest.
    """
    resolved_doc_ids = {doc.doc_id.lower() for doc in documents}
    resolved_bases = {doc.base_id.lower() for doc in documents}
    unresolved = []
    for requested in requested_ids:
        key = requested.strip().lower()
        is_versioned = _base_id(key) != key
        resolved = key in resolved_doc_ids if is_versioned else key in resolved_bases
        if not resolved:
            unresolved.append(requested)
    return unresolved


def _base_id(doc_id: str) -> str:
    """Mirrors ``Document.base_id`` for plain ID strings that aren't Documents yet."""
    head, sep, tail = doc_id.rpartition("v")
    return head if sep and tail.isdigit() else doc_id


def _chunked(items: Sequence[str], size: int) -> list[Sequence[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _parse_feed(body: bytes) -> list[Document]:
    root = ET.fromstring(body)
    ns = {"atom": ATOM_NS}
    documents = []
    for entry in root.findall("atom:entry", ns):
        # The entry <id> is a URL like http://arxiv.org/abs/2307.03172v3 -- the
        # trailing path segment IS the doc_id, version suffix included. This is the
        # only place a version can be dropped by accident, so it gets no processing
        # beyond the split.
        id_url = _text(entry.find("atom:id", ns))
        doc_id = id_url.rsplit("/", 1)[-1]
        authors = tuple(
            _text(author.find("atom:name", ns)) for author in entry.findall("atom:author", ns)
        )
        documents.append(
            Document(
                doc_id=doc_id,
                title=_normalize_whitespace(_text(entry.find("atom:title", ns))),
                authors=authors,
                published=_text(entry.find("atom:published", ns)),
                updated=_text(entry.find("atom:updated", ns)),
                abstract=_normalize_whitespace(_text(entry.find("atom:summary", ns))),
            )
        )
    return documents


def _text(element: ET.Element | None) -> str:
    return (element.text or "").strip() if element is not None else ""


def _normalize_whitespace(text: str) -> str:
    """Collapse arXiv's line-wrapped, indented title/summary text to single spaces.

    ``str.split()`` with no argument splits on any whitespace run (including the
    newlines and leading spaces the Atom feed wraps abstracts with) and drops empty
    strings, so this both de-wraps and strips in one pass. It does not otherwise
    touch the text -- span validation later needs quotes to be literal substrings
    of exactly this stored string, so nothing beyond whitespace may be altered.
    """
    return " ".join(text.split())
