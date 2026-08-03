"""Corpus ingest: the only package in this system with network access.

``fetch`` (see ``factcheck.cli.fetch``) is built entirely on ``ArxivClient`` and
``JsonlDocumentStore``. Everything past this package -- indexing, retrieval,
verification -- reads ``documents.jsonl`` from disk and never touches the network,
which is the offline/online split the whole project is organized around.
"""

from __future__ import annotations

from ..corpus import corpus_hash
from .arxiv import ArxivClient, FetchOutcome, find_unresolved
from .store import JsonlDocumentStore

__all__ = [
    "ArxivClient",
    "FetchOutcome",
    "JsonlDocumentStore",
    "corpus_hash",
    "find_unresolved",
]
