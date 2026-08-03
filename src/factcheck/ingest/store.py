"""``JsonlDocumentStore``: the on-disk form of a fetched corpus.

Structurally satisfies the ``DocumentStore`` port (``get``, ``iter_chunks``) with no
inheritance from it -- that's the whole point of ports being protocols. Chunking is
the indexing step's job, not this one's, but the store reads ``chunks.jsonl`` when
present so retrieval-side code has a single place to ask for either documents or
chunks without caring which stage produced them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from ..corpus import document_to_dict
from ..types import Chunk, Document

DOCUMENTS_FILENAME = "documents.jsonl"
CHUNKS_FILENAME = "chunks.jsonl"


class JsonlDocumentStore:
    def __init__(self, documents: dict[str, Document], directory: Path) -> None:
        self._documents = documents
        self._directory = directory

    def get(self, doc_id: str) -> Document:
        return self._documents[doc_id]

    def iter_chunks(self) -> Iterator[Chunk]:
        """Yield chunks written by indexing, or nothing if it hasn't run yet.

        Tolerant by design: a store constructed right after ``fetch`` and before
        ``index`` is a perfectly normal state, not an error.
        """
        chunks_path = self._directory / CHUNKS_FILENAME
        if not chunks_path.exists():
            return
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                yield Chunk(
                    chunk_id=raw["chunk_id"],
                    doc_id=raw["doc_id"],
                    text=raw["text"],
                    char_start=raw["char_start"],
                    char_end=raw["char_end"],
                )

    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents.values())

    def __len__(self) -> int:
        return len(self._documents)

    @classmethod
    def load(cls, directory: str | Path) -> JsonlDocumentStore:
        """Load whatever documents exist at ``directory``; an empty store if none do.

        Returning an empty store rather than raising is what makes ``fetch --resume``
        (the default) and a from-scratch first run the same code path.
        """
        directory = Path(directory)
        documents: dict[str, Document] = {}
        docs_path = directory / DOCUMENTS_FILENAME
        if docs_path.exists():
            with docs_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    doc = Document(
                        doc_id=raw["doc_id"],
                        title=raw["title"],
                        authors=tuple(raw["authors"]),
                        published=raw["published"],
                        updated=raw["updated"],
                        abstract=raw["abstract"],
                    )
                    documents[doc.doc_id] = doc
        return cls(documents, directory)

    @staticmethod
    def save(documents: Sequence[Document], directory: str | Path) -> None:
        """Rewrite ``documents.jsonl`` in full, sorted by doc_id for a stable diff.

        Whole-file rewrite rather than append: the caller (``fetch``) already holds
        the complete merged set in memory, and sorted, deterministic output is more
        valuable here than incremental I/O at this corpus size.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        docs_path = directory / DOCUMENTS_FILENAME
        with docs_path.open("w", encoding="utf-8") as handle:
            for doc in sorted(documents, key=lambda d: d.doc_id):
                handle.write(json.dumps(document_to_dict(doc), ensure_ascii=False))
                handle.write("\n")


