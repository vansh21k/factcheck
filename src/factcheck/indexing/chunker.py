"""Deterministic chunking with recorded character offsets.

The document text *is* the abstract (full-text ingestion is out of scope), so at the
configured 400-token size an abstract is almost always one chunk. That is expected,
not a bug to work around: the offsets machinery has to be real from day one, because
at corpus scale the only thing that changes is that windows actually split -- the
`char_start`/`char_end` contract that lets evidence carry exact offsets is the same
contract either way.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..config import ChunkConfig
from ..types import Chunk, Document

# Bumped whenever the windowing algorithm changes in a way that would shift offsets
# for the same document + config -- recorded in the manifest so two index builds can
# be told apart even when their config happens to match.
CHUNKER_VERSION = "v1"

# A "token" is a whitespace-delimited word. Finding spans (not just splitting) is what
# lets a chunk's text be `abstract[start:end]` by construction rather than by
# reconstruction -- there is no join step that could introduce whitespace drift.
_TOKEN_RE = re.compile(r"\S+")


def chunk_document(doc: Document, cfg: ChunkConfig) -> list[Chunk]:
    """Split one document into overlapping token windows.

    Edge cases: an empty (or whitespace-only) abstract produces zero chunks -- there
    is no span to index. ``overlap >= size`` would make the window advance by zero or
    a negative amount, so the step is floored at 1 token to guarantee termination.
    """
    tokens = [m.span() for m in _TOKEN_RE.finditer(doc.abstract)]
    if not tokens:
        return []

    step = max(1, cfg.size - cfg.overlap)
    n = len(tokens)
    chunks: list[Chunk] = []
    ordinal = 0
    i = 0
    while i < n:
        window = tokens[i : i + cfg.size]
        char_start, char_end = window[0][0], window[-1][1]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{ordinal}",
                doc_id=doc.doc_id,
                text=doc.abstract[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
            )
        )
        ordinal += 1
        if i + cfg.size >= n:
            break
        i += step
    return chunks


def chunk_documents(documents: Sequence[Document], cfg: ChunkConfig) -> list[Chunk]:
    """Chunk a corpus in document order. Chunk order is the index's on-disk order."""
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, cfg))
    return chunks
