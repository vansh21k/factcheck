"""Orchestrates a full index build, and the query-side counterpart that loads one.

No network access anywhere in this module, or anything it imports -- index builds
must be reproducible offline. `build_index` takes `Sequence[Document]` rather than a
path, so it has no dependency on how documents got onto disk; only the CLI reads
`documents.jsonl`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..corpus import corpus_hash
from ..ports import Retriever
from ..types import Chunk, Document
from .chunker import CHUNKER_VERSION, chunk_documents
from .dense import DenseBuilder, DenseLoader
from .embed import get_embedder
from .lexical import BM25Builder, BM25Loader
from .manifest import MANIFEST_FILENAME, Manifest

CHUNKS_FILENAME = "chunks.jsonl"


def build_index(
    documents: Sequence[Document],
    cfg: Config,
    out_dir: Path,
    rebuild: bool = False,
) -> Manifest:
    """Chunk every document, run every builder, write chunks + manifest.

    Refuses if a manifest already exists and `rebuild` is false -- expensive work
    (re-embedding the whole corpus) must never happen as a side effect of something
    else, e.g. a CLI invocation that meant to just inspect the index.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / MANIFEST_FILENAME
    if manifest_path.exists() and not rebuild:
        raise FileExistsError(
            f"an index already exists at {out_dir}. Pass --rebuild to replace it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = chunk_documents(documents, cfg.index.chunk)
    _write_chunks(chunks, out_dir)

    embedder = get_embedder(cfg.index.embedding)
    bm25_builder = BM25Builder(cfg.query.bm25)
    dense_builder = DenseBuilder(embedder)
    entries = {
        bm25_builder.name: bm25_builder.build(chunks, out_dir),
        dense_builder.name: dense_builder.build(chunks, out_dir),
    }

    manifest = Manifest(
        corpus_hash=corpus_hash(documents),
        embedding_model=embedder.model_id,
        embedding_dim=embedder.dim,
        chunker_version=CHUNKER_VERSION,
        chunk_size=cfg.index.chunk.size,
        chunk_overlap=cfg.index.chunk.overlap,
        built_at=datetime.now(timezone.utc).isoformat(),
        n_docs=len(documents),
        n_chunks=len(chunks),
        index_config=cfg.index_fingerprint(),
        entries=entries,
    )
    manifest.write(out_dir)
    return manifest


def load_retrievers(out_dir: Path, manifest: Manifest, cfg: Config) -> dict[str, Retriever]:
    """The query-time counterpart: build each `Retriever` from artifacts already on
    disk. Never re-embeds or re-fits anything -- that is what `build_index` is for.
    """
    out_dir = Path(out_dir)
    embedder = get_embedder(cfg.index.embedding)
    manifest_dict = manifest.entries
    return {
        "bm25": BM25Loader(cfg.query.bm25).load(out_dir, manifest_dict),
        "dense": DenseLoader(embedder, cfg.query.dense).load(out_dir, manifest_dict),
    }


def _write_chunks(chunks: Sequence[Chunk], out_dir: Path) -> None:
    """One JSON object per line, in chunk order -- the same order `dense.npy`'s rows
    are in, which is the positional contract `dense.py` relies on.
    """
    with (out_dir / CHUNKS_FILENAME).open("w", encoding="utf-8") as f:
        for c in chunks:
            record = {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")
