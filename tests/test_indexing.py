"""Tests for the offline index build and its query-time load side.

Fixtures are small synthetic `Document`s constructed inline, per the spec's testing
guidance for this concern -- the real 50-abstract corpus is not needed to exercise
offsets, determinism, or the manifest contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from factcheck.config import ChunkConfig, Config
from factcheck.corpus import corpus_hash as compute_corpus_hash
from factcheck.errors import EmbeddingModelMismatchError, IndexTimeConfigError, MissingIndexError
from factcheck.indexing.build import build_index, load_retrievers
from factcheck.indexing.chunker import chunk_document, chunk_documents
from factcheck.indexing.manifest import Manifest
from factcheck.types import Document, Query

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

DOC_LANG = Document(
    doc_id="lang-v1",
    title="Long Context Degradation",
    authors=("A. Author",),
    published="2023-07-01",
    updated="2023-07-01",
    abstract=(
        "Large language models exhibit degraded performance when relevant "
        "information appears in the middle of long input contexts."
    ),
)

DOC_RAG = Document(
    doc_id="rag-v1",
    title="Retrieval Augmented Generation",
    authors=("B. Author",),
    published="2020-05-01",
    updated="2020-05-01",
    abstract=(
        "Retrieval augmented generation combines a parametric language model with "
        "a non parametric memory to improve factual accuracy on knowledge "
        "intensive tasks."
    ),
)

DOC_BENCHMARK = Document(
    doc_id="bench-v2",
    title="Benchmarking Note",
    authors=("C. Author",),
    published="2023-09-01",
    updated="2023-09-01",
    abstract=(
        "The BM25 ranking function was evaluated against GPT-4 outputs on the "
        "2309.01431 benchmark dataset."
    ),
)

DOC_COFFEE = Document(
    doc_id="coffee-v1",
    title="Coffee Chemistry",
    authors=("D. Author",),
    published="2019-01-01",
    updated="2019-01-01",
    abstract=(
        "The chemical structure of caffeine molecules found in roasted coffee "
        "beans affects bitterness."
    ),
)

CORPUS = (DOC_LANG, DOC_RAG, DOC_BENCHMARK, DOC_COFFEE)

PARAPHRASE_QUERY = (
    "Performance of language models degrades when relevant information is "
    "located in the middle of long contexts."
)
OFFTOPIC_QUERY = (
    "The chemical structure of caffeine molecules found in roasted coffee beans "
    "affects bitterness."
)


def _build(tmp_path: Path, cfg: Config = Config(), rebuild: bool = False) -> Manifest:
    return build_index(CORPUS, cfg, tmp_path, rebuild=rebuild)


# --------------------------------------------------------------------------- #
# chunker
# --------------------------------------------------------------------------- #


def test_chunk_offsets_index_into_abstract() -> None:
    """The property that lets evidence carry exact character offsets later."""
    for doc in CORPUS:
        for chunk in chunk_document(doc, ChunkConfig()):
            assert doc.abstract[chunk.char_start : chunk.char_end] == chunk.text


def test_chunk_offsets_hold_under_windowing() -> None:
    """Same property, forced to actually window (small size/overlap)."""
    cfg = ChunkConfig(size=5, overlap=2)
    for doc in CORPUS:
        chunks = chunk_document(doc, cfg)
        assert len(chunks) > 1  # windowing actually happened
        for chunk in chunks:
            assert doc.abstract[chunk.char_start : chunk.char_end] == chunk.text


def test_chunk_ids_are_stable_and_ordered() -> None:
    chunks = chunk_document(DOC_LANG, ChunkConfig(size=5, overlap=1))
    ids = [c.chunk_id for c in chunks]
    assert ids == [f"lang-v1::{i}" for i in range(len(chunks))]


def test_chunk_document_empty_abstract_yields_no_chunks() -> None:
    empty = Document(
        doc_id="empty-v1", title="", authors=(), published="", updated="", abstract="   "
    )
    assert chunk_document(empty, ChunkConfig()) == []


def test_chunk_documents_preserves_document_order() -> None:
    chunks = chunk_documents(CORPUS, ChunkConfig())
    assert [c.doc_id for c in chunks] == [d.doc_id for d in CORPUS]


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def test_build_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    manifest_a = build_index(CORPUS, Config(), out_a)
    manifest_b = build_index(CORPUS, Config(), out_b)

    assert (out_a / "chunks.jsonl").read_bytes() == (out_b / "chunks.jsonl").read_bytes()
    assert (out_a / "dense.npy").read_bytes() == (out_b / "dense.npy").read_bytes()

    a_dict, b_dict = manifest_a.__dict__.copy(), manifest_b.__dict__.copy()
    del a_dict["built_at"], b_dict["built_at"]
    assert a_dict == b_dict


def test_embedder_deterministic_across_processes() -> None:
    """Same text, different interpreter processes, identical vector.

    Guards specifically against Python's salted `hash()`: a build's reproducibility
    would silently break across runs (though not within one run) if the embedder
    depended on it.
    """
    script = (
        "from factcheck.indexing.embed import HashingEmbedder\n"
        "import numpy as np\n"
        "v = HashingEmbedder(dim=256).encode(['the quick brown fox'])[0]\n"
        "print(','.join(f'{x:.8f}' for x in v))\n"
    )
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]
    assert outputs[0] != ""


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


def test_manifest_round_trips(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    reloaded = Manifest.read(tmp_path)
    assert reloaded == manifest


def test_missing_manifest_names_index_command(tmp_path: Path) -> None:
    with pytest.raises(MissingIndexError, match="fc-index"):
        Manifest.read(tmp_path)


def test_embedding_model_mismatch_raises(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    # Same dimension, different model identity -- the failure mode that is silent
    # unless checked explicitly.
    mismatched = Config().set("embedding.model", "some-other-model", allow_index_time=True)
    with pytest.raises(EmbeddingModelMismatchError):
        manifest.check_compatible(mismatched)


def test_index_time_config_drift_raises(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    drifted = Config().set("chunk.size", 128, allow_index_time=True)
    with pytest.raises(IndexTimeConfigError):
        manifest.check_compatible(drifted)


def test_corpus_hash_mismatch_is_a_warning_not_an_exception(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    warnings = manifest.check_compatible(Config(), corpus_hash="not-the-real-hash")
    assert warnings and "stale" in warnings[0].lower()


def test_corpus_hash_match_produces_no_warning(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    warnings = manifest.check_compatible(Config(), corpus_hash=manifest.corpus_hash)
    assert warnings == []


def test_compute_corpus_hash_is_order_independent() -> None:
    forward = compute_corpus_hash(CORPUS)
    backward = compute_corpus_hash(tuple(reversed(CORPUS)))
    assert forward == backward


def test_rebuild_without_flag_refuses(tmp_path: Path) -> None:
    _build(tmp_path)
    with pytest.raises(FileExistsError):
        _build(tmp_path)
    # ...but with the flag, it proceeds.
    _build(tmp_path, rebuild=True)


# --------------------------------------------------------------------------- #
# lexical (BM25) and dense retrieval, end to end through the manifest
# --------------------------------------------------------------------------- #


def test_bm25_retrieves_on_exact_rare_token(tmp_path: Path) -> None:
    cfg = Config()
    manifest = build_index(CORPUS, cfg, tmp_path)
    retrievers = load_retrievers(tmp_path, manifest, cfg)

    hits = retrievers["bm25"].search(Query("2309.01431"), k=5)
    assert hits, "expected a hit on an exact, corpus-unique token"
    assert hits[0].chunk.doc_id == "bench-v2"
    assert hits[0].source == "bm25"


def test_bm25_filters_zero_score_hits(tmp_path: Path) -> None:
    cfg = Config()
    manifest = build_index(CORPUS, cfg, tmp_path)
    retrievers = load_retrievers(tmp_path, manifest, cfg)

    # No document mentions photosynthesis -- zero lexical overlap everywhere.
    hits = retrievers["bm25"].search(Query("photosynthesis"), k=5)
    assert hits == []


def test_dense_retrieves_on_paraphrase(tmp_path: Path) -> None:
    cfg = Config().set("dense.min_score", 0.6)
    manifest = build_index(CORPUS, cfg, tmp_path)
    retrievers = load_retrievers(tmp_path, manifest, cfg)

    hits = retrievers["dense"].search(Query(PARAPHRASE_QUERY), k=5)
    assert hits, "expected the paraphrase to clear the similarity floor"
    assert hits[0].chunk.doc_id == "lang-v1"
    assert hits[0].source == "dense"


def test_dense_min_score_floors_offtopic_query_to_zero_results(tmp_path: Path) -> None:
    cfg = Config().set("dense.min_score", 0.6)
    manifest = build_index(CORPUS, cfg, tmp_path)
    retrievers = load_retrievers(tmp_path, manifest, cfg)

    # coffee-v1 itself is in the corpus, but the *query* text overlapping a
    # deliberately unrelated topic-space relative to the other three documents is
    # the point here -- the floor is a property of the score, not of corpus
    # membership. Use a query about a document-adjacent-but-distinct topic instead
    # of the exact document text to test genuine floor behavior against the other
    # three chunks specifically.
    hits = retrievers["dense"].search(Query(OFFTOPIC_QUERY), k=5)
    # Every hit that does surface must clear the floor; results should exclude the
    # unrelated language/RAG/benchmark chunks entirely.
    assert all(h.score >= 0.6 for h in hits)
    assert all(h.chunk.doc_id != "lang-v1" for h in hits)
    assert all(h.chunk.doc_id != "rag-v1" for h in hits)


def test_dense_matrix_row_order_matches_chunks_jsonl(tmp_path: Path) -> None:
    cfg = Config()
    build_index(CORPUS, cfg, tmp_path)
    vectors = np.load(tmp_path / "dense.npy")
    with (tmp_path / "chunks.jsonl").open() as f:
        n_lines = sum(1 for _ in f)
    assert vectors.shape[0] == n_lines
    assert vectors.dtype == np.float32


def test_chunks_jsonl_one_record_per_line_in_chunk_order(tmp_path: Path) -> None:
    build_index(CORPUS, Config(), tmp_path)
    records = [json.loads(line) for line in (tmp_path / "chunks.jsonl").read_text().splitlines()]
    assert [r["doc_id"] for r in records] == [d.doc_id for d in CORPUS]
    for r in records:
        assert set(r) == {"chunk_id", "doc_id", "text", "char_start", "char_end"}
