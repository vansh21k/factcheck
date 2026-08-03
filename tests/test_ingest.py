"""Tests for corpus ingest: ``ArxivClient``, ``JsonlDocumentStore``, and ``fc-fetch``.

Everything here runs against recorded fixtures (``tests/fixtures/arxiv_*.xml``) via
an injected ``fetch_url``, never the live service -- see the spec's "Ingest tests"
paragraph. The two things the spec calls out explicitly as load-bearing get direct
coverage: version-suffixed IDs staying distinct, and zero resolved documents
failing loudly instead of reading as a quiet, all-``unknown`` corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, ClassVar

import pytest

import factcheck.cli.fetch as fetch_cli
from factcheck.corpus import corpus_hash
from factcheck.errors import EmptyCorpusError
from factcheck.ingest.arxiv import ArxivClient, FetchOutcome, find_unresolved
from factcheck.ingest.store import JsonlDocumentStore
from factcheck.types import Chunk, Document

FIXTURES = Path(__file__).parent / "fixtures"
RESPONSE_XML = (FIXTURES / "arxiv_response.xml").read_bytes()
EMPTY_XML = (FIXTURES / "arxiv_empty.xml").read_bytes()


def _fetch_stub(body: bytes = RESPONSE_XML) -> Callable[[str], bytes]:
    def _fetch(url: str) -> bytes:
        return body

    return _fetch


def _no_sleep(seconds: float) -> None:
    pass


# --------------------------------------------------------------------------- #
# ArxivClient: parsing and version handling
# --------------------------------------------------------------------------- #


def test_version_suffixed_ids_preserved_as_distinct_documents() -> None:
    client = ArxivClient(fetch_url=_fetch_stub(), sleep=_no_sleep)
    outcome = client.fetch(["2307.03172v1", "2307.03172v3"])

    by_id = {doc.doc_id: doc for doc in outcome.documents}
    assert set(by_id) >= {"2307.03172v1", "2307.03172v3"}
    v1, v3 = by_id["2307.03172v1"], by_id["2307.03172v3"]

    assert v1.doc_id != v3.doc_id
    assert v1.abstract != v3.abstract
    assert v1.title == v3.title  # same paper, same title, different revision
    assert outcome.unresolved == []


def test_unsuffixed_request_resolves_to_the_latest_version() -> None:
    # The fixture's only entry for this paper is v4; requesting the bare ID must
    # still count as resolved, because the live API resolves an unsuffixed ID to
    # whatever is currently latest.
    client = ArxivClient(fetch_url=_fetch_stub(), sleep=_no_sleep)
    outcome = client.fetch(["2005.11401"])

    assert outcome.unresolved == []
    assert any(doc.doc_id == "2005.11401v4" for doc in outcome.documents)


def test_unresolved_ids_are_reported() -> None:
    client = ArxivClient(fetch_url=_fetch_stub(), sleep=_no_sleep)
    outcome = client.fetch(["2307.03172v1", "9999.99999"])

    assert outcome.unresolved == ["9999.99999"]


def test_whitespace_normalized_but_abstract_content_preserved() -> None:
    client = ArxivClient(fetch_url=_fetch_stub(), sleep=_no_sleep)
    outcome = client.fetch(["2307.03172v1"])
    doc = next(d for d in outcome.documents if d.doc_id == "2307.03172v1")

    assert doc.title == "Lost in the Middle: How Language Models Use Long Contexts"
    assert "\n" not in doc.title
    assert "\n" not in doc.abstract
    assert "  " not in doc.abstract
    # The wrapped/indented fixture text must survive as a literal substring once
    # collapsed -- this is exactly what span validation later depends on.
    assert "ORIGINAL-FINDING-V1." in doc.abstract
    assert not doc.abstract.startswith(" ")
    assert not doc.abstract.endswith(" ")


def test_zero_entries_yields_zero_documents_and_all_unresolved() -> None:
    # ArxivClient itself stays quiet on zero documents -- failing loudly is the
    # fetch CLI's job (see test_fetch_cli_raises_on_zero_documents below), because
    # ArxivClient.fetch is also used mid-resume where a temporarily empty batch
    # must not abort an otherwise-successful run.
    client = ArxivClient(fetch_url=_fetch_stub(EMPTY_XML), sleep=_no_sleep)
    outcome = client.fetch(["9999.99999"])

    assert outcome.documents == []
    assert outcome.unresolved == ["9999.99999"]


# --------------------------------------------------------------------------- #
# ArxivClient: batching and rate limiting
# --------------------------------------------------------------------------- #


def test_batches_into_ceil_n_over_batch_size_requests() -> None:
    calls = []

    def fetch_url(url: str) -> bytes:
        calls.append(url)
        return EMPTY_XML

    client = ArxivClient(batch_size=3, fetch_url=fetch_url, sleep=_no_sleep)
    client.fetch([f"id{i}" for i in range(7)])  # ceil(7 / 3) == 3

    assert len(calls) == 3


def test_batch_urls_carry_only_that_batchs_ids() -> None:
    calls = []

    def fetch_url(url: str) -> bytes:
        calls.append(url)
        return EMPTY_XML

    client = ArxivClient(batch_size=2, fetch_url=fetch_url, sleep=_no_sleep)
    client.fetch(["a", "b", "c"])

    assert "id_list=a%2Cb" in calls[0] or "id_list=a,b" in calls[0]
    assert "id_list=c" in calls[1]


def test_rate_limiter_called_between_but_not_before_first_request() -> None:
    sleeps: list[float] = []
    client = ArxivClient(
        batch_size=2, rate_limit_s=1.5, fetch_url=_fetch_stub(EMPTY_XML), sleep=sleeps.append
    )
    client.fetch(["a", "b", "c", "d", "e"])  # 3 batches -> 2 gaps

    assert sleeps == [1.5, 1.5]


def test_no_sleep_for_a_single_batch() -> None:
    sleeps: list[float] = []
    client = ArxivClient(
        batch_size=50, rate_limit_s=1.5, fetch_url=_fetch_stub(EMPTY_XML), sleep=sleeps.append
    )
    client.fetch(["a", "b"])

    assert sleeps == []


def test_on_batch_callback_fires_per_batch_before_the_next_request() -> None:
    seen: list[list[str]] = []
    client = ArxivClient(batch_size=1, fetch_url=_fetch_stub(), sleep=_no_sleep)
    client.fetch(
        ["2307.03172v1", "2307.03172v3"],
        on_batch=lambda docs: seen.append([d.doc_id for d in docs]),
    )

    assert len(seen) == 2  # one callback per request, not one at the end


# --------------------------------------------------------------------------- #
# find_unresolved
# --------------------------------------------------------------------------- #


def test_find_unresolved_is_case_insensitive() -> None:
    docs = [_doc("2005.11401v4")]
    assert find_unresolved(["2005.11401V4"], docs) == []


def test_find_unresolved_requires_exact_version_match() -> None:
    docs = [_doc("2307.03172v1")]
    assert find_unresolved(["2307.03172v3"], docs) == ["2307.03172v3"]


# --------------------------------------------------------------------------- #
# JsonlDocumentStore
# --------------------------------------------------------------------------- #


def test_documentstore_roundtrip_preserves_fields_exactly(tmp_path: Path) -> None:
    docs = [
        _doc(
            "2307.03172v1",
            title="Lost in the Middle",
            authors=("Nelson F. Liu", "Kevin Lin"),
            abstract='Abstract with a "curly quote" and an em—dash and Ünïcode.',
        ),
        _doc(
            "2307.03172v3",
            title="Lost in the Middle",
            authors=("Nelson F. Liu", "Kevin Lin"),
            abstract="A materially different revised abstract.",
        ),
    ]

    JsonlDocumentStore.save(docs, tmp_path)
    store = JsonlDocumentStore.load(tmp_path)

    assert store.get("2307.03172v1") == docs[0]
    assert store.get("2307.03172v3") == docs[1]
    with pytest.raises(KeyError):
        store.get("does-not-exist")


def test_documentstore_save_is_sorted_by_doc_id(tmp_path: Path) -> None:
    JsonlDocumentStore.save([_doc("b"), _doc("a")], tmp_path)
    lines = (tmp_path / "documents.jsonl").read_text(encoding="utf-8").splitlines()

    assert [json.loads(line)["doc_id"] for line in lines] == ["a", "b"]


def test_load_on_missing_directory_returns_empty_store(tmp_path: Path) -> None:
    store = JsonlDocumentStore.load(tmp_path / "does-not-exist-yet")
    assert len(store) == 0
    assert list(store) == []


def test_iter_chunks_yields_nothing_when_no_chunks_file(tmp_path: Path) -> None:
    JsonlDocumentStore.save([_doc("a")], tmp_path)
    store = JsonlDocumentStore.load(tmp_path)

    assert list(store.iter_chunks()) == []


def test_iter_chunks_reads_chunks_written_by_indexing(tmp_path: Path) -> None:
    JsonlDocumentStore.save([_doc("a")], tmp_path)
    chunk_row = {"chunk_id": "a:0", "doc_id": "a", "text": "hello", "char_start": 0, "char_end": 5}
    (tmp_path / "chunks.jsonl").write_text(json.dumps(chunk_row) + "\n", encoding="utf-8")

    store = JsonlDocumentStore.load(tmp_path)
    assert list(store.iter_chunks()) == [
        Chunk(chunk_id="a:0", doc_id="a", text="hello", char_start=0, char_end=5)
    ]


def test_corpus_hash_is_stable_regardless_of_input_order() -> None:
    a, b = _doc("a"), _doc("b")
    assert corpus_hash([a, b]) == corpus_hash([b, a])


def test_corpus_hash_changes_with_content() -> None:
    a = _doc("a", abstract="original")
    a_edited = _doc("a", abstract="edited")
    assert corpus_hash([a]) != corpus_hash([a_edited])


# --------------------------------------------------------------------------- #
# fc-fetch CLI: resumability, empty-corpus failure, reporting
# --------------------------------------------------------------------------- #


class _StubClient:
    """Drop-in replacement for ArxivClient that never touches the network.

    Records the IDs it was asked to fetch, so tests can assert on exactly what the
    CLI decided still needed fetching (the resumability contract) without caring
    how ArxivClient itself batches or rate-limits.
    """

    instances: ClassVar[list[_StubClient]] = []

    def __init__(self, **_: object) -> None:
        self.requested: list[str] = []
        _StubClient.instances.append(self)

    def fetch(
        self, ids: list[str], *, on_batch: Callable[[list[Document]], None] | None = None
    ) -> FetchOutcome:
        self.requested = list(ids)
        docs = [_doc(_response_doc_id(i)) for i in ids]
        if on_batch is not None:
            on_batch(docs)
        return FetchOutcome(documents=docs, unresolved=[])


def _response_doc_id(requested: str) -> str:
    """A stub 'server' resolving an unsuffixed request to a fake latest version."""
    return requested if "v" in requested.split(".")[-1] else f"{requested}v1"


@pytest.fixture(autouse=True)
def _patch_arxiv_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _StubClient.instances = []
    monkeypatch.setattr(fetch_cli, "ArxivClient", _StubClient)


def _write_ids(tmp_path: Path, ids: list[str]) -> Path:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return ids_file


def test_fetch_cli_resumes_and_skips_already_fetched_ids(tmp_path: Path) -> None:
    out_dir = tmp_path / "data"
    JsonlDocumentStore.save([_doc("2005.11401v1")], out_dir)
    ids_file = _write_ids(tmp_path, ["2005.11401", "2309.01431"])

    rc = fetch_cli.main(["--ids", str(ids_file), "--out", str(out_dir)])

    assert rc == 0
    assert _StubClient.instances[0].requested == ["2309.01431"]  # only the new one
    store = JsonlDocumentStore.load(out_dir)
    assert {"2005.11401v1", "2309.01431v1"} <= {doc.doc_id for doc in store}


def test_fetch_cli_force_refetches_everything(tmp_path: Path) -> None:
    out_dir = tmp_path / "data"
    JsonlDocumentStore.save([_doc("2005.11401v1")], out_dir)
    ids_file = _write_ids(tmp_path, ["2005.11401", "2309.01431"])

    fetch_cli.main(["--ids", str(ids_file), "--out", str(out_dir), "--force"])

    assert _StubClient.instances[0].requested == ["2005.11401", "2309.01431"]


def test_fetch_cli_writes_incrementally_per_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # _StubClient.fetch invokes on_batch once with everything, but the contract
    # under test is that main() persists via that callback rather than only after
    # ArxivClient.fetch returns -- so a store re-read mid-callback would see data.
    out_dir = tmp_path / "data"
    seen_during_batch: dict[str, int] = {}

    class RecordingClient(_StubClient):
        def fetch(
            self, ids: list[str], *, on_batch: Callable[[list[Document]], None] | None = None
        ) -> FetchOutcome:
            docs = [_doc(_response_doc_id(i)) for i in ids]
            if on_batch is not None:
                on_batch(docs)
                seen_during_batch["count"] = len(JsonlDocumentStore.load(out_dir))
            return FetchOutcome(documents=docs, unresolved=[])

    ids_file = _write_ids(tmp_path, ["2005.11401"])
    monkeypatch.setattr(fetch_cli, "ArxivClient", RecordingClient)
    fetch_cli.main(["--ids", str(ids_file), "--out", str(out_dir)])

    assert seen_during_batch["count"] == 1


def test_fetch_cli_raises_on_zero_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ZeroClient(_StubClient):
        def fetch(
            self, ids: list[str], *, on_batch: Callable[[list[Document]], None] | None = None
        ) -> FetchOutcome:
            return FetchOutcome(documents=[], unresolved=list(ids))

    monkeypatch.setattr(fetch_cli, "ArxivClient", ZeroClient)
    ids_file = _write_ids(tmp_path, ["9999.99999"])

    with pytest.raises(EmptyCorpusError):
        fetch_cli.main(["--ids", str(ids_file), "--out", str(tmp_path / "data")])


def test_fetch_cli_reports_unresolved_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class PartialClient(_StubClient):
        def fetch(
            self, ids: list[str], *, on_batch: Callable[[list[Document]], None] | None = None
        ) -> FetchOutcome:
            docs = [_doc("2005.11401v1")]
            if on_batch is not None:
                on_batch(docs)
            return FetchOutcome(documents=docs, unresolved=["9999.99999"])

    monkeypatch.setattr(fetch_cli, "ArxivClient", PartialClient)
    ids_file = _write_ids(tmp_path, ["2005.11401", "9999.99999"])

    rc = fetch_cli.main(["--ids", str(ids_file), "--out", str(tmp_path / "data")])

    assert rc == 0
    out = capsys.readouterr().out
    assert "9999.99999" in out
    assert "requested=2" in out
    assert "resolved=1" in out
    assert "unresolved=1" in out


def test_fetch_cli_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text(
        "# a comment\n\n2005.11401  # inline comment\n\n2309.01431\n", encoding="utf-8"
    )

    fetch_cli.main(["--ids", str(ids_file), "--out", str(tmp_path / "data")])

    assert _StubClient.instances[0].requested == ["2005.11401", "2309.01431"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _doc(
    doc_id: str,
    *,
    title: str = "Title",
    authors: tuple[str, ...] = ("Author",),
    published: str = "2023-01-01T00:00:00Z",
    updated: str = "2023-01-01T00:00:00Z",
    abstract: str = "Abstract text.",
) -> Document:
    return Document(
        doc_id=doc_id,
        title=title,
        authors=authors,
        published=published,
        updated=updated,
        abstract=abstract,
    )
