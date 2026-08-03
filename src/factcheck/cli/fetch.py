"""``fc-fetch``: pull the fixed corpus from arXiv once, store it, stop.

This is the only program allowed to touch the network (see ``ArxivClient``). It is
also resumable by default: a network failure on batch 4 of 5 must not force
re-fetching batches 1-3, so every batch is persisted to disk as it arrives and a
re-run skips whatever is already on disk.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..errors import EmptyCorpusError
from ..ingest.arxiv import ArxivClient, find_unresolved
from ..ingest.store import JsonlDocumentStore
from ..types import Document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fc-fetch", description="Fetch the arXiv corpus.")
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("corpus/arxiv_ids.txt"),
        help="File of arXiv IDs, one per line, '#' comments allowed.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data"), help="Directory to write documents.jsonl into."
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--rate-limit", type=float, default=3.0, help="Seconds between requests.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any documents already on disk and refetch every ID.",
    )
    return parser


def _read_ids(path: Path) -> list[str]:
    ids = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    requested_ids = _read_ids(args.ids)
    already_fetched = [] if args.force else list(JsonlDocumentStore.load(args.out))
    ids_to_fetch = requested_ids if args.force else find_unresolved(requested_ids, already_fetched)

    # doc_id keyed so a re-fetched version of an already-present doc_id (shouldn't
    # happen in practice, but --force + a partial prior run can produce it) simply
    # overwrites rather than duplicating a line in the saved file.
    fetched: dict[str, Document] = {doc.doc_id: doc for doc in already_fetched}

    def _persist(batch_documents: list[Document]) -> None:
        for doc in batch_documents:
            fetched[doc.doc_id] = doc
        JsonlDocumentStore.save(list(fetched.values()), args.out)

    client = ArxivClient(batch_size=args.batch_size, rate_limit_s=args.rate_limit)
    client.fetch(ids_to_fetch, on_batch=_persist)

    unresolved = find_unresolved(requested_ids, fetched.values())
    if unresolved:
        print("Unresolved IDs (requested but not returned by the API):")
        for doc_id in unresolved:
            print(f"  {doc_id}")

    if not fetched:
        raise EmptyCorpusError(
            "fetch resolved zero documents -- check network access, the endpoint, "
            "and that the ID list isn't empty before trusting any downstream verdict"
        )

    distinct_papers = len({doc.base_id for doc in fetched.values()})
    print(
        f"requested={len(requested_ids)} resolved={len(fetched)} "
        f"unresolved={len(unresolved)} distinct_papers={distinct_papers}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
