"""`fc-index`: build a retrieval index from `documents.jsonl`, with no network access.

Reads `documents.jsonl` with a small local loop rather than importing the ingest
package's document store -- `index` has no import-time dependency on `ingest`, only
a file-format agreement (`doc_id`, `title`, `authors`, `published`, `updated`,
`abstract` per line).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ..config import Config
from ..indexing.build import build_index
from ..types import Document

DOCUMENTS_FILENAME = "documents.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    overrides: dict[str, str] = {}
    for item in args.set:
        if "=" not in item:
            print(f"fc-index: --set expects path=value, got {item!r}")
            return 2
        path, value = item.split("=", 1)
        overrides[path] = value

    try:
        cfg = Config.load(args.config, overrides or None)
    except Exception as exc:  # ConfigError or a bad --config path
        print(f"fc-index: {exc}")
        return 2

    docs_path = Path(args.docs) / DOCUMENTS_FILENAME
    try:
        documents = _read_documents(docs_path)
    except FileNotFoundError as exc:
        print(f"fc-index: {exc}")
        return 2

    if not documents:
        print(f"fc-index: {docs_path} contains zero documents; refusing to build an index.")
        return 1

    out_dir = Path(args.out)
    try:
        manifest = build_index(documents, cfg, out_dir, rebuild=args.rebuild)
    except FileExistsError as exc:
        print(f"fc-index: {exc}")
        return 1

    print(f"indexed {manifest.n_docs} documents into {manifest.n_chunks} chunks")
    print(f"manifest: {out_dir / 'manifest.json'}")
    print(f"config hash: {cfg.hash}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="fc-index", description=__doc__)
    parser.add_argument("--docs", default="data/", help="directory containing documents.jsonl")
    parser.add_argument("--out", default="index/", help="directory to write the index into")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="path=value",
        help="override a config value, e.g. --set chunk.size=200 (repeatable)",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="replace an existing index at --out"
    )
    return parser.parse_args(argv)


def _read_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"no {DOCUMENTS_FILENAME} at {path}. Run `fc-fetch` first.")
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            documents.append(
                Document(
                    doc_id=obj["doc_id"],
                    title=obj["title"],
                    authors=tuple(obj["authors"]),
                    published=obj["published"],
                    updated=obj["updated"],
                    abstract=obj["abstract"],
                )
            )
    return documents


if __name__ == "__main__":
    raise SystemExit(main())
