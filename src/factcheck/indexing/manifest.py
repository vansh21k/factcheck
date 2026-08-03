"""The contract between build time and query time.

Everything query-time code needs to know about how an index was built lives here:
what corpus it was built from, what embedding model, what chunker settings, when.
`check_compatible` is where a config change against a prebuilt index becomes an
error instead of a plausible-looking number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import EmbeddingModelMismatchError, IndexTimeConfigError, MissingIndexError

MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class Manifest:
    """Recorded once at build time, read (never mutated) at query time."""

    corpus_hash: str
    embedding_model: str
    embedding_dim: int
    chunker_version: str
    chunk_size: int
    chunk_overlap: int
    built_at: str  # ISO-8601 UTC
    n_docs: int
    n_chunks: int
    index_config: dict[str, Any]
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def write(self, out_dir: Path) -> None:
        payload = asdict(self)
        (Path(out_dir) / MANIFEST_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def read(cls, out_dir: Path) -> Manifest:
        path = Path(out_dir) / MANIFEST_FILENAME
        if not path.exists():
            # Never an implicit rebuild: name the command that fixes this instead of
            # doing the expensive thing on the caller's behalf.
            raise MissingIndexError(
                f"no index at {out_dir} (no {MANIFEST_FILENAME}). "
                f"Run `fc-index --docs <dir> --out {out_dir}` to build one."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def check_compatible(self, cfg: Config, corpus_hash: str | None = None) -> list[str]:
        """Raise on hard incompatibilities; return warnings for soft ones.

        Order matters: the embedding-model identity check runs before the general
        index-config diff, because it covers the one failure mode that is *not*
        "loud on its own". A dimension change already breaks matrix shapes at query
        time -- that is caught by the config diff below regardless. A *different*
        model at the *same* dimension breaks nothing mechanically: it loads, it
        scores, it ranks, and it returns confident nonsense, because a dimension-256
        float32 vector carries no signature of which model produced it. Only an
        explicit identity check catches that, so it is checked first and separately.
        """
        embedding = cfg.index.embedding
        if embedding.model != self.embedding_model:
            raise EmbeddingModelMismatchError(
                f"configured embedding model {embedding.model!r} does not match this "
                f"index's {self.embedding_model!r}. Run `fc-index --rebuild` to "
                f"rebuild with the configured model."
            )

        current = cfg.index_fingerprint()
        if current != self.index_config:
            raise IndexTimeConfigError(
                "index-time config no longer matches the config this index was built "
                "with. Run `fc-index --rebuild` to rebuild against the current config."
            )

        warnings: list[str] = []
        if corpus_hash is not None and corpus_hash != self.corpus_hash:
            # A warning, not a raise: a stale index should still answer, just with a
            # visible caveat -- silently refusing to serve query time at all would be
            # a worse failure mode than a slightly stale corpus.
            warnings.append(
                f"index corpus hash ({self.corpus_hash[:12]}...) differs from the "
                f"current documents ({corpus_hash[:12]}...); the index may be stale. "
                f"Run `fc-index --rebuild` to refresh it."
            )
        return warnings
