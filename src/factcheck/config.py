"""The tunable surface, and the line it may not cross.

One frozen tree, loadable from file, overridable per invocation, hashed. Index-time
fields are a *different type* from query-time fields rather than a comment, because
sweeping ``chunk.size`` against a prebuilt index does not crash -- it produces a
plausible number as unrelated noise moves the metric.

There is deliberately no ``validator.fuzzy_match_threshold``. Normalization knobs
adjust *how* text is compared; none of them adjust *whether* the quote must be
present. Config tunes quality, never the invariant.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError, IndexTimeConfigError

VERIFIER_MODEL = "claude-sonnet-5"
AUDITOR_MODEL = "claude-haiku-4-5-20251001"
EXPANDER_MODEL = "claude-haiku-4-5-20251001"


# --------------------------------------------------------------------------- #
# index time: changing any of these invalidates the index
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkConfig:
    size: int = 400
    overlap: int = 64


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "hash-charngram-v1"
    dim: int = 256


@dataclass(frozen=True)
class IndexTimeConfig:
    chunk: ChunkConfig = ChunkConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()


# --------------------------------------------------------------------------- #
# query time: free to sweep, no rebuild
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExpanderConfig:
    enabled: bool = True
    n_queries: int = 3
    include_negation: bool = True
    model: str = EXPANDER_MODEL
    temperature: float = 0.0
    prompt_version: str = "v1"


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2
    b: float = 0.75
    top_k: int = 50
    # Zero here, and calibrated in a config profile instead. BM25 scores are not
    # comparable across corpora -- they move with corpus size and average document
    # length -- so a floor baked into the code is a per-corpus constant pretending
    # to be a default, and its failure mode on a corpus it was not tuned for is an
    # empty result set, which reads as appropriate abstention rather than as a
    # misconfiguration. `config/default.yaml` carries the value measured for this
    # corpus; see also `fusion`, which consumes ranks precisely to avoid this.
    min_score: float = 0.0


@dataclass(frozen=True)
class DenseConfig:
    top_k: int = 50
    # Cosine is bounded and corpus-size invariant, so unlike the BM25 floor this
    # one is a defensible code default -- but it is still *embedder* specific.
    # Measured against the hashed-feature embedder: unrelated claims peak near
    # 0.52, on-topic ones near 0.63. The 0.25 that reads like a sane default
    # filters literally nothing. The margin is thin because this embedder is
    # lexical, not semantic; a trained encoder separates far more widely.
    min_score: float = 0.55


@dataclass(frozen=True)
class FusionConfig:
    strategy: str = "rrf"
    rrf_k: int = 60


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = False
    min_score: float = 0.0
    top_n: int = 8


@dataclass(frozen=True)
class VerifierConfig:
    model: str = VERIFIER_MODEL
    temperature: float = 0.0
    max_tokens: int = 2048
    prompt_version: str = "v3"


@dataclass(frozen=True)
class ValidatorConfig:
    """Knobs that bound evidentiary weight, not presence.

    ``min_quote_chars`` exists because exact-substring validation has a hole: the
    quote "the" is perfectly verbatim and carries zero evidentiary weight. A model
    under pressure to cite can satisfy the span gate with fragments that prove
    nothing, and citation validity rate would read 100% while grounding quietly
    failed. ``max_quote_chars`` closes the mirror-image hole.
    """

    normalize_whitespace: bool = True
    normalize_unicode_quotes: bool = True
    case_insensitive: bool = False
    min_quote_chars: int = 40
    max_quote_chars: int = 400


@dataclass(frozen=True)
class AuditorConfig:
    enabled: bool = True
    model: str = AUDITOR_MODEL
    temperature: float = 0.0
    prompt_version: str = "v2"


@dataclass(frozen=True)
class AggregationConfig:
    policy: str = "contradiction_wins"
    min_supporting_quotes: int = 1


@dataclass(frozen=True)
class QueryTimeConfig:
    expander: ExpanderConfig = ExpanderConfig()
    bm25: BM25Config = BM25Config()
    dense: DenseConfig = DenseConfig()
    fusion: FusionConfig = FusionConfig()
    rerank: RerankConfig = RerankConfig()
    verifier: VerifierConfig = VerifierConfig()
    validator: ValidatorConfig = ValidatorConfig()
    auditor: AuditorConfig = AuditorConfig()
    aggregation: AggregationConfig = AggregationConfig()


@dataclass(frozen=True)
class Config:
    """A whole configuration. Immutable for the duration of a run."""

    index: IndexTimeConfig = IndexTimeConfig()
    query: QueryTimeConfig = QueryTimeConfig()

    # -- dotted-path access ------------------------------------------------- #

    def get(self, path: str) -> Any:
        section, field_name = _split_path(path)
        return getattr(self._section(section), field_name)

    def set(self, path: str, value: Any, *, allow_index_time: bool = False) -> Config:
        """Return a new config with ``path`` set. Never mutates."""
        section, field_name = _split_path(path)
        if section in _INDEX_SECTIONS and not allow_index_time:
            raise IndexTimeConfigError(
                f"'{path}' is an index-time setting and cannot be changed against a "
                f"prebuilt index. Run `fc-index --rebuild` to change it."
            )
        current = self._section(section)
        coerced = _coerce(value, getattr(current, field_name), path)
        updated = replace(current, **{field_name: coerced})
        half = "index" if section in _INDEX_SECTIONS else "query"
        parent = replace(getattr(self, half), **{section: updated})
        return replace(self, **{half: parent})

    def with_overrides(
        self, overrides: dict[str, Any], *, allow_index_time: bool = False
    ) -> Config:
        cfg = self
        for path, value in overrides.items():
            cfg = cfg.set(path, value, allow_index_time=allow_index_time)
        return cfg

    def _section(self, section: str) -> Any:
        half = "index" if section in _INDEX_SECTIONS else "query"
        try:
            return getattr(getattr(self, half), section)
        except AttributeError as exc:  # pragma: no cover - guarded by _split_path
            raise ConfigError(f"unknown config section '{section}'") from exc

    # -- serialization ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {"index": _asdict(self.index), "query": _asdict(self.query)}

    def index_fingerprint(self) -> dict[str, Any]:
        """The index-time subset, as recorded in the manifest."""
        return _asdict(self.index)

    @property
    def hash(self) -> str:
        """Short content hash. Every eval row records it, so a number is traceable."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:7]

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build from a nested or flat mapping, validating every key."""
        cfg = cls()
        for path, value in _flatten(data).items():
            cfg = cfg.set(path, value, allow_index_time=True)
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"config file {path} must contain a mapping")
        return cls.from_dict(data)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Config:
        cfg = cls.from_file(path) if path else cls()
        if overrides:
            cfg = cfg.with_overrides(overrides, allow_index_time=True)
        return cfg


_INDEX_SECTIONS = frozenset(f.name for f in fields(IndexTimeConfig))


def is_index_time(path: str) -> bool:
    section, _ = _split_path(path)
    return section in _INDEX_SECTIONS


def _split_path(path: str) -> tuple[str, str]:
    parts = path.split(".")
    # Tolerate a leading half name so `index.chunk.size` and `chunk.size` both work.
    if len(parts) == 3 and parts[0] in ("index", "query"):
        parts = parts[1:]
    if len(parts) != 2:
        raise ConfigError(f"config path must be 'section.field', got '{path}'")
    section, field_name = parts
    known = _SECTION_FIELDS.get(section)
    if known is None:
        raise ConfigError(f"unknown config section '{section}' in '{path}'")
    if field_name not in known:
        raise ConfigError(
            f"unknown config field '{path}'. Known fields in '{section}': "
            f"{', '.join(sorted(known))}"
        )
    return section, field_name


def _coerce(value: Any, current: Any, path: str) -> Any:
    """Coerce a string override to the type of the value it replaces."""
    if not isinstance(value, str) or isinstance(current, str):
        if isinstance(current, bool) and not isinstance(value, bool):
            raise ConfigError(f"'{path}' expects a boolean, got {value!r}")
        if isinstance(current, float) and isinstance(value, int):
            return float(value)
        return value
    text = value.strip()
    if isinstance(current, bool):
        if text.lower() in ("true", "yes", "1"):
            return True
        if text.lower() in ("false", "no", "0"):
            return False
        raise ConfigError(f"'{path}' expects a boolean, got {value!r}")
    try:
        if isinstance(current, int):
            return int(text)
        if isinstance(current, float):
            return float(text)
    except ValueError as exc:
        raise ConfigError(f"'{path}' expects a {type(current).__name__}, got {value!r}") from exc
    return value


def _asdict(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        out[f.name] = _asdict(value) if is_dataclass(value) else value
    return out


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into dotted paths, dropping the index/query half."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if not prefix and key in ("index", "query") and isinstance(value, dict):
            flat.update(_flatten(value))
        elif isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _section_fields() -> dict[str, frozenset[str]]:
    """Section name -> its known field names, derived from the dataclasses themselves.

    Derived rather than written out so that adding a knob cannot forget to register
    it, and so an unknown path (``validator.fuzzy_match_threshold``) is an error.
    """
    out: dict[str, frozenset[str]] = {}
    halves = ((IndexTimeConfig, IndexTimeConfig()), (QueryTimeConfig, QueryTimeConfig()))
    for holder, defaults in halves:
        for f in fields(holder):
            out[f.name] = frozenset(sf.name for sf in fields(getattr(defaults, f.name)))
    return out


_SECTION_FIELDS: dict[str, frozenset[str]] = _section_fields()
