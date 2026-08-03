"""Stage-level caching, keyed by stage, input hash, model, and temperature.

Sweeping a knob *downstream* of a stage replays that stage's recorded output, so
sweeping ``aggregation.policy``, ``validator.min_quote_chars`` or ``auditor.enabled``
costs almost no API calls across the whole gold set. Knobs *upstream* --
``expander.*``, ``rerank.top_n`` -- invalidate everything after them and re-pay in
full. Order a sweep from the bottom of the pipeline up and most of it is nearly free;
order it the other way and it is not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..ports import LLMClient


class StageCache:
    """One JSON file per key, under a directory that is safe to delete."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._path(key).write_text(
            json.dumps(value, sort_keys=True, ensure_ascii=False), encoding="utf-8"
        )

    def _path(self, key: str) -> Path:
        return self._root / f"{key}.json"


def cache_key(stage: str, model: str, temperature: float, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(f"{stage}|{model}|{temperature}|{body}".encode()).hexdigest()
    return f"{stage}-{digest[:32]}"


class CachingLLM:
    """Wraps a client so identical requests are paid for once.

    Caching at the client rather than inside each stage means a newly added stage is
    cached automatically, and the key already carries everything that can change an
    answer: the prompt, the model, and the temperature.
    """

    def __init__(self, inner: LLMClient, cache: StageCache, stage: str) -> None:
        self._inner = inner
        self._cache = cache
        self._stage = stage

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        tool: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        key = cache_key(
            self._stage,
            model,
            temperature,
            {"system": system, "user": user, "tool": (tool or {}).get("name", "")},
        )
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = self._inner.complete(
            system=system,
            user=user,
            model=model,
            temperature=temperature,
            tool=tool,
            max_tokens=max_tokens,
        )
        self._cache.put(key, result)
        return result
