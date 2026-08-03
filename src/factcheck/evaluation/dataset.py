"""The labeled claim set: the only ground truth this system has.

Thirty hand-written claims cannot resolve a three-point accuracy difference, so the
set is small by necessity, not by laziness -- which is exactly why a bad row in it
must fail loudly at load time rather than be silently dropped or silently miscounted.
Every field is validated on parse; an unknown verdict label or split is an error, not
a row that quietly disappears from the denominator.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import Verdict

_SPLITS = ("dev", "test")


@dataclass(frozen=True)
class GoldClaim:
    """One labeled row.

    ``expected_doc_id`` is optional: some *unknown* claims are unknown precisely
    because no single document should have been cited, so pinning one would
    misrepresent the label rather than sharpen it.
    """

    claim_id: str
    claim: str
    label: Verdict
    expected_doc_id: str | None
    split: str
    note: str = ""


def load_gold(path: str | Path) -> list[GoldClaim]:
    """Parse the JSONL gold set, one object per non-blank line.

    Malformed rows raise rather than get skipped. A silently shrunk eval set is worse
    than a crash: the crash is noticed, the shrinkage is a metric quietly computed
    over fewer claims than anyone believes it was.
    """
    claims: list[GoldClaim] = []
    seen_ids: set[str] = set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        claims.append(_parse_line(stripped, path, lineno, seen_ids))
    return claims


def split_of(claims: Sequence[GoldClaim], split: str) -> list[GoldClaim]:
    """Filter to one split. ``'all'`` is the escape hatch for a full-set report."""
    if split == "all":
        return list(claims)
    if split not in _SPLITS:
        raise ValueError(f"unknown split '{split}'. Known: {', '.join((*_SPLITS, 'all'))}")
    return [c for c in claims if c.split == split]


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def _parse_line(line: str, path: str | Path, lineno: int, seen_ids: set[str]) -> GoldClaim:
    try:
        row: Any = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"{path}:{lineno}: expected a JSON object, got {type(row).__name__}")

    claim_id = _require_str(row, "claim_id", path, lineno)
    if claim_id in seen_ids:
        raise ValueError(f"{path}:{lineno}: duplicate claim_id '{claim_id}'")
    seen_ids.add(claim_id)

    claim = _require_str(row, "claim", path, lineno)

    split = _require_str(row, "split", path, lineno)
    if split not in _SPLITS:
        raise ValueError(
            f"{path}:{lineno}: unknown split '{split}'. Known splits: {', '.join(_SPLITS)}"
        )

    label_raw = _require_str(row, "label", path, lineno)
    try:
        label = Verdict(label_raw)
    except ValueError:
        known = ", ".join(v.value for v in Verdict)
        raise ValueError(
            f"{path}:{lineno}: unknown verdict label '{label_raw}'. Known: {known}"
        ) from None

    expected_doc_id = row.get("expected_doc_id")
    if expected_doc_id is not None and not isinstance(expected_doc_id, str):
        raise ValueError(f"{path}:{lineno}: 'expected_doc_id' must be a string or null")

    note = row.get("note", "")
    if not isinstance(note, str):
        raise ValueError(f"{path}:{lineno}: 'note' must be a string")

    return GoldClaim(
        claim_id=claim_id,
        claim=claim,
        label=label,
        expected_doc_id=expected_doc_id,
        split=split,
        note=note,
    )


def _require_str(row: dict[str, Any], key: str, path: str | Path, lineno: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{lineno}: '{key}' must be a non-empty string")
    return value
