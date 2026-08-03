"""One-off runner: drives every row in gold.jsonl through the real pipeline and
writes the raw input/output pairs to results.json.

Not `fc-eval` -- that entry point is declared in pyproject.toml but its module
(factcheck.evaluation.harness) was never written. This script is scoped to what
was actually asked for here: run the gold claims for real and capture what came
back, so a report can be built from it. See eval/test_report.md for the write-up
of what it found, including that gap.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factcheck.config import Config  # noqa: E402
from factcheck.errors import ModelCallError  # noqa: E402
from factcheck.evaluation.dataset import load_gold  # noqa: E402
from factcheck.factory import build_session  # noqa: E402

ROOT = Path(__file__).parent.parent

# The free-tier Gemini key this project uses caps gemini-flash-latest at 5
# requests/minute, and one claim spends 3-4 calls (expand, verify, one audit per
# surviving quote). Neither GeminiClient nor the CLI retries on 429 -- that's a
# real gap, noted in the report -- so this one-off runner paces itself instead of
# bursting into a wall of ModelCallError.
CLAIM_SPACING_S = 45
MAX_RETRIES = 3
RETRY_BACKOFF_S = 20


def category_of(note: str) -> str:
    for part in note.split(";"):
        part = part.strip()
        if part.startswith("category="):
            return part[len("category=") :]
    return "uncategorized"


def main() -> None:
    gold = load_gold(ROOT / "eval" / "gold.jsonl")
    cfg = Config.load(ROOT / "config" / "default.yaml")
    session = build_session(
        cfg, ROOT / "index", ROOT / "data", cache_dir=ROOT / "eval" / ".cache"
    )

    rows = []
    for i, g in enumerate(gold):
        result = None
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = session.checker.check(g.claim)
                break
            except ModelCallError as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    print(f"  [{g.claim_id}] attempt {attempt} failed ({exc}); "
                          f"retrying in {RETRY_BACKOFF_S}s")
                    time.sleep(RETRY_BACKOFF_S)
        if result is None:
            print(f"[ERROR] {g.claim_id:<16} gave up after {MAX_RETRIES} attempts: {last_error}")
            rows.append({
                "claim_id": g.claim_id,
                "category": category_of(g.note),
                "input": g.claim,
                "expected_label": g.label.value,
                "expected_doc_id": g.expected_doc_id,
                "actual_verdict": "error",
                "match": False,
                "error": str(last_error),
                "note": g.note,
            })
            if i < len(gold) - 1:
                time.sleep(CLAIM_SPACING_S)
            continue

        actual = result.verdict.value
        expected = g.label.value
        match = actual == expected
        row = {
            "claim_id": g.claim_id,
            "category": category_of(g.note),
            "input": g.claim,
            "expected_label": expected,
            "expected_doc_id": g.expected_doc_id,
            "actual_verdict": actual,
            "match": match,
            "flags": [f.value for f in result.flags],
            "retrieved": list(result.retrieved),
            "evidence": [
                {"doc_id": e.doc_id, "quote": e.quote, "stance": e.stance.value}
                for e in result.evidence
            ],
            "stats": {
                "passages_retrieved": result.stats.passages_retrieved,
                "quotes_proposed": result.stats.quotes_proposed,
                "quotes_accepted": result.stats.quotes_accepted,
                "quotes_rejected": result.stats.quotes_rejected,
                "llm_calls": result.stats.llm_calls,
                "elapsed_s": round(result.stats.elapsed_s, 2),
            },
            "note": g.note,
        }
        rows.append(row)
        flag = "OK" if match else "MISMATCH"
        print(f"[{flag}] {g.claim_id:<16} expected={expected:<12} actual={actual:<12} "
              f"({result.stats.elapsed_s:.1f}s)")
        if i < len(gold) - 1:
            time.sleep(CLAIM_SPACING_S)

    out_path = ROOT / "eval" / "results.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
