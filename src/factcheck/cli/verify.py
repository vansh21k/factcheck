"""``verify`` -- load a prebuilt index, then answer claims.

Interactive by default, because checking several claims should not re-pay startup
cost each time. ``--claim`` and ``--claims-file`` exist so the tool composes into
scripts and so the evaluation harness drives the identical path.

This command never builds an index. Pointing it at a directory without one is an
error naming the index command: at 50 abstracts an implicit rebuild costs two seconds
and hides nothing, at 10^8 it is the entire problem, and a design that papers over it
at small scale has no answer at large scale.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from ..config import Config, is_index_time
from ..errors import (
    ConfigError,
    FactCheckError,
    IndexTimeConfigError,
    MissingAPIKeyError,
    ModelCallError,
)
from ..factory import Session, build_session
from ..ports import LLMClient
from .render import render_result, render_why, result_to_json

PROMPT = "claim> "


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fc-verify", description="Verify claims against a prebuilt arXiv index."
    )
    parser.add_argument("--index", type=Path, default=Path("index"), help="prebuilt index dir")
    parser.add_argument("--docs", type=Path, default=Path("data"), help="fetched documents dir")
    parser.add_argument("--config", type=Path, default=None, help="config YAML")
    parser.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE",
        help="override a config value, e.g. --set rerank.top_n=12",
    )
    parser.add_argument("--claim", default=None, help="verify one claim and exit")
    parser.add_argument("--claims-file", type=Path, default=None, help="verify a claim per line")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--cache", type=Path, default=None, help="stage cache dir")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    llm: LLMClient | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    out = stdout or sys.stdout

    try:
        cfg = Config.load(args.config, overrides=_parse_overrides(args.set))
        session = build_session(cfg, args.index, args.docs, llm=llm, cache_dir=args.cache)
    except MissingAPIKeyError as exc:
        # Diagnosed in seconds at startup rather than three claims into a session.
        print(f"error: {exc}", file=out)
        return 2
    except FactCheckError as exc:
        print(f"error: {exc}", file=out)
        return 1

    if args.claim is not None:
        return _one_shot(session, args.claim, args.json, out)
    if args.claims_file is not None:
        return _batch(session, args.claims_file, args.json, out)
    return _repl(session, stdin or sys.stdin, out)


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #


def _one_shot(session: Session, claim: str, as_json: bool, out: TextIO) -> int:
    try:
        result = session.checker.check(claim)
    except ModelCallError as exc:
        print(f"error: {exc}", file=out)
        return 3
    print(result_to_json(result) if as_json else render_result(result, session.store), file=out)
    return 0


def _batch(session: Session, path: Path, as_json: bool, out: TextIO) -> int:
    claims = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    failures = 0
    for claim in claims:
        try:
            result = session.checker.check(claim)
        except ModelCallError as exc:
            # One bad claim must not abandon the rest of the file; the harness runs
            # thirty of these and a transient failure should cost one row.
            print(f"error: {claim!r}: {exc}", file=out)
            failures += 1
            continue
        if as_json:
            print(result_to_json(result), file=out)
        else:
            print(f"\n{PROMPT}{claim}", file=out)
            print(render_result(result, session.store), file=out)
    return 3 if failures else 0


def _repl(session: Session, stdin: TextIO, out: TextIO) -> int:
    _print_banner(session, out)
    last = None

    while True:
        print(PROMPT, end="", file=out, flush=True)
        line = stdin.readline()
        if not line:
            break
        claim = line.strip()
        if not claim:
            continue
        if claim in (":q", ":quit", ":exit"):
            break
        if claim in (":why", ":explain"):
            print(render_why(last) if last else "  no claim checked yet", file=out)
            continue
        if claim.startswith(":set"):
            session = _apply_set(session, claim, out)
            continue
        if claim.startswith(":"):
            print("  commands: :why  :set PATH VALUE  :quit", file=out)
            continue

        try:
            last = session.checker.check(claim)
        except ModelCallError as exc:
            # Report and keep the session alive: losing a loaded index to a rate
            # limit would make the interactive mode pointless.
            print(f"  error: {exc}", file=out)
            continue
        print(render_result(last, session.store), file=out)

    return 0


# --------------------------------------------------------------------------- #
# session commands
# --------------------------------------------------------------------------- #


def _apply_set(session: Session, command: str, out: TextIO) -> Session:
    """Apply a query-time change live; refuse an index-time one by name.

    The boundary is enforced at the point of use, so the config split is something
    you feel immediately rather than a rule in a document.
    """
    parts = command.split()
    if len(parts) != 3:
        print("  usage: :set PATH VALUE   e.g. :set rerank.top_n 12", file=out)
        return session

    _, path, value = parts
    before = session.config.hash
    try:
        updated = session.config.set(path, value)
    except IndexTimeConfigError as exc:
        print(f"  {exc}", file=out)
        return session
    except ConfigError as exc:
        print(f"  {exc}", file=out)
        return session

    # Rebuilding the session re-wires the pipeline from the new config. It reloads
    # the index from disk, which is cheap here and is the honest thing to measure.
    rebuilt = build_session(
        updated,
        session.index_dir,
        session.docs_dir,
        llm=session.llm,
        cache_dir=session.cache_dir,
    )
    print(f"  query-time knob, no rebuild · config {before} -> {updated.hash}", file=out)
    return rebuilt


def _print_banner(session: Session, out: TextIO) -> None:
    """Always know what you are querying: counts, build time, and config hash."""
    manifest = session.manifest
    print(
        f"\n  {manifest.n_docs} documents · {manifest.n_chunks} chunks · "
        f"built {manifest.built_at}",
        file=out,
    )
    print(f"  config hash {session.config.hash}", file=out)
    for warning in session.warnings:
        print(f"  warning: {warning}", file=out)
    print("  :why explains the last verdict · :set changes a knob · :quit\n", file=out)


def _parse_overrides(pairs: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"--set expects PATH=VALUE, got '{pair}'")
        path, value = pair.split("=", 1)
        # Index-time overrides are rejected later by Config.set unless a rebuild is
        # explicit; catching the shape here keeps the error message about the shape.
        if is_index_time(path.strip()):
            raise IndexTimeConfigError(
                f"'{path.strip()}' is index-time. Run `fc-index --rebuild --set {pair}` instead."
            )
        overrides[path.strip()] = value.strip()
    return overrides


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
