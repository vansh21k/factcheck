"""End-to-end smoke tests: build a real index on disk, then drive the CLI over it.

Deliberately small. These cover the wiring between the three programs -- the seams
no unit test sees -- and the startup refusals, which are the whole reason the index
is a separate program. The model is scripted throughout: no network, no API key.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from conftest import CORPUS, RAG_BENCH, ScriptedLLM
from factcheck.cli import verify as verify_cli
from factcheck.config import Config
from factcheck.errors import EmbeddingModelMismatchError, MissingIndexError
from factcheck.factory import build_session
from factcheck.indexing.build import build_index
from factcheck.ingest.store import JsonlDocumentStore

GROUNDED_QUOTE = (
    "Retrieval-Augmented Generation (RAG) is a promising approach for mitigating the "
    "hallucination of large language models"
)
CLAIM = "RAG is a promising approach for mitigating hallucination in LLMs"


@pytest.fixture
def built(tmp_path: Path) -> tuple[Path, Path]:
    """A real corpus on disk with a real index built over it."""
    docs, index = tmp_path / "data", tmp_path / "index"
    JsonlDocumentStore.save(CORPUS, docs)
    build_index(CORPUS, Config(), index)
    return docs, index


def scripted(*, expand: bool = True) -> ScriptedLLM:
    """One expander call, one verifier call, one audit call."""
    responses: list[Any] = []
    if expand:
        responses.append(
            {
                "negation": "RAG does not reduce hallucination in LLMs",
                "subclaims": ["retrieval-augmented generation and hallucination"],
            }
        )
    responses.append(
        {
            "verdict": "supported",
            "evidence": [
                {"doc_id": RAG_BENCH.doc_id, "quote": GROUNDED_QUOTE, "stance": "entails"}
            ],
        }
    )
    responses.append({"stance": "entails"})
    return ScriptedLLM(responses)


def test_one_shot_verifies_a_claim_against_a_real_index(built: tuple[Path, Path]) -> None:
    docs, index = built
    out = io.StringIO()

    code = verify_cli.main(
        ["--index", str(index), "--docs", str(docs), "--claim", CLAIM],
        llm=scripted(),
        stdout=out,
    )

    assert code == 0
    text = out.getvalue()
    assert "SUPPORTED" in text
    assert RAG_BENCH.doc_id in text
    assert RAG_BENCH.title in text, "evidence is rendered with its source title"


def test_json_mode_is_machine_readable(built: tuple[Path, Path]) -> None:
    docs, index = built
    out = io.StringIO()

    verify_cli.main(
        ["--index", str(index), "--docs", str(docs), "--claim", CLAIM, "--json"],
        llm=scripted(),
        stdout=out,
    )

    payload = json.loads(out.getvalue().strip())
    assert payload["verdict"] == "supported"
    assert payload["evidence"][0]["doc_id"] == RAG_BENCH.doc_id
    assert payload["retrieved"], "the retrieved set powers recall@k and must be present"


def test_pointing_at_a_directory_with_no_index_names_the_index_command(tmp_path: Path) -> None:
    """Never a convenience rebuild -- at scale that is the entire cost."""
    out = io.StringIO()

    code = verify_cli.main(
        ["--index", str(tmp_path / "nope"), "--docs", str(tmp_path), "--claim", "x"],
        llm=ScriptedLLM([]),
        stdout=out,
    )

    assert code == 1
    assert "fc-index" in out.getvalue()


def test_embedding_model_mismatch_refuses_to_start(built: tuple[Path, Path]) -> None:
    """A different model at the same dimension returns confident nonsense in
    plausible rank order, so identity is checked and refusal is at startup."""
    docs, index = built
    cfg = Config().set("embedding.model", "some-other-model", allow_index_time=True)

    with pytest.raises(EmbeddingModelMismatchError):
        build_session(cfg, index, docs, llm=ScriptedLLM([]))


def test_missing_index_raises_rather_than_building_one(tmp_path: Path) -> None:
    with pytest.raises(MissingIndexError):
        build_session(Config(), tmp_path / "absent", tmp_path, llm=ScriptedLLM([]))


def test_repl_set_refuses_an_index_time_knob_and_names_the_rebuild(
    built: tuple[Path, Path]
) -> None:
    """The boundary is enforced where the mistake is actually made."""
    docs, index = built
    out = io.StringIO()
    stdin = io.StringIO(":set chunk.size 256\n:quit\n")

    verify_cli.main(
        ["--index", str(index), "--docs", str(docs)],
        llm=ScriptedLLM([]),
        stdin=stdin,
        stdout=out,
    )

    text = out.getvalue()
    assert "index-time" in text
    assert "--rebuild" in text


def test_repl_banner_reports_what_is_being_queried(built: tuple[Path, Path]) -> None:
    docs, index = built
    out = io.StringIO()

    verify_cli.main(
        ["--index", str(index), "--docs", str(docs)],
        llm=ScriptedLLM([]),
        stdin=io.StringIO(":quit\n"),
        stdout=out,
    )

    text = out.getvalue()
    assert f"{len(CORPUS)} documents" in text
    assert "config hash" in text


def test_why_explains_the_last_verdict(built: tuple[Path, Path]) -> None:
    docs, index = built
    out = io.StringIO()

    verify_cli.main(
        ["--index", str(index), "--docs", str(docs)],
        llm=scripted(),
        stdin=io.StringIO(f"{CLAIM}\n:why\n:quit\n"),
        stdout=out,
    )

    text = out.getvalue()
    assert "fused ranks" in text
    assert "quote audit" in text
    assert "rejected spans" in text
    # The negation query must actually reach retrieval: without it, refuting
    # passages are never retrieved and `contradicted` is structurally unreachable.
    assert "[negation]" in text
    assert "RAG does not reduce hallucination" in text


def test_batch_mode_drives_the_same_path(built: tuple[Path, Path], tmp_path: Path) -> None:
    docs, index = built
    claims = tmp_path / "claims.txt"
    claims.write_text(f"{CLAIM}\n", encoding="utf-8")
    out = io.StringIO()

    code = verify_cli.main(
        ["--index", str(index), "--docs", str(docs), "--claims-file", str(claims), "--json"],
        llm=scripted(),
        stdout=out,
    )

    assert code == 0
    assert json.loads(out.getvalue().strip())["verdict"] == "supported"
