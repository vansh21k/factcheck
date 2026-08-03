"""Configuration is a value, and the index-time boundary is a type-level fact."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from factcheck.config import Config, is_index_time
from factcheck.errors import ConfigError, IndexTimeConfigError


def test_config_is_immutable_for_the_duration_of_a_run() -> None:
    """A result always corresponds to exactly one configuration."""
    cfg = Config()
    with pytest.raises(FrozenInstanceError):
        cfg.query.rerank.top_n = 12  # type: ignore[misc]


def test_setting_a_value_returns_a_new_config_and_leaves_the_original_alone() -> None:
    cfg = Config()
    updated = cfg.set("rerank.top_n", 12)

    assert updated.get("rerank.top_n") == 12
    assert cfg.get("rerank.top_n") == 8
    assert updated.hash != cfg.hash


def test_hash_is_stable_across_construction_paths() -> None:
    """A number in an eval table has to be traceable to what produced it."""
    a = Config().set("rerank.top_n", 12).set("dense.min_score", 0.3)
    b = Config().set("dense.min_score", 0.3).set("rerank.top_n", 12)

    assert a.hash == b.hash
    assert Config.from_dict(a.to_dict()).hash == a.hash


def test_index_time_change_against_a_prebuilt_index_is_refused_by_name() -> None:
    """Sweeping chunk.size against a prebuilt index does not crash -- it produces a
    plausible number as unrelated noise moves the metric. So it is an error."""
    with pytest.raises(IndexTimeConfigError) as excinfo:
        Config().set("chunk.size", 256)

    assert "fc-index --rebuild" in str(excinfo.value)


def test_index_time_change_is_allowed_when_a_rebuild_is_explicit() -> None:
    cfg = Config().set("chunk.size", 256, allow_index_time=True)

    assert cfg.get("chunk.size") == 256
    assert cfg.index_fingerprint()["chunk"]["size"] == 256


def test_query_time_changes_need_no_permission() -> None:
    for path, value in [
        ("expander.enabled", False),
        ("auditor.enabled", False),
        ("validator.min_quote_chars", 20),
        ("aggregation.policy", "majority_stance"),
    ]:
        assert Config().set(path, value).get(path) == value


@pytest.mark.parametrize(
    "path,expected",
    [
        ("chunk.size", True),
        ("chunk.overlap", True),
        ("embedding.model", True),
        ("embedding.dim", True),
        ("rerank.top_n", False),
        ("verifier.prompt_version", False),
    ],
)
def test_the_split_is_readable_off_the_config_itself(path: str, expected: bool) -> None:
    """So I know what requires a rebuild without consulting a document."""
    assert is_index_time(path) is expected


def test_there_is_no_knob_that_relaxes_the_substring_requirement() -> None:
    """The most natural knob in the world to add, and the one that converts the
    grounding guarantee back into a tuning parameter."""
    with pytest.raises(ConfigError):
        Config().set("validator.fuzzy_match_threshold", 0.85)


def test_unknown_sections_and_fields_are_errors_not_silent_no_ops() -> None:
    for path in ["retriever.top_k", "validator.mimimum_quote_chars", "rerank"]:
        with pytest.raises(ConfigError):
            Config().set(path, 1)


def test_config_loads_from_file_and_is_overridable_per_invocation(tmp_path: Path) -> None:
    """Experiments must not require code edits."""
    path = tmp_path / "sweep.yaml"
    path.write_text(
        "index:\n  chunk:\n    size: 256\nquery:\n  rerank:\n    top_n: 4\n"
        "  auditor:\n    enabled: false\n",
        encoding="utf-8",
    )

    cfg = Config.load(path, overrides={"rerank.top_n": "16"})

    assert cfg.get("chunk.size") == 256
    assert cfg.get("auditor.enabled") is False
    assert cfg.get("rerank.top_n") == 16, "CLI overrides beat the file"


def test_string_overrides_are_coerced_to_the_type_they_replace() -> None:
    """CLI flags arrive as strings; a silently-string threshold would compare wrongly."""
    cfg = Config().with_overrides(
        {"rerank.top_n": "12", "dense.min_score": "0.4", "auditor.enabled": "false"}
    )

    assert cfg.get("rerank.top_n") == 12
    assert cfg.get("dense.min_score") == pytest.approx(0.4)
    assert cfg.get("auditor.enabled") is False


def test_an_uncoercible_override_fails_loudly() -> None:
    with pytest.raises(ConfigError):
        Config().set("rerank.top_n", "twelve")
    with pytest.raises(ConfigError):
        Config().set("auditor.enabled", "maybe")


def test_index_fingerprint_carries_only_index_time_fields() -> None:
    """It is what goes in the manifest, so query-time noise must not enter it."""
    fingerprint = Config().index_fingerprint()

    assert set(fingerprint) == {"chunk", "embedding"}
    assert Config().set("rerank.top_n", 99).index_fingerprint() == fingerprint
