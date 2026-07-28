"""Dataset manifest loading tests (Phase A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from epistemic_sycophancy.data import (
    ManifestMismatchError,
    count_questions_by_split,
    load_split_manifest,
    validate_dataset_manifest,
)

FIXTURE_SPLIT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "fixtures" / "data" / "split_manifest.csv"
)

# DATA-001: regression counts for the current pinned dataset version.
EXPECTED_SPLIT_COUNTS = {
    "feature_selection": 316,
    "optimization": 237,
    "behavior_validation": 118,
    "holdout_test_behavior": 119,
}
EXPECTED_TOTAL = 790


@pytest.mark.unit
def test_dataset__current_manifest__contains_expected_question_counts() -> None:
    """DATA-001: pinned split manifest matches expected question counts."""
    rows = load_split_manifest(FIXTURE_SPLIT_MANIFEST)
    counts = count_questions_by_split(rows)

    assert counts == EXPECTED_SPLIT_COUNTS
    assert sum(counts.values()) == EXPECTED_TOTAL


def _valid_dataset_manifest(**overrides: object) -> dict[str, object]:
    """Minimal valid reproducibility manifest (DEC-008)."""
    manifest: dict[str, object] = {
        "source_file_hashes": {
            "truthfulqa_csv": "a" * 64,
            "mc_task_json": "b" * 64,
        },
        "preprocessing_version": "0.1.0",
        "split_seed": 42,
        "ro_seed": 42,
        "belief_generation_provenance": "belief_triples_v1",
        "prompt_template_version": "v1",
    }
    manifest.update(overrides)
    return manifest


@pytest.mark.unit
def test_dataset__manifest__records_hashes_and_seeds() -> None:
    """DATA-009: dataset manifest records required hashes, seeds, and versions."""
    validate_dataset_manifest(_valid_dataset_manifest())

    required_keys = (
        "source_file_hashes",
        "preprocessing_version",
        "split_seed",
        "ro_seed",
        "belief_generation_provenance",
        "prompt_template_version",
    )
    for key in required_keys:
        incomplete = _valid_dataset_manifest()
        del incomplete[key]
        with pytest.raises(ManifestMismatchError):
            validate_dataset_manifest(incomplete)

        with_none = _valid_dataset_manifest(**{key: None})
        with pytest.raises(ManifestMismatchError):
            validate_dataset_manifest(with_none)

    with pytest.raises(ManifestMismatchError):
        validate_dataset_manifest(_valid_dataset_manifest(source_file_hashes={}))

    with pytest.raises(ManifestMismatchError):
        validate_dataset_manifest(_valid_dataset_manifest(preprocessing_version=""))

    with pytest.raises(ManifestMismatchError):
        validate_dataset_manifest(_valid_dataset_manifest(prompt_template_version=""))

    with pytest.raises(ManifestMismatchError):
        validate_dataset_manifest(
            _valid_dataset_manifest(belief_generation_provenance="")
        )
