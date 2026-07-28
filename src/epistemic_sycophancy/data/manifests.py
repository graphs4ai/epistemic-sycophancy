"""Dataset manifest loading and counting."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


class ManifestMismatchError(Exception):
    """Raised when a dataset manifest is missing required reproducibility metadata."""


def load_split_manifest(path: Path) -> list[dict[str, str]]:
    """Load parent split-assignment rows from a CSV manifest.

    Requires at least ``question_id`` and ``split`` columns.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"split manifest has no header: {path}")
        required = {"question_id", "split"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"split manifest missing required columns {sorted(missing)}: {path}"
            )
        return [
            {"question_id": row["question_id"], "split": row["split"]}
            for row in reader
        ]


def count_questions_by_split(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Count question rows per split label.

    Each row is treated as one parent-question assignment. Counts are returned
    as a plain ``dict`` keyed by split name.
    """
    counts = Counter(str(row["split"]) for row in rows)
    return dict(counts)


def validate_dataset_manifest(manifest: Mapping[str, object]) -> None:
    """Validate required dataset reproducibility metadata.

    Invariant (DATA-009 / DEC-008): required hashes, seeds, and version /
    provenance fields are present and non-null with the typed constraints below.
    """
    required_keys = (
        "source_file_hashes",
        "preprocessing_version",
        "split_seed",
        "ro_seed",
        "belief_generation_provenance",
        "prompt_template_version",
    )
    missing = [key for key in required_keys if key not in manifest]
    if missing:
        raise ManifestMismatchError(
            f"dataset manifest missing required keys: {missing!r}"
        )

    null_keys = [key for key in required_keys if manifest[key] is None]
    if null_keys:
        raise ManifestMismatchError(
            f"dataset manifest required keys must not be None: {null_keys!r}"
        )

    source_file_hashes = manifest["source_file_hashes"]
    if not isinstance(source_file_hashes, Mapping) or len(source_file_hashes) == 0:
        raise ManifestMismatchError(
            "source_file_hashes must be a non-empty mapping of source label to "
            f"SHA-256 hex; got {source_file_hashes!r}"
        )

    for key in (
        "preprocessing_version",
        "belief_generation_provenance",
        "prompt_template_version",
    ):
        value = manifest[key]
        if not isinstance(value, str) or value == "":
            raise ManifestMismatchError(
                f"{key} must be a non-empty string; got {value!r}"
            )

    for key in ("split_seed", "ro_seed"):
        value = manifest[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ManifestMismatchError(
                f"{key} must be an int; got {value!r}"
            )
