"""Dataset manifest loading and counting."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


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
