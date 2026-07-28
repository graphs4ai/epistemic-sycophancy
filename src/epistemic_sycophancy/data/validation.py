"""Dataset integrity validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


class DataIntegrityError(Exception):
    """Raised when a dataset record or manifest violates a required invariant."""


def assert_question_ids_in_exactly_one_split(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Assert each question_id appears under exactly one split.

    Invariant (DATA-002): for every ``question_id``, ``n_unique(split) == 1``.
    Multiple rows with the same ``question_id`` in the same split are allowed.
    """
    splits_by_question: dict[object, set[object]] = defaultdict(set)
    for row in rows:
        splits_by_question[row["question_id"]].add(row["split"])

    leaked = {
        question_id: sorted(splits)
        for question_id, splits in splits_by_question.items()
        if len(splits) != 1
    }
    if leaked:
        raise DataIntegrityError(
            "question_id must appear in exactly one split; "
            f"found cross-split membership: {leaked!r}"
        )
