"""Dataset integrity validation."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence


class DataIntegrityError(Exception):
    """Raised when a dataset record or manifest violates a required invariant."""


def normalize_question_text(text: str) -> str:
    """Identity-normalize question text for content-hash leakage checks (DEC-004)."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[.!?]+$", "", normalized)
    return normalized.casefold()


def question_content_hash(text: str) -> str:
    """SHA-256 hex digest of UTF-8 identity-normalized question text (DEC-004)."""
    return hashlib.sha256(normalize_question_text(text).encode("utf-8")).hexdigest()


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


def assert_derived_variants_inherit_parent_split(
    parents: Sequence[Mapping[str, object]],
    derived: Sequence[Mapping[str, object]],
) -> None:
    """Assert each derived row inherits its parent question_id split.

    Invariant (DATA-003): every derived row's ``split`` equals the split
    assigned to its parent ``question_id``.
    """
    assert_question_ids_in_exactly_one_split(parents)
    parent_split = {row["question_id"]: row["split"] for row in parents}

    mismatches: list[dict[str, object]] = []
    for row in derived:
        question_id = row["question_id"]
        if question_id not in parent_split:
            mismatches.append(
                {
                    "question_id": question_id,
                    "derived_split": row["split"],
                    "parent_split": None,
                }
            )
            continue
        expected = parent_split[question_id]
        if row["split"] != expected:
            mismatches.append(
                {
                    "question_id": question_id,
                    "derived_split": row["split"],
                    "parent_split": expected,
                }
            )

    if mismatches:
        raise DataIntegrityError(
            "derived rows must inherit parent question_id split; "
            f"found mismatches: {mismatches!r}"
        )


def assert_normalized_question_hash_does_not_cross_splits(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Assert normalized question content hashes do not cross splits.

    Invariant (DATA-004 / DEC-004): the same content hash must not appear under
    more than one split.
    """
    splits_by_hash: dict[str, set[object]] = defaultdict(set)
    for row in rows:
        content_hash = question_content_hash(str(row["question_text"]))
        splits_by_hash[content_hash].add(row["split"])

    leaked = {
        content_hash: sorted(splits)
        for content_hash, splits in splits_by_hash.items()
        if len(splits) != 1
    }
    if leaked:
        raise DataIntegrityError(
            "normalized question content hash must not cross splits; "
            f"found cross-split hashes: {leaked!r}"
        )


def assert_mc_targets_are_complete_and_noncontradictory(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Assert MC targets are complete and noncontradictory per format.

    Invariant (DATA-007): for each row, truthful and incorrect targets exist and
    differ; MC0 has exactly one of each; MC1 has exactly one official truthful;
    MC2 has at least one truthful and at least one false candidate.
    """
    violations: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        format_name = str(row["format"]).upper()
        targets = row.get("targets")
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            violations.append(
                {
                    "index": index,
                    "question_id": row.get("question_id"),
                    "format": format_name,
                    "reason": "targets must be a sequence",
                }
            )
            continue

        truthful = [t for t in targets if isinstance(t, Mapping) and t.get("label") == 1]
        incorrect = [t for t in targets if isinstance(t, Mapping) and t.get("label") == 0]

        if not truthful or not incorrect:
            violations.append(
                {
                    "index": index,
                    "question_id": row.get("question_id"),
                    "format": format_name,
                    "reason": "truthful and incorrect targets must both exist",
                    "n_truthful": len(truthful),
                    "n_incorrect": len(incorrect),
                }
            )
            continue

        truthful_ids = {t.get("answer_id") for t in truthful}
        incorrect_ids = {t.get("answer_id") for t in incorrect}
        truthful_texts = {t.get("text") for t in truthful}
        incorrect_texts = {t.get("text") for t in incorrect}
        if truthful_ids & incorrect_ids or truthful_texts & incorrect_texts:
            violations.append(
                {
                    "index": index,
                    "question_id": row.get("question_id"),
                    "format": format_name,
                    "reason": "truthful and incorrect targets must differ",
                }
            )
            continue

        if format_name == "MC0":
            if len(truthful) != 1 or len(incorrect) != 1:
                violations.append(
                    {
                        "index": index,
                        "question_id": row.get("question_id"),
                        "format": format_name,
                        "reason": (
                            "MC0 requires exactly one truthful and one selected "
                            "incorrect answer"
                        ),
                        "n_truthful": len(truthful),
                        "n_incorrect": len(incorrect),
                    }
                )
        elif format_name == "MC1":
            if len(truthful) != 1:
                violations.append(
                    {
                        "index": index,
                        "question_id": row.get("question_id"),
                        "format": format_name,
                        "reason": "MC1 requires exactly one official truthful target",
                        "n_truthful": len(truthful),
                    }
                )
        elif format_name == "MC2":
            if len(truthful) < 1 or len(incorrect) < 1:
                violations.append(
                    {
                        "index": index,
                        "question_id": row.get("question_id"),
                        "format": format_name,
                        "reason": (
                            "MC2 requires at least one truthful and at least one "
                            "false candidate"
                        ),
                        "n_truthful": len(truthful),
                        "n_incorrect": len(incorrect),
                    }
                )
        else:
            violations.append(
                {
                    "index": index,
                    "question_id": row.get("question_id"),
                    "format": format_name,
                    "reason": f"unsupported format {format_name!r}",
                }
            )

    if violations:
        raise DataIntegrityError(
            "MC targets must be complete and noncontradictory; "
            f"found violations: {violations!r}"
        )
