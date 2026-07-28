"""Dataset records, manifests, and integrity validation."""

from epistemic_sycophancy.data.validation import (
    DataIntegrityError,
    assert_derived_variants_inherit_parent_split,
    assert_normalized_question_hash_does_not_cross_splits,
    assert_question_ids_in_exactly_one_split,
    normalize_question_text,
    question_content_hash,
)

__all__ = [
    "DataIntegrityError",
    "assert_derived_variants_inherit_parent_split",
    "assert_normalized_question_hash_does_not_cross_splits",
    "assert_question_ids_in_exactly_one_split",
    "normalize_question_text",
    "question_content_hash",
]
