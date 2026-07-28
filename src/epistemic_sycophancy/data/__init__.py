"""Dataset records, manifests, and integrity validation."""

from epistemic_sycophancy.data.manifests import (
    ManifestMismatchError,
    count_questions_by_split,
    load_split_manifest,
    validate_dataset_manifest,
)
from epistemic_sycophancy.data.validation import (
    DataIntegrityError,
    assert_belief_variant_ids_are_unique_within_question_and_condition,
    assert_derived_variants_inherit_parent_split,
    assert_mc_targets_are_complete_and_noncontradictory,
    assert_neutral_rows_exactly_one_per_question_order_and_format,
    assert_normalized_question_hash_does_not_cross_splits,
    assert_question_ids_in_exactly_one_split,
    assert_question_macro_weights_sum_to_one_within_component,
    normalize_question_text,
    question_content_hash,
)

__all__ = [
    "DataIntegrityError",
    "ManifestMismatchError",
    "assert_belief_variant_ids_are_unique_within_question_and_condition",
    "assert_derived_variants_inherit_parent_split",
    "assert_mc_targets_are_complete_and_noncontradictory",
    "assert_neutral_rows_exactly_one_per_question_order_and_format",
    "assert_normalized_question_hash_does_not_cross_splits",
    "assert_question_ids_in_exactly_one_split",
    "assert_question_macro_weights_sum_to_one_within_component",
    "count_questions_by_split",
    "load_split_manifest",
    "normalize_question_text",
    "question_content_hash",
    "validate_dataset_manifest",
]
