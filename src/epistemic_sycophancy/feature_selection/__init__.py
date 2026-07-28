"""Gradient-based sparse feature selection (Phase F)."""

from epistemic_sycophancy.feature_selection.components import (
    component_question_subset,
    logistic_preservation_surrogate,
    selection_component_prompts,
)
from epistemic_sycophancy.feature_selection.exceptions import LayerMismatchError
from epistemic_sycophancy.feature_selection.indexing import (
    assert_layer_tensors_aligned,
    final_prompt_token_index,
    select_final_token_states,
)
from epistemic_sycophancy.feature_selection.projected_gradient import (
    StreamingJacobianAccumulator,
    coefficient_jacobian,
    coefficient_jacobian_aggregate_first,
    project_residual_gradient,
    question_macro_jacobian,
    question_macro_prompt_weights,
    sum_coefficient_jacobians,
)
from epistemic_sycophancy.feature_selection.ranking import (
    SuppressionCandidate,
    rank_suppression_candidates,
)

__all__ = [
    "LayerMismatchError",
    "StreamingJacobianAccumulator",
    "SuppressionCandidate",
    "assert_layer_tensors_aligned",
    "coefficient_jacobian",
    "coefficient_jacobian_aggregate_first",
    "component_question_subset",
    "final_prompt_token_index",
    "logistic_preservation_surrogate",
    "project_residual_gradient",
    "question_macro_jacobian",
    "question_macro_prompt_weights",
    "rank_suppression_candidates",
    "select_final_token_states",
    "selection_component_prompts",
    "sum_coefficient_jacobians",
]
