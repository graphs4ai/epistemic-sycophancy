"""Gradient-based sparse feature selection (Phase F)."""

from epistemic_sycophancy.feature_selection.artifacts import (
    FeatureSelectionArtifact,
    FeatureSelectionRow,
)
from epistemic_sycophancy.feature_selection.components import (
    component_question_subset,
    isolate_component_jacobian,
    logistic_preservation_surrogate,
    selection_component_prompts,
)
from epistemic_sycophancy.feature_selection.exceptions import (
    HoldoutAccessError,
    LayerMismatchError,
    ScopeMismatchError,
)
from epistemic_sycophancy.feature_selection.indexing import (
    assert_layer_tensors_aligned,
    final_prompt_token_index,
    select_final_token_states,
)
from epistemic_sycophancy.feature_selection.pool import (
    EligibilityResult,
    eligible_suppression_candidates,
)
from epistemic_sycophancy.feature_selection.projected_gradient import (
    AttributionScopeResolution,
    StreamingJacobianAccumulator,
    coefficient_jacobian,
    coefficient_jacobian_aggregate_first,
    multi_token_coefficient_jacobian,
    project_residual_gradient,
    question_macro_jacobian,
    question_macro_prompt_weights,
    resolve_attribution_scope,
    sum_coefficient_jacobians,
)
from epistemic_sycophancy.feature_selection.ranking import (
    AnnotatedSuppressionCandidate,
    SuppressionCandidate,
    annotate_preservation_jacobians,
    rank_suppression_candidates,
)

__all__ = [
    "AnnotatedSuppressionCandidate",
    "AttributionScopeResolution",
    "EligibilityResult",
    "FeatureSelectionArtifact",
    "FeatureSelectionRow",
    "HoldoutAccessError",
    "LayerMismatchError",
    "ScopeMismatchError",
    "StreamingJacobianAccumulator",
    "SuppressionCandidate",
    "annotate_preservation_jacobians",
    "assert_layer_tensors_aligned",
    "coefficient_jacobian",
    "coefficient_jacobian_aggregate_first",
    "component_question_subset",
    "eligible_suppression_candidates",
    "final_prompt_token_index",
    "isolate_component_jacobian",
    "logistic_preservation_surrogate",
    "multi_token_coefficient_jacobian",
    "project_residual_gradient",
    "question_macro_jacobian",
    "question_macro_prompt_weights",
    "rank_suppression_candidates",
    "resolve_attribution_scope",
    "select_final_token_states",
    "selection_component_prompts",
    "sum_coefficient_jacobians",
]
