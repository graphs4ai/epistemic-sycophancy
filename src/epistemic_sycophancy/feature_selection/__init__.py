"""Gradient-based sparse feature selection (Phase F)."""

from epistemic_sycophancy.feature_selection.components import (
    component_question_subset,
    selection_component_prompts,
)
from epistemic_sycophancy.feature_selection.indexing import (
    final_prompt_token_index,
    select_final_token_states,
)
from epistemic_sycophancy.feature_selection.projected_gradient import (
    coefficient_jacobian,
    project_residual_gradient,
)
from epistemic_sycophancy.feature_selection.ranking import (
    SuppressionCandidate,
    rank_suppression_candidates,
)

__all__ = [
    "SuppressionCandidate",
    "coefficient_jacobian",
    "component_question_subset",
    "final_prompt_token_index",
    "project_residual_gradient",
    "rank_suppression_candidates",
    "select_final_token_states",
    "selection_component_prompts",
]
