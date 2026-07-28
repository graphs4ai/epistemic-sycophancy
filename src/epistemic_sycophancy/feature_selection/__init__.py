"""Gradient-based sparse feature selection (Phase F)."""

from epistemic_sycophancy.feature_selection.indexing import (
    final_prompt_token_index,
    select_final_token_states,
)

__all__ = [
    "final_prompt_token_index",
    "select_final_token_states",
]
