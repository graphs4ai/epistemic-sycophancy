"""Prompt-specific final-token indexing for feature selection (Phase F)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from epistemic_sycophancy.feature_selection import (
    final_prompt_token_index,
    select_final_token_states,
)

_TOY_GRADIENTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "feature_selection"
    / "toy_gradients.py"
)
_spec = importlib.util.spec_from_file_location(
    "toy_gradients_indexing", _TOY_GRADIENTS_PATH
)
assert _spec is not None and _spec.loader is not None
_toy_gradients = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_toy_gradients)
final_token_contents = _toy_gradients.final_token_contents
left_padded_batch = _toy_gradients.left_padded_batch
right_padded_batch = _toy_gradients.right_padded_batch


@pytest.mark.unit
def test_feature_selection__t_star__uses_last_nonpadding_token_of_each_rendered_prompt() -> (
    None
):
    """FEAT-001: t* is each prompt's own last non-padding position."""
    right_residual, right_mask = right_padded_batch()
    left_residual, left_mask = left_padded_batch()

    right_index = final_prompt_token_index(attention_mask=right_mask)
    left_index = final_prompt_token_index(attention_mask=left_mask)

    # Prompt lengths (3, 5, 2) in a width-5 batch.
    assert right_index.tolist() == [2, 4, 1]
    assert left_index.tolist() == [4, 4, 4]

    expected_states = final_token_contents()
    right_states = select_final_token_states(
        residual=right_residual, attention_mask=right_mask
    )
    left_states = select_final_token_states(
        residual=left_residual, attention_mask=left_mask
    )

    assert torch.equal(right_states, expected_states)
    assert torch.equal(left_states, expected_states)
