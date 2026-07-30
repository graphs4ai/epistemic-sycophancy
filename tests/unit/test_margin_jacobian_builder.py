"""GRAD-003: selected-pool margin Jacobian via projected coefficient_jacobian."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
def test_margin_jacobian__selected_pool__matches_projected_coefficient_jacobian() -> None:
    """GRAD-003: ∂M/∂β_selected = index(s ⊙ 1[z>0] ⊙ h); reuse FEAT-004 golden.

    Hand-derived FEAT-004: h=[2,-3,1], z=[0.5,0,2], s=[2,4,0.5] → J_full=[4,0,0.5].
    Selected indices (0, 2) → length-m row [4.0, 0.5].
    """
    from epistemic_sycophancy.runner.adapters.margin_jacobian import (
        project_selected_margin_jacobian,
    )

    import importlib.util
    from pathlib import Path

    toy_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "feature_selection"
        / "toy_gradients.py"
    )
    spec = importlib.util.spec_from_file_location("toy_gradients_grad003", toy_path)
    assert spec is not None and spec.loader is not None
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)

    jac_row = project_selected_margin_jacobian(
        residual_gradient=toy.spec_gradient(),
        latents=toy.spec_latents(),
        decoder=toy.spec_decoder(),
        feature_scales=toy.spec_scales(),
        selected_indices=(0, 2),
    )
    assert jac_row.dtype == torch.float64
    assert jac_row.shape == (2,)
    assert jac_row.tolist() == [4.0, 0.5]


@pytest.mark.unit
def test_margin_jacobian__assemble_maps__aligns_ib_cb_neutral_shapes() -> None:
    """GRAD-003: assemble per-prompt selected rows into evaluate_objective_with_grad maps."""
    from epistemic_sycophancy.runner.adapters.margin_jacobian import (
        assemble_margin_jacobian_maps,
    )

    row_n_q1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    row_ib_q1 = torch.tensor([0.5, 0.5], dtype=torch.float64)
    row_ib_q1_b2 = torch.tensor([0.25, -0.25], dtype=torch.float64)
    row_cb_q2 = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    row_n_q2 = torch.tensor([0.1, 0.2], dtype=torch.float64)

    maps = assemble_margin_jacobian_maps(
        prompts=(
            {"question_id": "q1", "belief_condition": "N", "jac_row": row_n_q1},
            {"question_id": "q1", "belief_condition": "IB", "jac_row": row_ib_q1},
            {"question_id": "q1", "belief_condition": "IB", "jac_row": row_ib_q1_b2},
            {"question_id": "q2", "belief_condition": "CB", "jac_row": row_cb_q2},
            {"question_id": "q2", "belief_condition": "N", "jac_row": row_n_q2},
        )
    )
    assert set(maps.keys()) == {
        "ib_margin_jac",
        "cb_margin_jac",
        "neutral_margin_jac",
    }
    assert list(maps["ib_margin_jac"]["q1"]) == [row_ib_q1, row_ib_q1_b2]
    assert list(maps["cb_margin_jac"]["q2"]) == [row_cb_q2]
    assert torch.equal(maps["neutral_margin_jac"]["q1"], row_n_q1)
    assert torch.equal(maps["neutral_margin_jac"]["q2"], row_n_q2)
