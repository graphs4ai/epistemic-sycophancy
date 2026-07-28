"""Exact local coefficient Jacobian (Phase F)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from epistemic_sycophancy.feature_selection import (
    coefficient_jacobian,
    question_macro_jacobian,
)
from epistemic_sycophancy.intervention.sae_delta import apply_selected_latent_update

_TOY_GRADIENTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "feature_selection"
    / "toy_gradients.py"
)
_spec = importlib.util.spec_from_file_location(
    "toy_gradients_jacobian", _TOY_GRADIENTS_PATH
)
assert _spec is not None and _spec.loader is not None
_toy_gradients = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_toy_gradients)
spec_decoder = _toy_gradients.spec_decoder
spec_gradient = _toy_gradients.spec_gradient
spec_latents = _toy_gradients.spec_latents
spec_scales = _toy_gradients.spec_scales


@pytest.mark.unit
def test_feature_jacobian__scale_and_relu_mask__match_chain_rule() -> None:
    """FEAT-004: J_j = s_j * 1[z_j > 0] * h_j.

    Hand-derived from spec §11 with h=[2,-3,1], z=[0.5,0,2], s=[2,4,0.5]:
    J = [2*1*2, 4*0*(-3), 0.5*1*1] = [4.0, 0.0, 0.5].
    """
    jacobian = coefficient_jacobian(
        raw_projection=spec_gradient() @ spec_decoder().T,
        latents=spec_latents(),
        feature_scales=spec_scales(),
    )

    assert jacobian.tolist() == [4.0, 0.0, 0.5]


@pytest.mark.unit
def test_feature_jacobian__inactive_feature__has_zero_feasible_derivative() -> None:
    """FEAT-007: z_j = 0 with beta_j <= 0 keeps the latent and derivative at zero."""
    inactive_latent = torch.tensor([0.0], dtype=torch.float64)
    feature_scale = torch.tensor([3.0], dtype=torch.float64)

    for raw_projection in (-5.0, 5.0):
        jacobian = coefficient_jacobian(
            raw_projection=torch.tensor([raw_projection], dtype=torch.float64),
            latents=inactive_latent,
            feature_scales=feature_scale,
        )
        assert jacobian.item() == 0.0

    for beta in (0.0, -1e-8, -2.0):
        updated = apply_selected_latent_update(
            latents=[0.0],
            selected_indices=[0],
            scales=[3.0],
            beta=[beta],
        )
        assert updated == [0.0]


@pytest.mark.unit
def test_feature_jacobian__unequal_variant_counts__mean_within_question_then_across_questions() -> (
    None
):
    """FEAT-013: question-macro Jacobian, not prompt pooling.

    q1: ten variants with per-prompt Jacobian [4, 0]
    q2: one variant with per-prompt Jacobian [0, 6]
    → q1 mean = [4, 0], q2 mean = [0, 6], overall = [2, 3]
    Feature 1 must outrank feature 0. Prompt pooling would give [40/11, 6/11].
    """
    jacobians_by_question = {
        "q1": [torch.tensor([4.0, 0.0], dtype=torch.float64) for _ in range(10)],
        "q2": [torch.tensor([0.0, 6.0], dtype=torch.float64)],
    }

    overall = question_macro_jacobian(jacobians_by_question)

    assert overall.tolist() == [2.0, 3.0]
    assert overall[1] > overall[0]

    # Prompt pooling would weight q1 ten times more and flip the ranking.
    pooled = torch.stack(
        [j for variants in jacobians_by_question.values() for j in variants]
    ).mean(dim=0)
    assert pooled.tolist() != pytest.approx([2.0, 3.0])
    assert pooled[0] > pooled[1]


@pytest.mark.unit
def test_feature_jacobian__varying_activation_masks__break_naive_aggregate_first_formula() -> (
    None
):
    """FEAT-016: projecting mean(g) cannot recover exact masked scaled Jacobians.

    Two prompts have different residual gradients and different activity masks.
    Exact mean of per-prompt Jacobians is not recoverable by projecting the
    mean residual gradient under any single post-hoc mask.
    """
    from epistemic_sycophancy.feature_selection import project_residual_gradient

    decoder = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    scales = torch.tensor([2.0, 3.0, 0.5], dtype=torch.float64)
    gradients = torch.tensor(
        [
            [2.0, -1.0],  # h = [2, -1, 1]
            [0.0, 4.0],  # h = [0, 4, 4]
        ],
        dtype=torch.float64,
    )
    latents = torch.tensor(
        [
            [1.0, 0.0, 1.0],  # mask [1, 0, 1] → J = [4, 0, 0.5]
            [0.0, 1.0, 1.0],  # mask [0, 1, 1] → J = [0, 12, 2]
        ],
        dtype=torch.float64,
    )

    raw = project_residual_gradient(gradient=gradients, decoder=decoder)
    exact_per_prompt = coefficient_jacobian(
        raw_projection=raw,
        latents=latents,
        feature_scales=scales,
    )
    exact = exact_per_prompt.mean(dim=0)
    # Hand-derived: mean([4, 0, 0.5], [0, 12, 2]) = [2, 6, 1.25]
    assert exact.tolist() == pytest.approx([2.0, 6.0, 1.25])

    naive_raw = project_residual_gradient(
        gradient=gradients.mean(dim=0), decoder=decoder
    )
    for mask in (
        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64),
        (latents > 0).to(torch.float64).mean(dim=0),
    ):
        naive = scales * mask * naive_raw
        assert not torch.allclose(naive, exact, atol=1e-12, rtol=0.0)
