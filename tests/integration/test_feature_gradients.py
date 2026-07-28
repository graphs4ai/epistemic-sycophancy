"""Toy-model gradient equivalence for feature selection (Phase F)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from epistemic_sycophancy.feature_selection import (
    coefficient_jacobian,
    project_residual_gradient,
)
from epistemic_sycophancy.intervention.sae_delta import apply_additive_sae_delta

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def _load(module_name: str, relative_path: str):
    path = _FIXTURE_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_toy_sae = _load("toy_sae_feature_gradients", "intervention/toy_sae.py")
_toy_gradients = _load(
    "toy_gradients_feature_gradients", "feature_selection/toy_gradients.py"
)
decoder_weight = _toy_sae.decoder_weight
imperfect_encoder_params = _toy_sae.imperfect_encoder_params
asymmetric_head = _toy_gradients.asymmetric_head

DTYPE = torch.float64
TAU = 1.0
# CF prompt: truthful candidate is A, so M = s_A - s_B.
TRUTHFUL_LABEL = "A"
# DEC-021: one-sided suppression step and comparison tolerance.
FD_STEP = 1e-8
FD_ATOL = 1e-6
FD_RTOL = 1e-6


def _frozen_toy_parameters() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w_dec = decoder_weight(dtype=DTYPE).detach().clone()
    w_enc, b_enc = imperfect_encoder_params(dtype=DTYPE)
    w_enc = w_enc.detach().clone()
    b_enc = b_enc.detach().clone()
    head = asymmetric_head(dtype=DTYPE).detach().clone()
    for param in (w_dec, w_enc, b_enc, head):
        param.requires_grad_(False)
    return w_dec, w_enc, b_enc, head


def _truthful_margin_from_residual(
    residual: torch.Tensor, *, head: torch.Tensor
) -> torch.Tensor:
    logits = head @ residual
    if TRUTHFUL_LABEL == "A":
        return logits[0] - logits[1]
    return logits[1] - logits[0]


def _logistic_margin_loss(margin: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(-margin / TAU)


@pytest.mark.integration
def test_feature_jacobian__active_linear_region__matches_autograd_beta_gradient() -> None:
    """FEAT-005: s_j 1[z_j>0] <g, d_j> equals autograd d(loss)/d(beta_j)."""
    w_dec, w_enc, b_enc, head = _frozen_toy_parameters()
    residual = torch.tensor([2.0, 3.0], dtype=DTYPE)
    selected_indices = [0, 1, 2]
    scales = torch.tensor([2.0, 4.0, 0.5], dtype=DTYPE)

    # All latents are strictly positive, so beta=0 sits in a linear region.
    latents = torch.relu(residual @ w_enc.T + b_enc)
    assert bool((latents > 0).all())

    beta = torch.zeros(len(selected_indices), dtype=DTYPE, requires_grad=True)
    intervened = apply_additive_sae_delta(
        residual=residual,
        selected_indices=selected_indices,
        scales=scales,
        beta=beta,
        encoder_weight=w_enc,
        encoder_bias=b_enc,
        decoder_weight=w_dec,
    )
    autograd_jacobian = torch.autograd.grad(
        _logistic_margin_loss(_truthful_margin_from_residual(intervened, head=head)),
        beta,
    )[0]

    residual_leaf = residual.clone().requires_grad_(True)
    residual_gradient = torch.autograd.grad(
        _logistic_margin_loss(
            _truthful_margin_from_residual(residual_leaf, head=head)
        ),
        residual_leaf,
    )[0]
    projected_jacobian = coefficient_jacobian(
        raw_projection=project_residual_gradient(
            gradient=residual_gradient, decoder=w_dec
        ),
        latents=latents,
        feature_scales=scales,
    )

    assert torch.allclose(projected_jacobian, autograd_jacobian, atol=1e-8, rtol=1e-6)
    assert bool((projected_jacobian != 0).all())


@pytest.mark.integration
def test_feature_jacobian__suppression_one_sided_difference__matches_local_prediction() -> (
    None
):
    """FEAT-006: (L(-eps e_j) - L(0)) / (-eps) matches J_j (DEC-021)."""
    w_dec, w_enc, b_enc, head = _frozen_toy_parameters()
    residual = torch.tensor([2.0, 3.0], dtype=DTYPE)
    selected_indices = [0, 1, 2]
    scales = torch.tensor([2.0, 4.0, 0.5], dtype=DTYPE)
    latents = torch.relu(residual @ w_enc.T + b_enc)

    # DEC-021 feasibility guard: the step may not cross an active ReLU boundary.
    assert bool((FD_STEP * scales < latents).all())

    def component_loss(beta: torch.Tensor) -> torch.Tensor:
        intervened = apply_additive_sae_delta(
            residual=residual,
            selected_indices=selected_indices,
            scales=scales,
            beta=beta,
            encoder_weight=w_enc,
            encoder_bias=b_enc,
            decoder_weight=w_dec,
        )
        return _logistic_margin_loss(
            _truthful_margin_from_residual(intervened, head=head)
        )

    residual_leaf = residual.clone().requires_grad_(True)
    residual_gradient = torch.autograd.grad(
        _logistic_margin_loss(
            _truthful_margin_from_residual(residual_leaf, head=head)
        ),
        residual_leaf,
    )[0]
    projected_jacobian = coefficient_jacobian(
        raw_projection=project_residual_gradient(
            gradient=residual_gradient, decoder=w_dec
        ),
        latents=latents,
        feature_scales=scales,
    )

    baseline_loss = component_loss(torch.zeros(len(selected_indices), dtype=DTYPE))
    for feature in selected_indices:
        suppressed_beta = torch.zeros(len(selected_indices), dtype=DTYPE)
        suppressed_beta[feature] = -FD_STEP
        one_sided_derivative = (
            component_loss(suppressed_beta) - baseline_loss
        ) / -FD_STEP
        assert one_sided_derivative.item() == pytest.approx(
            projected_jacobian[feature].item(), abs=FD_ATOL, rel=FD_RTOL
        )


@pytest.mark.integration
def test_feature_components__logistic_preservation_surrogates__can_have_nonzero_null_gradient() -> (
    None
):
    """FEAT-012: logistic N/CB surrogates have nonzero null gradient matching autograd.

    Unlike baseline-relative hinges (FEAT-011), the logistic preservation
    surrogates remain informative at β=0 and must not be forced to zero.
    """
    from epistemic_sycophancy.feature_selection import logistic_preservation_surrogate

    w_dec, w_enc, b_enc, head = _frozen_toy_parameters()
    residual = torch.tensor([2.0, 3.0], dtype=DTYPE)
    selected_indices = [0, 1, 2]
    scales = torch.tensor([2.0, 4.0, 0.5], dtype=DTYPE)
    latents = torch.relu(residual @ w_enc.T + b_enc)
    assert bool((latents > 0).all())

    beta = torch.zeros(len(selected_indices), dtype=DTYPE, requires_grad=True)
    intervened = apply_additive_sae_delta(
        residual=residual,
        selected_indices=selected_indices,
        scales=scales,
        beta=beta,
        encoder_weight=w_enc,
        encoder_bias=b_enc,
        decoder_weight=w_dec,
    )
    surrogate_loss = logistic_preservation_surrogate(
        margin=_truthful_margin_from_residual(intervened, head=head),
        tau=TAU,
    )
    autograd_jacobian = torch.autograd.grad(surrogate_loss, beta)[0]

    residual_leaf = residual.clone().requires_grad_(True)
    residual_gradient = torch.autograd.grad(
        logistic_preservation_surrogate(
            margin=_truthful_margin_from_residual(residual_leaf, head=head),
            tau=TAU,
        ),
        residual_leaf,
    )[0]
    projected_jacobian = coefficient_jacobian(
        raw_projection=project_residual_gradient(
            gradient=residual_gradient, decoder=w_dec
        ),
        latents=latents,
        feature_scales=scales,
    )

    assert bool((autograd_jacobian != 0).any())
    assert torch.allclose(projected_jacobian, autograd_jacobian, atol=1e-8, rtol=1e-6)


@pytest.mark.integration
def test_feature_jacobian__weighted_component_backward__matches_explicit_question_macro_gradients() -> (
    None
):
    """FEAT-014: one weighted scalar backward equals explicit question-macro J.

    Weights w_p = 1 / (|Q_u| |B_{q,u}|) are applied exactly once in the
    scalar; they must not be re-multiplied after projection (spec §11.3).
    """
    from epistemic_sycophancy.feature_selection import (
        question_macro_jacobian,
        question_macro_prompt_weights,
        sum_coefficient_jacobians,
    )

    w_dec, w_enc, b_enc, head = _frozen_toy_parameters()
    # Two questions with unequal variant counts (mirrors FEAT-013 structure).
    residuals = torch.stack(
        [
            torch.tensor([2.0, 3.0], dtype=DTYPE),  # q1, v0
            torch.tensor([2.1, 2.9], dtype=DTYPE),  # q1, v1
            torch.tensor([1.5, 2.5], dtype=DTYPE),  # q2, v0
        ],
        dim=0,
    )
    question_ids = ["q1", "q1", "q2"]
    selected_indices = [0, 1, 2]
    scales = torch.tensor([2.0, 4.0, 0.5], dtype=DTYPE)
    latents = torch.relu(residuals @ w_enc.T + b_enc)
    assert bool((latents > 0).all())

    weights = question_macro_prompt_weights(question_ids=question_ids)
    # Explicit per-prompt Jacobians → question-macro reference.
    per_prompt: dict[str, list[torch.Tensor]] = {"q1": [], "q2": []}
    for row, question_id in enumerate(question_ids):
        leaf = residuals[row].clone().requires_grad_(True)
        loss = _logistic_margin_loss(
            _truthful_margin_from_residual(leaf, head=head)
        )
        residual_gradient = torch.autograd.grad(loss, leaf)[0]
        per_prompt[question_id].append(
            coefficient_jacobian(
                raw_projection=project_residual_gradient(
                    gradient=residual_gradient, decoder=w_dec
                ),
                latents=latents[row],
                feature_scales=scales,
            ).detach()
        )
    explicit = question_macro_jacobian(per_prompt)

    # Single weighted scalar backward (weights applied once).
    residuals_leaf = residuals.clone().requires_grad_(True)
    prompt_losses = torch.stack(
        [
            _logistic_margin_loss(
                _truthful_margin_from_residual(residuals_leaf[row], head=head)
            )
            for row in range(residuals_leaf.shape[0])
        ]
    )
    weighted = (weights * prompt_losses).sum()
    residual_grads = torch.autograd.grad(weighted, residuals_leaf)[0]
    from_weighted = sum_coefficient_jacobians(
        residual_gradients=residual_grads,
        latents=latents,
        decoder=w_dec,
        feature_scales=scales,
    )

    assert torch.allclose(from_weighted, explicit, atol=1e-8, rtol=1e-6)
