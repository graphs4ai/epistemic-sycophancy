"""Projected Adam optimizer tests (Phase H OPT-005+)."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
def test_projected_adam__optimizer_step__clamps_beta_to_bounds() -> None:
    """OPT-005: after Adam step, β is projected onto configured bounds."""
    from epistemic_sycophancy.optimization.projected_adam import ProjectedAdam

    beta_lower = -2.0
    beta_upper = 0.0
    # Start at upper bound with large negative gradient → would go above 0 without clamp.
    beta = torch.tensor([0.0, -1.0, -2.0], dtype=torch.float64, requires_grad=True)
    optimizer = ProjectedAdam(
        beta=beta,
        adam_lr=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_microbatch_questions=1,
        beta_lower=beta_lower,
        beta_upper=beta_upper,
    )
    # Synthetic grad: push first coord positive, second more negative, third below lower.
    beta.grad = torch.tensor([-10.0, 10.0, 10.0], dtype=torch.float64)
    optimizer.step()

    values = beta.detach().tolist()
    assert all(beta_lower <= v <= beta_upper for v in values)
    assert values[0] == pytest.approx(0.0)
    assert values[2] == pytest.approx(-2.0)


@pytest.mark.unit
def test_projected_adam__trainable_parameters__contains_only_beta() -> None:
    """OPT-006: Adam trainable parameter set contains only β."""
    from epistemic_sycophancy.optimization.projected_adam import ProjectedAdam

    beta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    decoy = torch.ones(2, dtype=torch.float64, requires_grad=True)
    optimizer = ProjectedAdam(
        beta=beta,
        adam_lr=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_microbatch_questions=1,
        beta_lower=-2.0,
        beta_upper=0.0,
    )
    params = [p for group in optimizer.torch_optimizer.param_groups for p in group["params"]]
    assert len(params) == 1
    assert params[0] is beta
    assert not any(p is decoy for p in params)


@pytest.mark.unit
def test_projected_adam__zero_learning_rate__leaves_beta_unchanged() -> None:
    """OPT-008: lr=0 leaves β unchanged after a step."""
    from epistemic_sycophancy.optimization.projected_adam import ProjectedAdam

    beta = torch.tensor([-1.0, -0.5, 0.0], dtype=torch.float64, requires_grad=True)
    before = beta.detach().clone()
    optimizer = ProjectedAdam(
        beta=beta,
        adam_lr=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_microbatch_questions=1,
        beta_lower=-2.0,
        beta_upper=0.0,
    )
    beta.grad = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64)
    optimizer.step()
    assert torch.equal(beta.detach(), before)
