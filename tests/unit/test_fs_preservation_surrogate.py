"""FSC-004: preservation FS components use φ(M), nonzero at β=0."""

from __future__ import annotations

import pytest
import torch

from epistemic_sycophancy.feature_selection.components import (
    logistic_preservation_surrogate,
)
from epistemic_sycophancy.objective.losses import baseline_relative_hard_hinge


@pytest.mark.unit
@pytest.mark.parametrize("component", ["neutral_surrogate", "correct_surrogate"])
def test_fs_adapter__preservation_components__use_phi_nonzero_at_beta_zero(
    component: str,
) -> None:
    """FSC-004 / FEAT-011/012: production FS uses φ, never hinges at β=0."""
    from epistemic_sycophancy.runner.adapters.fs_batch import fs_component_margin_loss

    tau = 1.0
    baseline = 2.0
    margin = torch.tensor(baseline, dtype=torch.float64, requires_grad=True)
    loss = fs_component_margin_loss(margin=margin, tau=tau, component=component)
    expected = logistic_preservation_surrogate(margin=margin.detach(), tau=tau)
    assert float(loss.item()) == pytest.approx(float(expected.item()))

    (grad,) = torch.autograd.grad(loss, margin)
    assert grad is not None
    assert abs(float(grad.item())) > 0.0

    # Contrasts with forbidden hard hinge (flat at β=0 when δ>0).
    hinge = baseline_relative_hard_hinge(
        baseline_margin=baseline,
        current_margin=torch.tensor(baseline, dtype=torch.float64, requires_grad=True),
        delta=0.25,
    )
    assert float(hinge.item()) == 0.0
