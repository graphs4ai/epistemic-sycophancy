"""FSC-003: §11.3 weighted scalar backward matches explicit question-macro."""

from __future__ import annotations

import pytest
import torch

from epistemic_sycophancy.feature_selection.components import (
    logistic_preservation_surrogate,
)
from epistemic_sycophancy.feature_selection.projected_gradient import (
    coefficient_jacobian,
    project_residual_gradient,
    question_macro_jacobian,
    question_macro_prompt_weights,
    sum_coefficient_jacobians,
)


DTYPE = torch.float64


@pytest.mark.unit
def test_fs_batch__weighted_scalar_backward__matches_explicit_question_macro() -> None:
    """FSC-003 / FEAT-014: w_p applied once; not twice after projection."""
    from epistemic_sycophancy.runner.adapters.fs_batch import (
        weighted_component_residual_grads,
    )

    # Unequal variant counts: q1 has 2 prompts, q2 has 1.
    residuals = torch.tensor(
        [[2.0, 1.0], [2.5, 0.5], [1.0, 2.0]],
        dtype=DTYPE,
        requires_grad=True,
    )
    question_ids = ["q1", "q1", "q2"]
    # Toy margin = x[0] - x[1]; φ(M)=softplus(-M/τ).
    margins = residuals[:, 0] - residuals[:, 1]
    prompt_losses = logistic_preservation_surrogate(margin=margins, tau=1.0)

    decoder = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=DTYPE)
    scales = torch.tensor([1.0, 1.0, 0.5], dtype=DTYPE)
    latents = torch.relu(residuals.detach() @ decoder.T) + 0.1  # all active

    # Explicit per-prompt → question-macro reference.
    per_prompt: dict[str, list[torch.Tensor]] = {"q1": [], "q2": []}
    for row, qid in enumerate(question_ids):
        leaf = residuals[row].detach().clone().requires_grad_(True)
        m = leaf[0] - leaf[1]
        loss = logistic_preservation_surrogate(margin=m, tau=1.0)
        g = torch.autograd.grad(loss, leaf)[0]
        per_prompt[qid].append(
            coefficient_jacobian(
                raw_projection=project_residual_gradient(gradient=g, decoder=decoder),
                latents=latents[row],
                feature_scales=scales,
            ).detach()
        )
    explicit = question_macro_jacobian(per_prompt)

    residual_grads = weighted_component_residual_grads(
        prompt_losses=prompt_losses,
        residual=residuals,
        question_ids=question_ids,
    )
    from_weighted = sum_coefficient_jacobians(
        residual_gradients=residual_grads,
        latents=latents,
        decoder=decoder,
        feature_scales=scales,
    )
    assert torch.allclose(from_weighted, explicit, atol=1e-8, rtol=1e-6)

    # Guard: applying weights again after a weighted backward must NOT match.
    weights = question_macro_prompt_weights(question_ids=question_ids)
    double = (residual_grads * weights.unsqueeze(-1)).sum(dim=0)
    # project via sum path with double-weighted grads would diverge unless
    # all weights equal — assert the raw weighted-sum of grads differs from
    # the correct sum of residual_grads when variant counts are unequal.
    correct_sum = residual_grads.sum(dim=0)
    assert not torch.allclose(double, correct_sum, atol=1e-8, rtol=1e-6)
