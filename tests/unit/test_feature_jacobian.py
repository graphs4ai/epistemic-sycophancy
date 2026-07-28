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


@pytest.mark.unit
def test_feature_jacobian__constant_masks_and_scales__permit_aggregate_first_equivalence() -> (
    None
):
    """FEAT-017: aggregate-first is exact only when masks (and scales) are constant."""
    from epistemic_sycophancy.feature_selection import (
        coefficient_jacobian_aggregate_first,
        project_residual_gradient,
    )

    decoder = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    scales = torch.tensor([2.0, 3.0, 0.5], dtype=torch.float64)
    gradients = torch.tensor(
        [
            [2.0, -1.0],
            [0.0, 4.0],
        ],
        dtype=torch.float64,
    )
    constant_latents = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.5, 0.0, 1.0],  # same activity pattern: [True, False, True]
        ],
        dtype=torch.float64,
    )

    exact = coefficient_jacobian(
        raw_projection=project_residual_gradient(
            gradient=gradients, decoder=decoder
        ),
        latents=constant_latents,
        feature_scales=scales,
    ).mean(dim=0)
    fast = coefficient_jacobian_aggregate_first(
        residual_gradients=gradients,
        latents=constant_latents,
        decoder=decoder,
        feature_scales=scales,
    )
    assert torch.allclose(fast, exact, atol=1e-12, rtol=0.0)

    varying_latents = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="constant"):
        coefficient_jacobian_aggregate_first(
            residual_gradients=gradients,
            latents=varying_latents,
            decoder=decoder,
            feature_scales=scales,
        )


@pytest.mark.unit
def test_feature_jacobian__streamed_batches__match_single_batch_reference() -> None:
    """FEAT-019: batch size / row order must not change the accumulated Jacobian."""
    from epistemic_sycophancy.feature_selection import (
        StreamingJacobianAccumulator,
        project_residual_gradient,
        question_macro_jacobian,
    )

    decoder = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    scales = torch.tensor([2.0, 3.0, 0.5], dtype=torch.float64)
    # Three prompts, two questions (unequal variants).
    gradients = torch.tensor(
        [
            [2.0, -1.0],  # q1
            [1.0, 0.0],  # q1
            [0.0, 4.0],  # q2
        ],
        dtype=torch.float64,
    )
    latents = torch.tensor(
        [
            [1.0, 0.5, 2.0],
            [0.5, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    question_ids = ["q1", "q1", "q2"]

    per_prompt = coefficient_jacobian(
        raw_projection=project_residual_gradient(
            gradient=gradients, decoder=decoder
        ),
        latents=latents,
        feature_scales=scales,
    )
    by_question: dict[str, list[torch.Tensor]] = {"q1": [], "q2": []}
    for row, question_id in enumerate(question_ids):
        by_question[question_id].append(per_prompt[row])
    reference = question_macro_jacobian(by_question)

    def _stream(order: list[int], prompt_batch_size: int) -> torch.Tensor:
        acc = StreamingJacobianAccumulator(
            n_features=decoder.shape[0],
            feature_chunk_size=2,
            prompt_batch_size=prompt_batch_size,
        )
        ordered_grads = gradients[order]
        ordered_latents = latents[order]
        ordered_qids = [question_ids[i] for i in order]
        for start in range(0, len(order), prompt_batch_size):
            end = start + prompt_batch_size
            acc.update(
                residual_gradients=ordered_grads[start:end],
                latents=ordered_latents[start:end],
                decoder=decoder,
                feature_scales=scales,
                question_ids=ordered_qids[start:end],
            )
        return acc.finalize()

    for batch_size in (1, 2, 3):
        for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
            streamed = _stream(order, batch_size)
            assert torch.allclose(streamed, reference, atol=1e-10, rtol=1e-9)


@pytest.mark.unit
def test_feature_jacobian__multi_token_scope__equals_sum_of_token_level_contributions() -> (
    None
):
    """FEAT-022: J_j = sum_{t in S_p} s_j 1[z_{j,t}>0] <g_t, d_j>."""
    from epistemic_sycophancy.feature_selection import (
        multi_token_coefficient_jacobian,
        project_residual_gradient,
    )

    decoder = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    scales = torch.tensor([2.0, 3.0, 0.5], dtype=torch.float64)
    # One prompt, two intervened tokens.
    token_gradients = torch.tensor(
        [
            [2.0, -1.0],  # t0: h=[2,-1,1]
            [0.0, 4.0],  # t1: h=[0,4,4]
        ],
        dtype=torch.float64,
    )
    token_latents = torch.tensor(
        [
            [1.0, 0.0, 2.0],  # mask [1,0,1] → contrib [4, 0, 0.5]
            [0.0, 1.0, 1.0],  # mask [0,1,1] → contrib [0, 12, 2]
        ],
        dtype=torch.float64,
    )

    summed = multi_token_coefficient_jacobian(
        token_gradients=token_gradients,
        token_latents=token_latents,
        decoder=decoder,
        feature_scales=scales,
    )
    # Hand-derived sum: [4, 0, 0.5] + [0, 12, 2] = [4, 12, 2.5]
    assert summed.tolist() == pytest.approx([4.0, 12.0, 2.5])

    # Final-token-only would miss the first token's contribution.
    final_only = coefficient_jacobian(
        raw_projection=project_residual_gradient(
            gradient=token_gradients[-1], decoder=decoder
        ),
        latents=token_latents[-1],
        feature_scales=scales,
    )
    assert final_only.tolist() == pytest.approx([0.0, 12.0, 2.0])
    assert not torch.allclose(summed, final_only)
