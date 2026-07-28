"""Decoder-direction projection of residual gradients (Phase F)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def project_residual_gradient(
    *,
    gradient: torch.Tensor,  # [..., d_model]
    decoder: torch.Tensor,  # [n_features, d_model]
) -> torch.Tensor:  # [..., n_features]
    """Return the raw projection h = g W_dec^T onto decoder directions."""
    return gradient @ decoder.T


def coefficient_jacobian(
    *,
    raw_projection: torch.Tensor,  # [..., n_features]
    latents: torch.Tensor,  # [..., n_features]
    feature_scales: torch.Tensor,  # [n_features]
) -> torch.Tensor:  # [..., n_features]
    """Return the exact local derivative w.r.t. normalized coefficients.

    J_j = s_j * 1[z_j > 0] * h_j (AGENTS.md §5.8). The activity mask is
    strict: a latent sitting exactly at zero contributes nothing under a
    feasible nonpositive coefficient change.
    """
    activity_mask = (latents > 0).to(raw_projection.dtype)
    return feature_scales * activity_mask * raw_projection


def question_macro_jacobian(
    jacobians_by_question: Mapping[object, Sequence[torch.Tensor]],
) -> torch.Tensor:
    """Mean within each question, then mean across questions (FEAT-013).

    Each question receives equal weight regardless of its number of variants.
    Prompt pooling is forbidden.

    Args:
        jacobians_by_question: Mapping from question_id to per-prompt
            Jacobian tensors of shape ``[n_features]``.

    Returns:
        Overall Jacobian of shape ``[n_features]``.
    """
    if not jacobians_by_question:
        raise ValueError("jacobians_by_question must be non-empty")
    question_means: list[torch.Tensor] = []
    for variants in jacobians_by_question.values():
        if not variants:
            raise ValueError("each question must have at least one Jacobian")
        question_means.append(torch.stack(list(variants), dim=0).mean(dim=0))
    return torch.stack(question_means, dim=0).mean(dim=0)


def question_macro_prompt_weights(
    *,
    question_ids: Sequence[object],
) -> torch.Tensor:
    """Return w_p = 1 / (|Q_u| |B_{q,u}|) for each prompt (spec §11.3).

    Shape: ``[n_prompts]``. Applying these weights once in a scalar loss
    makes a single backward equal to the explicit question-macro mean.
    """
    if not question_ids:
        raise ValueError("question_ids must be non-empty")
    counts: dict[object, int] = {}
    for question_id in question_ids:
        counts[question_id] = counts.get(question_id, 0) + 1
    n_questions = len(counts)
    return torch.tensor(
        [1.0 / (n_questions * counts[question_id]) for question_id in question_ids],
        dtype=torch.float64,
    )


def sum_coefficient_jacobians(
    *,
    residual_gradients: torch.Tensor,  # [batch, d_model]
    latents: torch.Tensor,  # [batch, n_features]
    decoder: torch.Tensor,  # [n_features, d_model]
    feature_scales: torch.Tensor,  # [n_features]
) -> torch.Tensor:  # [n_features]
    """Sum per-prompt coefficient Jacobians without re-applying weights.

    Intended for residual gradients that already came from a weighted scalar
    backward (FEAT-014). Do not multiply by prompt weights again.
    """
    raw = project_residual_gradient(gradient=residual_gradients, decoder=decoder)
    per_prompt = coefficient_jacobian(
        raw_projection=raw,
        latents=latents,
        feature_scales=feature_scales,
    )
    return per_prompt.sum(dim=0)
