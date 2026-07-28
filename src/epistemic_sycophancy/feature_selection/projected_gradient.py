"""Decoder-direction projection of residual gradients (Phase F)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def project_residual_gradient(
    *,
    gradient: torch.Tensor,  # [..., d_model]
    decoder: torch.Tensor,  # [n_features, d_model]
    feature_chunk_size: int | None = None,
) -> torch.Tensor:  # [..., n_features]
    """Return the raw projection h = g W_dec^T onto decoder directions.

    When ``feature_chunk_size`` is provided (DEC-022), project in feature
    chunks so a wide SAE need not materialize the full matmul at once. The
    final chunk may be uneven. ``None`` selects the dense path.
    """
    if feature_chunk_size is None:
        return gradient @ decoder.T
    if feature_chunk_size < 1:
        raise ValueError(
            f"feature_chunk_size must be a positive int; got {feature_chunk_size!r}"
        )
    n_features = decoder.shape[0]
    chunks: list[torch.Tensor] = []
    for start in range(0, n_features, feature_chunk_size):
        end = min(start + feature_chunk_size, n_features)
        chunks.append(gradient @ decoder[start:end].T)
    return torch.cat(chunks, dim=-1)


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


def coefficient_jacobian_aggregate_first(
    *,
    residual_gradients: torch.Tensor,  # [batch, d_model]
    latents: torch.Tensor,  # [batch, n_features]
    decoder: torch.Tensor,  # [n_features, d_model]
    feature_scales: torch.Tensor,  # [n_features]
) -> torch.Tensor:  # [n_features]
    """Fast path: project mean(g) then apply a shared activity mask (FEAT-017).

    Exact only when activity masks are identical across the batch. Raises
    ``ValueError`` when masks vary; use per-prompt ``coefficient_jacobian``
    then mean in that case (FEAT-016).
    """
    activity = latents > 0
    if activity.ndim != 2:
        raise ValueError(f"latents must be [batch, n_features]; got {tuple(latents.shape)}")
    if not bool(torch.all(activity == activity[:1])):
        raise ValueError(
            "aggregate-first requires constant activity masks across prompts; "
            "use per-prompt coefficient_jacobian when masks vary"
        )
    shared_mask = activity[0].to(residual_gradients.dtype)
    mean_raw = project_residual_gradient(
        gradient=residual_gradients.mean(dim=0), decoder=decoder
    )
    return feature_scales * shared_mask * mean_raw
