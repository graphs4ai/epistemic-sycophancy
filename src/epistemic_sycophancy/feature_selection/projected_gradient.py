"""Decoder-direction projection of residual gradients (Phase F)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from epistemic_sycophancy.feature_selection.exceptions import ScopeMismatchError


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


class StreamingJacobianAccumulator:
    """Stream prompt batches into a question-macro Jacobian (FEAT-019 / DEC-022).

    Accumulates per-prompt Jacobians keyed by question_id in float64. Does not
    materialize a dataset-sized ``[all_prompts, all_features]`` tensor; each
    ``update`` projects only the current batch (optionally feature-chunked).
    ``prompt_batch_size`` and ``feature_chunk_size`` are explicit required
    constructor arguments with no hidden defaults.
    """

    def __init__(
        self,
        *,
        n_features: int,
        feature_chunk_size: int,
        prompt_batch_size: int,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be positive; got {n_features!r}")
        if feature_chunk_size < 1:
            raise ValueError(
                f"feature_chunk_size must be a positive int; got {feature_chunk_size!r}"
            )
        if prompt_batch_size < 1:
            raise ValueError(
                f"prompt_batch_size must be a positive int; got {prompt_batch_size!r}"
            )
        self.n_features = n_features
        self.feature_chunk_size = feature_chunk_size
        self.prompt_batch_size = prompt_batch_size
        self._by_question: dict[object, list[torch.Tensor]] = {}

    def update(
        self,
        *,
        residual_gradients: torch.Tensor,  # [batch, d_model]
        latents: torch.Tensor,  # [batch, n_features]
        decoder: torch.Tensor,  # [n_features, d_model]
        feature_scales: torch.Tensor,  # [n_features]
        question_ids: Sequence[object],
    ) -> None:
        """Accumulate coefficient Jacobians for one prompt batch."""
        batch = residual_gradients.shape[0]
        if batch > self.prompt_batch_size:
            raise ValueError(
                f"batch size {batch} exceeds prompt_batch_size={self.prompt_batch_size}"
            )
        if len(question_ids) != batch:
            raise ValueError(
                f"question_ids length must equal batch; got {len(question_ids)} vs {batch}"
            )
        if decoder.shape[0] != self.n_features:
            raise ValueError(
                f"decoder n_features {decoder.shape[0]} != accumulator {self.n_features}"
            )
        raw = project_residual_gradient(
            gradient=residual_gradients,
            decoder=decoder,
            feature_chunk_size=self.feature_chunk_size,
        )
        per_prompt = coefficient_jacobian(
            raw_projection=raw,
            latents=latents,
            feature_scales=feature_scales,
        )
        for row, question_id in enumerate(question_ids):
            self._by_question.setdefault(question_id, []).append(
                per_prompt[row].detach().to(dtype=torch.float64)
            )

    def finalize(self) -> torch.Tensor:
        """Return the question-macro mean Jacobian over all streamed prompts."""
        return question_macro_jacobian(self._by_question)


@dataclass(frozen=True)
class AttributionScopeResolution:
    """Resolved attribution/intervention scope pairing (DEC-023)."""

    attribution_scope: str
    intervention_token_scope: str
    scope_label: str  # "exact" | "heuristic"


def resolve_attribution_scope(
    *,
    attribution_scope: str,
    intervention_token_scope: str,
    allow_heuristic_mismatch: bool = False,
) -> AttributionScopeResolution:
    """Require attribution_scope == intervention token_scope unless overridden.

    A mismatch raises ``ScopeMismatchError`` unless
    ``allow_heuristic_mismatch=True``, in which case the result is labeled
    ``heuristic`` and both scopes are retained (FEAT-021 / DEC-023).
    """
    if attribution_scope == intervention_token_scope:
        return AttributionScopeResolution(
            attribution_scope=attribution_scope,
            intervention_token_scope=intervention_token_scope,
            scope_label="exact",
        )
    if allow_heuristic_mismatch:
        return AttributionScopeResolution(
            attribution_scope=attribution_scope,
            intervention_token_scope=intervention_token_scope,
            scope_label="heuristic",
        )
    raise ScopeMismatchError(
        "attribution_scope must equal intervention token_scope; "
        f"got attribution_scope={attribution_scope!r}, "
        f"intervention_token_scope={intervention_token_scope!r}"
    )


def multi_token_coefficient_jacobian(
    *,
    token_gradients: torch.Tensor,  # [n_tokens, d_model]
    token_latents: torch.Tensor,  # [n_tokens, n_features]
    decoder: torch.Tensor,  # [n_features, d_model]
    feature_scales: torch.Tensor,  # [n_features]
) -> torch.Tensor:  # [n_features]
    """Sum token-level coefficient Jacobians over intervention positions S_p.

    J_j = sum_{t in S_p} s_j 1[z_{j,t}>0] <g_t, d_j> (FEAT-022).
    """
    raw = project_residual_gradient(gradient=token_gradients, decoder=decoder)
    per_token = coefficient_jacobian(
        raw_projection=raw,
        latents=token_latents,
        feature_scales=feature_scales,
    )
    return per_token.sum(dim=0)
