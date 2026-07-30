"""Selected-pool margin Jacobian builder (GRAD-003 / DEC-084).

Reuses Phase F ``project_residual_gradient`` + ``coefficient_jacobian``.
Does not reimplement the exact local coefficient Jacobian.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from epistemic_sycophancy.feature_selection.projected_gradient import (
    coefficient_jacobian,
    project_residual_gradient,
)


def project_selected_margin_jacobian(
    *,
    residual_gradient: torch.Tensor,
    latents: torch.Tensor,
    decoder: torch.Tensor,
    feature_scales: torch.Tensor,
    selected_indices: Sequence[int],
    feature_chunk_size: int | None = None,
) -> torch.Tensor:
    """Return ∂M/∂β over selected features: length-m float64 row.

    ``residual_gradient`` is ∂M/∂x at the intervened token (not ∂φ/∂x).
    Activity mask and scales follow AGENTS.md §5.8 / DEC-053.
    """
    if residual_gradient.ndim != 1:
        raise ValueError(
            f"residual_gradient must be rank-1 [d_model], got {tuple(residual_gradient.shape)}"
        )
    if latents.ndim != 1:
        raise ValueError(f"latents must be rank-1 [n_features], got {tuple(latents.shape)}")
    raw = project_residual_gradient(
        gradient=residual_gradient.to(dtype=torch.float64),
        decoder=decoder.to(dtype=torch.float64),
        feature_chunk_size=feature_chunk_size,
    )
    full = coefficient_jacobian(
        raw_projection=raw,
        latents=latents.to(dtype=torch.float64),
        feature_scales=feature_scales.to(dtype=torch.float64),
    )
    if not selected_indices:
        raise ValueError("selected_indices must be nonempty")
    return torch.stack([full[int(i)] for i in selected_indices]).to(dtype=torch.float64)


def assemble_margin_jacobian_maps(
    *,
    prompts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble IB/CB sequences and neutral map for ``evaluate_objective_with_grad``.

    Each prompt mapping requires ``question_id``, ``belief_condition`` ∈ {N,IB,CB},
    and ``jac_row`` (``Tensor[m]``).
    """
    ib: dict[str, list[torch.Tensor]] = {}
    cb: dict[str, list[torch.Tensor]] = {}
    neutral: dict[str, torch.Tensor] = {}
    for prompt in prompts:
        qid = str(prompt["question_id"])
        condition = str(prompt["belief_condition"]).upper()
        jac_row = prompt["jac_row"]
        if not isinstance(jac_row, torch.Tensor):
            jac_row = torch.as_tensor(jac_row, dtype=torch.float64)
        else:
            jac_row = jac_row.to(dtype=torch.float64)
        if condition == "N":
            if qid in neutral:
                raise ValueError(f"duplicate neutral margin jac for question_id={qid!r}")
            neutral[qid] = jac_row
        elif condition == "IB":
            ib.setdefault(qid, []).append(jac_row)
        elif condition == "CB":
            cb.setdefault(qid, []).append(jac_row)
        else:
            raise ValueError(f"unsupported belief_condition={condition!r}")
    return {
        "ib_margin_jac": ib,
        "cb_margin_jac": cb,
        "neutral_margin_jac": neutral,
    }
