"""Selected-pool margin Jacobian builder (GRAD-003 / DEC-084 / GRAD-014).

Reuses Phase F ``project_residual_gradient`` + ``coefficient_jacobian``.
Does not reimplement the exact local coefficient Jacobian. Live β≠0 uses
shifted activity mask ``1[z + sβ > 0]`` via ``shift_latents_for_live_beta``.
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
    Activity mask and scales follow AGENTS.md §5.8 / DEC-053. For live β≠0,
    pass pre-ReLU shifted latents ``z + s⊙β`` so the mask is ``1[z+sβ>0]``
    (GRAD-014); at β=0 this reduces to ``1[z>0]``.
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
    # evaluate_objective_with_grad uses CPU float64 β (DEC-027); keep jac on CPU.
    return (
        torch.stack([full[int(i)] for i in selected_indices])
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )


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


def _selected_indices_for_layer(
    feature_ids: Sequence[object],
    *,
    layer: int,
) -> list[int]:
    """Return SAE feature indices for ``layer`` in pool order (DEC-054)."""
    indices: list[int] = []
    for key in feature_ids:
        if isinstance(key, (tuple, list)) and len(key) == 2:
            key_layer, feature_id = int(key[0]), int(key[1])
            if key_layer == layer:
                indices.append(feature_id)
        else:
            # Legacy flat int IDs: treat as same-layer indices in order.
            indices.append(int(key))
    return indices


def shift_latents_for_live_beta(
    latents: torch.Tensor,
    *,
    selected_indices: Sequence[int],
    feature_scales: torch.Tensor,
    beta: Sequence[float],
) -> torch.Tensor:
    """Return pre-ReLU shifted latents z + s⊙β on selected coordinates (GRAD-014).

    ``coefficient_jacobian`` masks with ``1[· > 0]``, so shifted values yield the
    live intervention mask ``1[z_j + s_j β_j > 0]`` required by
    ``z'_j = ReLU(z_j + s_j β_j)``. Non-selected coordinates are unchanged.
    """
    if len(selected_indices) != len(beta):
        raise ValueError(
            f"selected_indices length {len(selected_indices)} != beta length {len(beta)}"
        )
    shifted = latents.to(dtype=torch.float64).clone()
    scales = feature_scales.to(dtype=torch.float64)
    for pool_i, feat_i in enumerate(selected_indices):
        idx = int(feat_i)
        shifted[..., idx] = shifted[..., idx] + scales[idx] * float(beta[pool_i])
    return shifted


def build_margin_jacobian_fn(
    study: Any,
    stack: Any,
    *,
    corpus: Sequence[Mapping[str, object]] | None = None,
    split_question_ids: Mapping[str, Sequence[str]] | None = None,
    order_regime: str = "CF",
) -> Any:
    """Build ``(*, beta, question_ids, partitions) -> jac maps`` (DEC-084).

    Uses ``stack.margin_projection_batch`` when present; otherwise requires
    ``corpus`` + ``split_question_ids`` and ``compute_margin_projection_batch``.
    Live activity masks use ``1[z + sβ > 0]`` (GRAD-014), even when the batch
    returns baseline (unshifted) JumpReLU latents.
    """
    batch_fn = getattr(stack, "margin_projection_batch", None)
    if batch_fn is None:
        if corpus is None or split_question_ids is None:
            raise ValueError(
                "build_margin_jacobian_fn requires stack.margin_projection_batch "
                "or corpus+split_question_ids for projected ∂M/∂β (DEC-084)"
            )
        batch_fn = _make_production_margin_batch_fn(
            study,
            stack,
            corpus=corpus,
            split_question_ids=split_question_ids,
            order_regime=order_regime,
        )

    def margin_jacobian_fn(
        *,
        beta: Sequence[float],
        question_ids: Sequence[str],
        partitions: Mapping[str, Any],
    ) -> dict[str, Any]:
        del partitions
        qids = tuple(str(q) for q in question_ids)
        beta_t = tuple(float(b) for b in beta)
        prompts: list[dict[str, Any]] = []
        chunk = int(study.run.feature_chunk_size)
        m = int(study.experiment.coefficient_length)
        for belief in ("N", "IB", "CB"):
            batch = batch_fn(
                belief_condition=belief,
                question_ids=qids,
                beta=beta_t,
            )
            residual_gradients = batch["residual_gradients"]
            decoder = batch["decoder"]
            feature_scales = batch["feature_scales"]
            batch_qids = [str(q) for q in batch["question_ids"]]
            layer = int(batch.get("layer", study.stack.sae.layers[0]))
            selected = _selected_indices_for_layer(
                study.experiment.feature_ids,
                layer=layer,
            )
            if len(selected) != m:
                raise ValueError(
                    f"selected features at layer={layer} ({len(selected)}) != "
                    f"coefficient_length={m}; multi-layer margin jac unsupported"
                )
            if len(beta_t) != m:
                raise ValueError(
                    f"beta length {len(beta_t)} != coefficient_length={m}"
                )
            # Live mask 1[z+sβ>0]: shift baseline post-encode latents before
            # coefficient_jacobian's strict >0 gate (GRAD-014 / DEC-084).
            latents = shift_latents_for_live_beta(
                batch["latents"],
                selected_indices=selected,
                feature_scales=feature_scales,
                beta=beta_t,
            )
            for row_idx, qid in enumerate(batch_qids):
                jac_row = project_selected_margin_jacobian(
                    residual_gradient=residual_gradients[row_idx],
                    latents=latents[row_idx],
                    decoder=decoder,
                    feature_scales=feature_scales,
                    selected_indices=selected,
                    feature_chunk_size=chunk,
                )
                prompts.append(
                    {
                        "question_id": qid,
                        "belief_condition": belief,
                        "jac_row": jac_row,
                    }
                )
        return assemble_margin_jacobian_maps(prompts=prompts)

    return margin_jacobian_fn


def _make_production_margin_batch_fn(
    study: Any,
    stack: Any,
    *,
    corpus: Sequence[Mapping[str, object]],
    split_question_ids: Mapping[str, Sequence[str]],
    order_regime: str,
):
    from epistemic_sycophancy.prompts.render import render_mc0_subset
    from epistemic_sycophancy.runner.adapters.margin_batch import (
        compute_margin_projection_batch,
    )

    tok = stack.tokenizer
    token_a = list(
        tok.encode(study.experiment.continuation_A, add_special_tokens=False)
    )
    token_b = list(
        tok.encode(study.experiment.continuation_B, add_special_tokens=False)
    )
    layer = int(study.stack.sae.layers[0])

    def margin_projection_batch(
        *,
        belief_condition: str,
        question_ids: Sequence[str],
        beta: Sequence[float],
    ) -> dict[str, Any]:
        # Baseline JumpReLU z + unintervened ∂M/∂x. Live activity mask
        # 1[z + sβ > 0] is applied in build_margin_jacobian_fn (GRAD-014).
        _ = beta
        qids = tuple(str(q) for q in question_ids)
        rendered = render_mc0_subset(
            corpus_rows=corpus,
            question_ids=qids,
            split_question_ids=split_question_ids,
            order_regime=order_regime,
            belief_condition=belief_condition,
        )
        if belief_condition == "N":
            rows: list[Any] = []
            seen: set[str] = set()
            for row in rendered:
                if row.question_id in seen:
                    continue
                seen.add(row.question_id)
                rows.append(row)
        else:
            rows = list(rendered)
        if not rows:
            raise ValueError(
                f"margin_projection_batch found no rows for "
                f"belief={belief_condition!r} qids={qids!r}"
            )
        return compute_margin_projection_batch(
            stack,
            layer=layer,
            texts=[r.text for r in rows],
            question_ids=[r.question_id for r in rows],
            continuation_token_ids_A=token_a,
            continuation_token_ids_B=token_b,
            truthful_labels=[r.truthful_label for r in rows],
            prompt_batch_size=int(study.run.prompt_batch_size),
        )

    return margin_projection_batch
