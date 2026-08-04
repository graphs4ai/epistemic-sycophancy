"""Selected-pool margin Jacobian builder (GRAD-003 / DEC-084 / GRAD-014 / GRAD-015).

Reuses Phase F ``project_residual_gradient`` + ``coefficient_jacobian``.
Does not reimplement the exact local coefficient Jacobian. Live β≠0 uses
shifted activity mask ``1[z + sβ > 0]`` via ``shift_latents_for_live_beta``.
Multi-layer pools (DEC-054) project per SAE and scatter into the length-m row.
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


def _normalize_pool_keys(
    feature_ids: Sequence[object],
    *,
    default_layer: int,
) -> tuple[tuple[int, int], ...]:
    """Normalize CFG feature_ids to ``(layer, feature_id)`` pool keys (DEC-054)."""
    keys: list[tuple[int, int]] = []
    for key in feature_ids:
        if isinstance(key, (tuple, list)) and len(key) == 2:
            keys.append((int(key[0]), int(key[1])))
        else:
            keys.append((int(default_layer), int(key)))
    return tuple(keys)


def _layer_pool_slices(
    feature_ids: Sequence[object],
    *,
    default_layer: int,
) -> dict[int, list[tuple[int, int]]]:
    """Map layer → ``[(pool_index, sae_feature_id), ...]`` in pool order."""
    slices: dict[int, list[tuple[int, int]]] = {}
    for pool_idx, (layer, feature_id) in enumerate(
        _normalize_pool_keys(feature_ids, default_layer=default_layer)
    ):
        slices.setdefault(layer, []).append((pool_idx, feature_id))
    return slices


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

    ``selected_indices`` and ``beta`` must be a layer-local aligned slice (GRAD-015).
    """
    if len(selected_indices) != len(beta):
        raise ValueError(
            f"selected_indices length {len(selected_indices)} != beta length {len(beta)}"
        )
    shifted = latents.to(dtype=torch.float64).clone()
    scales = feature_scales.to(dtype=torch.float64)
    for local_i, feat_i in enumerate(selected_indices):
        idx = int(feat_i)
        shifted[..., idx] = shifted[..., idx] + scales[idx] * float(beta[local_i])
    return shifted


def build_margin_jacobian_fn(
    study: Any,
    stack: Any,
    *,
    corpus: Sequence[Mapping[str, object]] | None = None,
    split_question_ids: Mapping[str, Sequence[str]] | None = None,
    order_regime: str = "CF",
) -> Any:
    """Build ``(*, beta, question_ids, partitions) -> jac maps`` (DEC-084 / GRAD-015).

    Uses ``stack.margin_projection_batch`` when present; otherwise requires
    ``corpus`` + ``split_question_ids`` and ``compute_margin_projection_batch``.
    Live activity masks use ``1[z + sβ > 0]`` (GRAD-014), even when the batch
    returns baseline (unshifted) JumpReLU latents.

    Multi-layer pools project each SAE separately and scatter into the length-m
    jac row in ascending ``(layer, feature_id)`` pool order (DEC-054).
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

    default_layer = int(study.stack.sae.layers[0])
    layer_slices = _layer_pool_slices(
        study.experiment.feature_ids,
        default_layer=default_layer,
    )
    if not layer_slices:
        raise ValueError("margin jacobian requires nonempty experiment.feature_ids")

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
        if len(beta_t) != m:
            raise ValueError(
                f"beta length {len(beta_t)} != coefficient_length={m}"
            )
        n_selected = sum(len(entries) for entries in layer_slices.values())
        if n_selected != m:
            raise ValueError(
                f"pool feature count {n_selected} != coefficient_length={m}"
            )
        for belief in ("N", "IB", "CB"):
            layer_batches: dict[int, dict[str, Any]] = {}
            ref_qids: list[str] | None = None
            for layer in sorted(layer_slices):
                batch = batch_fn(
                    belief_condition=belief,
                    question_ids=qids,
                    beta=beta_t,
                    layer=int(layer),
                )
                batch_qids = [str(q) for q in batch["question_ids"]]
                if ref_qids is None:
                    ref_qids = batch_qids
                elif batch_qids != ref_qids:
                    raise ValueError(
                        f"margin projection question_ids mismatch across layers: "
                        f"layer={layer} got {batch_qids!r}, expected {ref_qids!r}"
                    )
                layer_batches[int(layer)] = batch
            assert ref_qids is not None
            for row_idx, qid in enumerate(ref_qids):
                jac_row = torch.zeros(m, dtype=torch.float64)
                for layer, entries in layer_slices.items():
                    batch = layer_batches[layer]
                    pool_indices = [pool_i for pool_i, _ in entries]
                    selected = [feat_i for _, feat_i in entries]
                    beta_slice = tuple(beta_t[pool_i] for pool_i in pool_indices)
                    latents = shift_latents_for_live_beta(
                        batch["latents"],
                        selected_indices=selected,
                        feature_scales=batch["feature_scales"],
                        beta=beta_slice,
                    )
                    partial = project_selected_margin_jacobian(
                        residual_gradient=batch["residual_gradients"][row_idx],
                        latents=latents[row_idx],
                        decoder=batch["decoder"],
                        feature_scales=batch["feature_scales"],
                        selected_indices=selected,
                        feature_chunk_size=chunk,
                    )
                    for local_i, pool_i in enumerate(pool_indices):
                        jac_row[pool_i] = partial[local_i]
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
    default_layer = int(study.stack.sae.layers[0])

    def margin_projection_batch(
        *,
        belief_condition: str,
        question_ids: Sequence[str],
        beta: Sequence[float],
        layer: int | None = None,
    ) -> dict[str, Any]:
        # Baseline JumpReLU z + unintervened ∂M/∂x. Live activity mask
        # 1[z + sβ > 0] is applied in build_margin_jacobian_fn (GRAD-014).
        _ = beta
        resolved_layer = int(default_layer if layer is None else layer)
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
            layer=resolved_layer,
            texts=[r.text for r in rows],
            question_ids=[r.question_id for r in rows],
            continuation_token_ids_A=token_a,
            continuation_token_ids_B=token_b,
            truthful_labels=[r.truthful_label for r in rows],
            prompt_batch_size=int(study.run.prompt_batch_size),
        )

    return margin_projection_batch
