"""RUN-011: common pool keys align with CFG β layout and per-layer scatter."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.feature_selection.pool import build_common_feature_pool
from epistemic_sycophancy.stack.beta_layout import (
    align_cfg_feature_ids_with_pool,
    scatter_beta_by_layer,
)


@pytest.mark.unit
def test_stack__common_pool__feature_keys_and_scales_align_with_cfg_beta_layout() -> None:
    """RUN-011: pool/CFG keys are (layer, feature_id); scatter groups by layer."""
    lists = {
        ("CF", "resistance"): {(9, 2): 1.0, (17, 1): 0.5},
        ("CF", "recovery"): {(9, 2): 0.2},
        ("IF", "resistance"): {(17, 1): 0.9},
        ("IF", "recovery"): {(22, 0): 1.1},
        ("RO", "resistance"): {(9, 2): 0.3},
        ("RO", "recovery"): {(22, 0): 0.4},
    }
    scales_map = {(9, 2): 1.0, (17, 1): 2.0, (22, 0): 0.5}
    pool = build_common_feature_pool(
        lists_by_order_and_component=lists,
        feature_scales=scales_map,
        pool_quota_per_list=8,
    )
    assert all(isinstance(k, tuple) and len(k) == 2 for k in pool.feature_ids)
    assert len(pool.feature_ids) == len(pool.scales)

    cfg_ids, cfg_scales = align_cfg_feature_ids_with_pool(pool)
    assert cfg_ids == pool.feature_ids
    assert cfg_scales == pool.scales

    beta = tuple(-0.1 * (i + 1) for i in range(len(cfg_ids)))
    scattered = scatter_beta_by_layer(
        feature_ids=cfg_ids,
        scales=cfg_scales,
        beta=beta,
    )
    assert set(scattered.keys()) == {layer for layer, _ in cfg_ids}
    for layer, payload in scattered.items():
        assert len(payload.selected_indices) == len(payload.scales) == len(payload.beta)
        assert all(isinstance(i, int) for i in payload.selected_indices)


@pytest.mark.unit
def test_stack__common_pool_scales__decoder_norm_from_sae_metadata() -> None:
    """WIRE-008: decoder_norm scales from SAE decoder rows (DEC-061)."""
    import torch
    from types import SimpleNamespace

    from epistemic_sycophancy.stack.scales import scales_for_layer_feature_keys

    decoder = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    saes = {17: SimpleNamespace(decoder_weight=decoder)}
    keys = ((17, 0), (17, 1))
    scales = scales_for_layer_feature_keys(
        keys=keys, saes=saes, scale_source="decoder_norm"
    )
    assert scales == (3.0, 4.0)
