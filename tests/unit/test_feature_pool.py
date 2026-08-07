"""Candidate-pool eligibility for suppression (Phase F FEAT-025+)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.feature_selection import (
    annotate_preservation_jacobians,
    eligible_suppression_candidates,
    rank_suppression_candidates,
)


@pytest.mark.unit
def test_feature_pool__suppression_only__excludes_nonpositive_behavior_jacobians_by_default() -> (
    None
):
    """FEAT-025: default eligibility is signed_jacobian > 0 (DEC-019)."""
    scores = {
        (0, 1): 2.0,
        (0, 2): 0.0,
        (1, 3): -1.0,
        (1, 4): 0.5,
    }
    eligible = eligible_suppression_candidates(
        signed_jacobians=scores,
        pool_eligibility_override=False,
    )
    assert eligible.pool_eligibility_override is False
    assert [(c.layer, c.feature_id) for c in eligible.candidates] == [(0, 1), (1, 4)]
    assert all(c.signed_jacobian > 0.0 for c in eligible.candidates)

    overridden = eligible_suppression_candidates(
        signed_jacobians=scores,
        pool_eligibility_override=True,
    )
    assert overridden.pool_eligibility_override is True
    assert len(overridden.candidates) == 4
    assert [(c.layer, c.feature_id) for c in overridden.candidates] == [
        (0, 1),
        (1, 4),
        (0, 2),
        (1, 3),
    ]


@pytest.mark.unit
def test_feature_pool__selected_behavior_features__retain_signed_preservation_jacobians() -> (
    None
):
    """FEAT-026: preservation Jacobians annotate selected features; never rank them.

    Behavior ranking uses resistance/recovery signed Jacobians only. Neutral and
    correct-surrogate sensitivities are retained as annotations for later
    filtering/interpretation (DEC-019).
    """
    behavior_scores = {
        (0, 1): 2.0,
        (0, 2): -3.0,  # large |J| but nonpositive → excluded by default
        (1, 4): 0.5,
    }
    neutral_jacobians = {
        (0, 1): -1.5,
        (0, 2): 9.0,  # would dominate if mixed into behavior rank
        (1, 4): 0.25,
    }
    correct_surrogate_jacobians = {
        (0, 1): 0.8,
        (0, 2): -4.0,
        (1, 4): -0.1,
    }

    eligible = eligible_suppression_candidates(
        signed_jacobians=behavior_scores,
        pool_eligibility_override=False,
    )
    annotated = annotate_preservation_jacobians(
        candidates=eligible.candidates,
        neutral_jacobians=neutral_jacobians,
        correct_surrogate_jacobians=correct_surrogate_jacobians,
    )

    assert [(c.layer, c.feature_id) for c in annotated] == [(0, 1), (1, 4)]
    by_key = {(c.layer, c.feature_id): c for c in annotated}
    assert by_key[(0, 1)].neutral_jacobian == -1.5
    assert by_key[(0, 1)].correct_surrogate_jacobian == 0.8
    assert by_key[(1, 4)].neutral_jacobian == 0.25
    assert by_key[(1, 4)].correct_surrogate_jacobian == -0.1

    # Behavior rank ignores preservation scores: mixing them into ranking must
    # not change the behavior order, and (0,2) stays out despite huge neutral J.
    behavior_rank = rank_suppression_candidates(signed_jacobians=behavior_scores)
    assert [(c.layer, c.feature_id) for c in behavior_rank] == [
        (0, 1),
        (1, 4),
        (0, 2),
    ]
    mixed_would_invert = rank_suppression_candidates(
        signed_jacobians={
            key: behavior_scores[key] + neutral_jacobians[key]
            for key in behavior_scores
        }
    )
    assert [(c.layer, c.feature_id) for c in mixed_would_invert] != [
        (c.layer, c.feature_id) for c in behavior_rank
    ]
    assert (0, 2) not in {(c.layer, c.feature_id) for c in annotated}


@pytest.mark.unit
def test_feature_pool__all_order_optimizers__receive_identical_feature_ids_scales_and_ordering() -> (
    None
):
    """FEAT-032: quota-union of supplied lists is deterministic (pool API).

    Under DEC-087 each study supplies one order's lists; the builder still
    accepts multi-order maps for pure unit coverage of the union rule.
    """
    from epistemic_sycophancy.feature_selection import build_common_feature_pool

    # Six lists: (CF/IF/RO) × (resistance/recovery). Distinct order-specific
    # positives; union must be identical for every optimizer study.
    lists = {
        ("CF", "resistance"): {(0, 2): 5.0, (1, 0): 3.0, (0, 1): -1.0},
        ("CF", "recovery"): {(0, 3): 4.0, (1, 1): 2.0},
        ("IF", "resistance"): {(0, 2): 1.0, (2, 0): 6.0},
        ("IF", "recovery"): {(1, 0): 2.5, (0, 4): 0.5},
        ("RO", "resistance"): {(0, 3): 1.0, (2, 1): 7.0},
        ("RO", "recovery"): {(1, 1): 3.0, (2, 0): 0.1},
    }
    scales = {
        (0, 1): 1.0,
        (0, 2): 2.0,
        (0, 3): 3.0,
        (0, 4): 4.0,
        (1, 0): 1.5,
        (1, 1): 2.5,
        (2, 0): 0.5,
        (2, 1): 1.25,
    }
    pool_cf = build_common_feature_pool(
        lists_by_order_and_component=lists,
        feature_scales=scales,
        pool_quota_per_list=8,
    )
    pool_if = build_common_feature_pool(
        lists_by_order_and_component=lists,
        feature_scales=scales,
        pool_quota_per_list=8,
    )
    pool_ro = build_common_feature_pool(
        lists_by_order_and_component=lists,
        feature_scales=scales,
        pool_quota_per_list=8,
    )
    assert pool_cf.feature_ids == pool_if.feature_ids == pool_ro.feature_ids
    assert pool_cf.scales == pool_if.scales == pool_ro.scales
    assert pool_cf.feature_ids == tuple(
        sorted(pool_cf.feature_ids, key=lambda k: (k[0], k[1]))
    )
    # All positive keys across the six lists (quota=8 takes all positives).
    expected = (
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    )
    assert pool_cf.feature_ids == expected
    assert pool_cf.scales == tuple(scales[k] for k in expected)


@pytest.mark.unit
def test_feature_pool__opposite_order_gradients__remain_eligible_under_quota_union() -> (
    None
):
    """FEAT-033: +CF / −IF feature stays in pool via CF quota; averaging cancels it."""
    from epistemic_sycophancy.feature_selection import build_common_feature_pool

    key = (0, 7)
    lists = {
        ("CF", "resistance"): {key: 4.0, (0, 1): 1.0},
        ("CF", "recovery"): {(0, 2): 1.0},
        ("IF", "resistance"): {key: -4.0, (0, 3): 1.0},  # opposite sign; mean 0
        ("IF", "recovery"): {(0, 4): 1.0},
        ("RO", "resistance"): {(0, 5): 1.0},
        ("RO", "recovery"): {(0, 6): 1.0},
    }
    scales = {k: 1.0 for k in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), key]}
    pool = build_common_feature_pool(
        lists_by_order_and_component=lists,
        feature_scales=scales,
        pool_quota_per_list=8,
    )
    assert key in pool.feature_ids

    # Averaging CF and IF Jacobians cancels the feature and would drop it.
    averaged = {
        k: (
            lists[("CF", "resistance")].get(k, 0.0)
            + lists[("IF", "resistance")].get(k, 0.0)
        )
        / 2.0
        for k in {key, (0, 1), (0, 3)}
    }
    assert averaged[key] == 0.0
    averaged_pool = build_common_feature_pool(
        lists_by_order_and_component={
            ("CF", "resistance"): averaged,
            ("CF", "recovery"): {},
            ("IF", "resistance"): {},
            ("IF", "recovery"): {},
            ("RO", "resistance"): {},
            ("RO", "recovery"): {},
        },
        feature_scales=scales,
        pool_quota_per_list=8,
    )
    assert key not in averaged_pool.feature_ids


@pytest.mark.unit
def test_feature_pool__quota_union_deduplication_and_fill__matches_frozen_policy() -> (
    None
):
    """FEAT-034: DEC-019 dedupe, shortfall, ties, fill=no-op, size=|union|."""
    from epistemic_sycophancy.feature_selection import build_common_feature_pool

    # Exact ties within a list: ascending (layer, feature_id) decides quota cut.
    # quota=2 keeps (0,1) and (0,2); (1,0) loses the tie for third place.
    tied_list = {(0, 2): 1.0, (1, 0): 1.0, (0, 1): 1.0}
    # Duplicate of (0, 1) across lists collapses to one entry.
    lists = {
        ("CF", "resistance"): tied_list,
        ("CF", "recovery"): {(0, 1): 5.0},  # duplicate of CF-resistance pick
        ("IF", "resistance"): {(0, 9): 2.0, (0, 8): -3.0},  # only one positive
        ("IF", "recovery"): {},  # shortfall: zero positives → contributes nothing
        ("RO", "resistance"): {(2, 0): 0.5},
        ("RO", "recovery"): {(2, 0): 0.5},  # duplicate across RO components
    }
    scales = {
        (0, 1): 1.0,
        (0, 2): 2.0,
        (0, 8): 1.0,
        (0, 9): 3.0,
        (1, 0): 1.5,
        (2, 0): 0.25,
    }
    pool = build_common_feature_pool(
        lists_by_order_and_component=lists,
        feature_scales=scales,
        pool_quota_per_list=2,
    )
    # Union of quota picks: CF-res top2={(0,1),(0,2)}; CF-rec={(0,1)};
    # IF-res={(0,9)} (nonpositive excluded, no pad); RO={(2,0)}.
    # (1,0) excluded by CF quota cut; (0,8) nonpositive never padded in.
    assert pool.feature_ids == ((0, 1), (0, 2), (0, 9), (2, 0))
    assert len(pool.feature_ids) == 4  # |union|, fill is a no-op
    assert (1, 0) not in pool.feature_ids
    assert (0, 8) not in pool.feature_ids
    assert pool.scales == (1.0, 2.0, 3.0, 0.25)


@pytest.mark.unit
def test_feature_pool__bidirectional__ranks_by_abs_jacobian_and_keeps_negative() -> None:
    """FEAT-025b / DEC-105: bidirectional pool uses |J| rank; includes J<0.

    Hand-derived with quota=2 per list:
      resistance: (0,1)=+2.0, (0,2)=-3.0, (0,3)=+0.5
        top-|J|: (0,2) then (0,1); (0,3) cut
      recovery: (1,0)=-1.5, (1,1)=+1.0
        top-|J|: (1,0) then (1,1)
      union ascending (layer,fid): (0,1),(0,2),(1,0),(1,1)
      preferred signs: -sign(J) → -1, +1, +1, -1
    """
    from epistemic_sycophancy.feature_selection import build_common_feature_pool

    lists = {
        ("CF", "resistance"): {(0, 1): 2.0, (0, 2): -3.0, (0, 3): 0.5},
        ("CF", "recovery"): {(1, 0): -1.5, (1, 1): 1.0},
    }
    scales = {(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0, (1, 0): 1.5, (1, 1): 0.5}
    suppression = build_common_feature_pool(
        lists_by_order_and_component=lists,
        feature_scales=scales,
        pool_quota_per_list=2,
        coefficient_mode="suppression",
    )
    # Suppression keeps only J>0: res top2=(0,1),(0,3); rec=(1,1)
    assert suppression.feature_ids == ((0, 1), (0, 3), (1, 1))

    bidirectional = build_common_feature_pool(
        lists_by_order_and_component=lists,
        feature_scales=scales,
        pool_quota_per_list=2,
        coefficient_mode="bidirectional",
    )
    assert bidirectional.feature_ids == ((0, 1), (0, 2), (1, 0), (1, 1))
    assert bidirectional.scales == (1.0, 2.0, 1.5, 0.5)
    assert bidirectional.preferred_bidirectional_signs == (-1.0, 1.0, 1.0, -1.0)
    # Negative-J excitation candidate present; zero-J never eligible.
    assert (0, 2) in bidirectional.feature_ids
    assert (0, 2) not in suppression.feature_ids

