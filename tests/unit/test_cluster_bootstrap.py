"""Question-cluster bootstrap tests (Phase I STAT)."""

from __future__ import annotations

from collections import Counter

import pytest

from epistemic_sycophancy.statistics.cluster_bootstrap import sample_question_clusters


@pytest.mark.unit
def test_cluster_bootstrap__samples_question_ids_not_prompt_rows() -> None:
    """STAT-001: bootstrap samples question IDs; all variants travel together.

    Toy clusters have unequal variant counts. Sampling at the prompt-row level
    would freely mix variants across questions; sampling question IDs must keep
    each question's full variant set intact whenever that question is drawn.
    """
    clusters = {
        "q1": ("ib_a", "ib_b", "ib_c"),
        "q2": ("ib_only",),
    }
    sampled = sample_question_clusters(
        clusters,
        n_samples=4,
        seed=0,
    )
    assert len(sampled) == 4
    for question_id, variants in sampled:
        assert question_id in clusters
        assert variants == clusters[question_id]
        assert len(variants) == len(clusters[question_id])

    # Sampling unit is question_id: each draw is one complete cluster, not a
    # single prompt row. Prompt-row sampling would return length-1 items from
    # the flattened variant pool of size 4.
    assert all(len(variants) == len(clusters[qid]) for qid, variants in sampled)
    sampled_ids = [qid for qid, _ in sampled]
    assert set(sampled_ids).issubset({"q1", "q2"})
    # With replacement over 2 questions and 4 draws, multiplicity is allowed.
    assert sum(Counter(sampled_ids).values()) == 4


@pytest.mark.unit
def test_cluster_bootstrap__duplicate_sampled_question__duplicates_complete_question_cluster() -> None:
    """STAT-002: with-replacement multiplicity duplicates the entire cluster.

    Hand-check: when question q1 (3 variants) appears twice in the sampled ID
    multiset, both draws must carry the full 3-variant tuple — not a partial
    or prompt-row subset.
    """
    clusters = {
        "q1": ("v1", "v2", "v3"),
        "q2": ("only",),
    }
    # Inject an explicit ID multiset with a known duplicate of q1.
    sampled = sample_question_clusters(
        clusters,
        n_samples=3,
        seed=0,
        sample_question_ids=["q1", "q2", "q1"],
    )
    assert len(sampled) == 3
    assert [qid for qid, _ in sampled] == ["q1", "q2", "q1"]
    assert sampled[0] == ("q1", ("v1", "v2", "v3"))
    assert sampled[2] == ("q1", ("v1", "v2", "v3"))
    assert sampled[1] == ("q2", ("only",))
    # Multiplicity of the complete cluster matches sample multiplicity.
    q1_clusters = [variants for qid, variants in sampled if qid == "q1"]
    assert len(q1_clusters) == 2
    assert all(variants == ("v1", "v2", "v3") for variants in q1_clusters)


@pytest.mark.unit
def test_cluster_bootstrap__paired_change__uses_same_sampled_question_ids_for_both_conditions() -> None:
    """STAT-003: paired Δ uses identical sampled question IDs for both arms.

    Baseline and intervention must receive the same question-ID multiset in
    each replicate so the paired change is well-defined.
    """
    from epistemic_sycophancy.statistics.cluster_bootstrap import paired_cluster_resample

    question_ids = ("q1", "q2", "q3")
    baseline_values = {"q1": 1.0, "q2": 2.0, "q3": 3.0}
    intervention_values = {"q1": 1.5, "q2": 2.5, "q3": 3.5}
    paired = paired_cluster_resample(
        question_ids=question_ids,
        baseline_by_question=baseline_values,
        intervention_by_question=intervention_values,
        n_samples=5,
        seed=0,
        sample_question_ids=["q2", "q1", "q2", "q3", "q1"],
    )
    assert paired.sampled_question_ids == ("q2", "q1", "q2", "q3", "q1")
    assert paired.baseline_values == (2.0, 1.0, 2.0, 3.0, 1.0)
    assert paired.intervention_values == (2.5, 1.5, 2.5, 3.5, 1.5)
    # Same ID multiset for both conditions (pairing invariant).
    assert len(paired.baseline_values) == len(paired.intervention_values)
    assert len(paired.sampled_question_ids) == 5


@pytest.mark.unit
def test_cluster_bootstrap__selectivity_interval__recomputes_ftw_and_cbr_in_each_replicate() -> None:
    """STAT-004: each replicate recomputes FTW and CBR, then Selectivity=CBR−FTW.

    Must not bootstrap a precomputed Selectivity column. Hand-check: full-sample
    replicate recovers golden Selectivity = 5/12 from FTW=0.25 and CBR=2/3.
    """
    from tests.fixtures.metrics.golden_behavioral import (
        GOLDEN_CBR,
        GOLDEN_CURRENT_CB_MARGINS,
        GOLDEN_CURRENT_IB_MARGINS,
        GOLDEN_CURRENT_NEUTRAL_MARGINS,
        GOLDEN_FTW,
        GOLDEN_SELECTIVITY,
    )

    from epistemic_sycophancy.metrics.baseline_partition import build_baseline_partition
    from epistemic_sycophancy.statistics.cluster_bootstrap import (
        bootstrap_selectivity_interval,
    )

    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins={"q1": 2.0, "q2": -1.0, "q3": 0.5},
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    # One replicate that draws each question once → golden metrics.
    result = bootstrap_selectivity_interval(
        frozen_partition=partition,
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        current_ib_margins=GOLDEN_CURRENT_IB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        epsilon=1e-6,
        n_replicates=1,
        seed=0,
        bootstrap_ci_percentile=95.0,
        replicate_sample_ids=[["q1", "q2", "q3"]],
    )
    assert len(result.replicate_ftw) == 1
    assert len(result.replicate_cbr) == 1
    assert len(result.replicate_selectivity) == 1
    assert result.replicate_ftw[0] == pytest.approx(GOLDEN_FTW)
    assert result.replicate_cbr[0] == pytest.approx(GOLDEN_CBR)
    assert result.replicate_selectivity[0] == pytest.approx(GOLDEN_SELECTIVITY)
    # Selectivity is recomputed from FTW/CBR, not an independent precomputed column.
    assert result.replicate_selectivity[0] == pytest.approx(
        result.replicate_cbr[0] - result.replicate_ftw[0]
    )


@pytest.mark.unit
def test_cluster_bootstrap__fixed_seed__reproduces_replicates_and_ci() -> None:
    """STAT-005: explicit seed reproduces replicate metrics and CI bounds."""
    from tests.fixtures.metrics.golden_behavioral import (
        GOLDEN_CURRENT_CB_MARGINS,
        GOLDEN_CURRENT_IB_MARGINS,
        GOLDEN_CURRENT_NEUTRAL_MARGINS,
    )

    from epistemic_sycophancy.metrics.baseline_partition import build_baseline_partition
    from epistemic_sycophancy.statistics.cluster_bootstrap import (
        bootstrap_selectivity_interval,
    )

    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins={"q1": 2.0, "q2": -1.0, "q3": 0.5},
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    kwargs = dict(
        frozen_partition=partition,
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        current_ib_margins=GOLDEN_CURRENT_IB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        epsilon=1e-6,
        n_replicates=20,
        seed=0,
        bootstrap_ci_percentile=95.0,
    )
    a = bootstrap_selectivity_interval(**kwargs)
    b = bootstrap_selectivity_interval(**kwargs)
    assert a.replicate_ftw == b.replicate_ftw
    assert a.replicate_cbr == b.replicate_cbr
    assert a.replicate_selectivity == b.replicate_selectivity
    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high


@pytest.mark.unit
def test_cluster_bootstrap__constant_question_effects__produce_zero_width_interval() -> None:
    """STAT-007: constant question-level effects ⇒ zero-width CI (atol 1e-12)."""
    from epistemic_sycophancy.metrics.baseline_partition import build_baseline_partition
    from epistemic_sycophancy.statistics.cluster_bootstrap import (
        bootstrap_selectivity_interval,
    )

    # Every question has identical IB/CB pattern so every replicate Selectivity
    # is identical → CI width must be zero.
    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins={"q1": 1.0, "q2": -1.0, "q3": 1.0},
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    result = bootstrap_selectivity_interval(
        frozen_partition=partition,
        current_neutral_margins={"q1": 1.0, "q2": -1.0, "q3": 1.0},
        current_ib_margins={"q1": [-1.0], "q2": [-1.0], "q3": [-1.0]},
        current_cb_margins={"q1": [1.0], "q2": [1.0], "q3": [1.0]},
        epsilon=1e-6,
        n_replicates=20,
        seed=0,
        bootstrap_ci_percentile=95.0,
    )
    width = result.ci_high - result.ci_low
    assert width == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_statistics__public_api__does_not_accept_prompt_row_as_default_resampling_unit() -> None:
    """STAT-008: public statistics API rejects prompt-row as resampling unit."""
    from epistemic_sycophancy.statistics import sample_question_clusters
    from epistemic_sycophancy.statistics.cluster_bootstrap import sample_question_clusters as sample_fn

    clusters = {"q1": ("a", "b"), "q2": ("c",)}
    # Keyword resampling_unit must not default to / accept "prompt_row".
    with pytest.raises((TypeError, ValueError)):
        sample_fn(  # type: ignore[call-arg]
            clusters,
            n_samples=2,
            seed=0,
            resampling_unit="prompt_row",
        )
    # Public export is question-cluster sampling only.
    assert sample_question_clusters is sample_fn


@pytest.mark.unit
def test_cluster_bootstrap__conditional_metrics__retain_or_recompute_valid_denominators() -> None:
    """STAT-009 / DEC-038: empty Q+ or Q− replicate is invalid; never substitute 0.

    Fixture: Q+={q1}, Q−={q2}. A replicate that samples only q1 empties Q−;
    that replicate must be counted invalid and excluded from the CI values.
    """
    from epistemic_sycophancy.metrics.baseline_partition import build_baseline_partition
    from epistemic_sycophancy.statistics.cluster_bootstrap import (
        bootstrap_selectivity_interval,
    )

    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins={"q1": 1.0, "q2": -1.0},
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    result = bootstrap_selectivity_interval(
        frozen_partition=partition,
        current_neutral_margins={"q1": 1.0, "q2": -1.0},
        current_ib_margins={"q1": [-1.0], "q2": [-1.0]},
        current_cb_margins={"q1": [1.0], "q2": [1.0]},
        epsilon=1e-6,
        n_replicates=3,
        seed=0,
        bootstrap_ci_percentile=95.0,
        replicate_sample_ids=[
            ["q1", "q1"],  # empty Q− → invalid
            ["q1", "q2"],  # valid
            ["q2", "q2"],  # empty Q+ → invalid
        ],
    )
    assert result.n_invalid_replicates == 2
    assert len(result.replicate_selectivity) == 1
    assert len(result.replicate_ftw) == 1
    assert len(result.replicate_cbr) == 1
    # Valid replicate must not be a silent zero substitute for the invalids.
    assert result.replicate_ftw[0] != 0.0 or result.replicate_cbr[0] != 0.0
    # Selectivity still equals recomputed CBR − FTW on the valid replicate.
    assert result.replicate_selectivity[0] == pytest.approx(
        result.replicate_cbr[0] - result.replicate_ftw[0]
    )
