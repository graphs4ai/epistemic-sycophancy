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
