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
