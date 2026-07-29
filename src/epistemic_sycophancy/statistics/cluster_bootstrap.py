"""Question-cluster bootstrap (Phase I STAT)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def sample_question_clusters(
    clusters: Mapping[str, Sequence[Any]],
    *,
    n_samples: int,
    seed: int,
    sample_question_ids: Sequence[str] | None = None,
) -> list[tuple[str, tuple[Any, ...]]]:
    """Bootstrap-sample question IDs with replacement; return full clusters.

    DEC-037: explicit seed; resampling unit is always question_id.
    When a question is sampled, all of its variants travel together (STAT-001).
    Duplicate IDs duplicate the complete cluster (STAT-002).

    If ``sample_question_ids`` is provided, use that multiset instead of RNG
    draws (length must equal ``n_samples``); ``seed`` is ignored in that case.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if not clusters:
        raise ValueError("clusters must be non-empty")
    if sample_question_ids is not None:
        if len(sample_question_ids) != n_samples:
            raise ValueError("sample_question_ids length must equal n_samples")
        drawn = [str(qid) for qid in sample_question_ids]
        for qid in drawn:
            if qid not in clusters:
                raise KeyError(f"unknown question_id in sample: {qid!r}")
    else:
        question_ids = sorted(clusters.keys())
        rng = np.random.default_rng(seed)
        drawn = [str(qid) for qid in rng.choice(question_ids, size=n_samples, replace=True)]
    return [(qid, tuple(clusters[qid])) for qid in drawn]


@dataclass(frozen=True)
class PairedClusterResample:
    """Paired baseline/intervention values under one shared question-ID sample."""

    sampled_question_ids: tuple[str, ...]
    baseline_values: tuple[float, ...]
    intervention_values: tuple[float, ...]


def paired_cluster_resample(
    *,
    question_ids: Sequence[str],
    baseline_by_question: Mapping[str, float],
    intervention_by_question: Mapping[str, float],
    n_samples: int,
    seed: int,
    sample_question_ids: Sequence[str] | None = None,
) -> PairedClusterResample:
    """Resample question IDs once; index both conditions with the same IDs (STAT-003)."""
    clusters = {qid: (qid,) for qid in question_ids}
    sampled = sample_question_clusters(
        clusters,
        n_samples=n_samples,
        seed=seed,
        sample_question_ids=sample_question_ids,
    )
    ids = tuple(qid for qid, _ in sampled)
    return PairedClusterResample(
        sampled_question_ids=ids,
        baseline_values=tuple(float(baseline_by_question[qid]) for qid in ids),
        intervention_values=tuple(float(intervention_by_question[qid]) for qid in ids),
    )
