"""Question-cluster bootstrap (Phase I STAT)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def sample_question_clusters(
    clusters: Mapping[str, Sequence[Any]],
    *,
    n_samples: int,
    seed: int,
) -> list[tuple[str, tuple[Any, ...]]]:
    """Bootstrap-sample question IDs with replacement; return full clusters.

    DEC-037: explicit seed; resampling unit is always question_id.
    When a question is sampled, all of its variants travel together (STAT-001).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if not clusters:
        raise ValueError("clusters must be non-empty")
    question_ids = sorted(clusters.keys())
    rng = np.random.default_rng(seed)
    drawn = rng.choice(question_ids, size=n_samples, replace=True)
    return [
        (str(qid), tuple(clusters[str(qid)]))
        for qid in drawn
    ]
