"""Random-feature controls (Phase I CTRL)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def sample_random_features(
    *,
    n_features: int,
    selected_feature_ids: Sequence[int],
    control_seed: int,
    allow_overlap: bool = False,
) -> list[int]:
    """Sample a random-feature control set (DEC-039 / CTRL-001…003).

    Default: sample without replacement from ``[0, n_features)`` excluding
    selected IDs. Overlap allowed only when ``allow_overlap=True``.
    """
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    selected = [int(x) for x in selected_feature_ids]
    k = len(selected)
    if k < 1:
        raise ValueError("selected_feature_ids must be non-empty")
    universe = list(range(n_features))
    if allow_overlap:
        pool = universe
    else:
        selected_set = set(selected)
        pool = [i for i in universe if i not in selected_set]
        if len(pool) < k:
            raise ValueError(
                "insufficient non-selected features for random control without overlap"
            )
    rng = np.random.default_rng(control_seed)
    chosen = rng.choice(pool, size=k, replace=False)
    return [int(x) for x in chosen]
