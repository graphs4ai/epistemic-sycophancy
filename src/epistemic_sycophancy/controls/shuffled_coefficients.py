"""Shuffled-coefficient controls (Phase I CTRL)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def shuffle_coefficients(
    *,
    feature_ids: Sequence[int],
    betas: Sequence[float],
    seed: int,
    require_nontrivial: bool = False,
    max_reshuffles: int = 100,
) -> list[float]:
    """Permute β across selected features (DEC-040 / CTRL-004/005).

    Preserves the exact coefficient multiset. When ``require_nontrivial`` is
    True and β is non-constant, reshuffle until at least one assignment differs
    from the input (bounded by ``max_reshuffles``).
    """
    del feature_ids  # length/alignment check only
    values = [float(x) for x in betas]
    if len(values) < 1:
        raise ValueError("betas must be non-empty")
    rng = np.random.default_rng(seed)
    shuffled = list(values)
    attempts = max_reshuffles if require_nontrivial else 1
    for _ in range(attempts):
        rng.shuffle(shuffled)
        if not require_nontrivial or shuffled != values:
            return shuffled
        shuffled = list(values)
    if len(set(values)) == 1:
        raise ValueError("cannot produce nontrivial shuffle of constant betas")
    raise RuntimeError("failed to obtain nontrivial coefficient shuffle")
