"""Shuffled-coefficient control tests (Phase I CTRL)."""

from __future__ import annotations

from collections import Counter

import pytest

from epistemic_sycophancy.controls.shuffled_coefficients import shuffle_coefficients


@pytest.mark.unit
def test_shuffled_coefficient_control__preserves_exact_coefficient_multiset() -> None:
    """CTRL-004: shuffle permutes assignment; coefficient multiset unchanged."""
    feature_ids = [10, 20, 30]
    betas = [-1.0, -0.5, 0.0]
    shuffled = shuffle_coefficients(
        feature_ids=feature_ids,
        betas=betas,
        seed=0,
    )
    assert Counter(shuffled) == Counter(betas)
    assert sorted(shuffled) == sorted(betas)


@pytest.mark.unit
def test_shuffled_coefficient_control__nontrivial_vector__changes_at_least_one_feature_assignment() -> None:
    """CTRL-005: nontrivial shuffle changes ≥1 feature↔coefficient assignment."""
    feature_ids = [10, 20, 30]
    betas = [-1.0, -0.5, 0.0]
    shuffled = shuffle_coefficients(
        feature_ids=feature_ids,
        betas=betas,
        seed=0,
        require_nontrivial=True,
    )
    assert Counter(shuffled) == Counter(betas)
    assert shuffled != betas
