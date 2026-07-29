"""Random-feature control tests (Phase I CTRL)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.controls.random_features import sample_random_features


@pytest.mark.unit
def test_random_feature_control__matches_selected_feature_count() -> None:
    """CTRL-001: random-feature control has the same cardinality as selected."""
    selected = [1, 3]
    random_ids = sample_random_features(
        n_features=10,
        selected_feature_ids=selected,
        control_seed=0,
        allow_overlap=False,
    )
    assert len(random_ids) == len(selected)


@pytest.mark.unit
def test_random_feature_control__has_no_overlap_unless_explicitly_permitted() -> None:
    """CTRL-002: default no overlap with selected; overlap only if allow_overlap=True."""
    selected = [0, 1, 2]
    no_overlap = sample_random_features(
        n_features=10,
        selected_feature_ids=selected,
        control_seed=1,
        allow_overlap=False,
    )
    assert set(no_overlap).isdisjoint(set(selected))

    with_overlap = sample_random_features(
        n_features=3,
        selected_feature_ids=selected,
        control_seed=2,
        allow_overlap=True,
    )
    assert len(with_overlap) == len(selected)
    assert set(with_overlap).issubset({0, 1, 2})
