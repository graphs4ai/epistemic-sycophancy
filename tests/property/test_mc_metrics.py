"""Property tests for MC1/MC2 invariants (MC-004…006)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from epistemic_sycophancy.scoring.mc import mc1_evaluate, mc2_truthful_mass

_finite = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=64,
    min_value=-20.0,
    max_value=20.0,
)
_scores = st.lists(_finite, min_size=1, max_size=5)


@pytest.mark.property
@given(truthful=_scores, false=_scores)
@settings(max_examples=50)
def test_mc2__score__is_between_zero_and_one(
    truthful: list[float],
    false: list[float],
) -> None:
    """MC-004: MC2 ∈ [0, 1]."""
    mass = mc2_truthful_mass(truthful_scores=truthful, false_scores=false)
    assert 0.0 <= mass <= 1.0


@pytest.mark.property
@given(truthful=_scores, false=_scores, shift=_finite)
@settings(max_examples=50)
def test_mc2__adding_constant_to_all_candidate_scores__does_not_change_mass(
    truthful: list[float],
    false: list[float],
    shift: float,
) -> None:
    """MC-005: adding a constant to all scores leaves MC2 unchanged."""
    base = mc2_truthful_mass(truthful_scores=truthful, false_scores=false)
    shifted = mc2_truthful_mass(
        truthful_scores=[s + shift for s in truthful],
        false_scores=[s + shift for s in false],
    )
    assert shifted == pytest.approx(base, abs=1e-9, rel=1e-9)


@pytest.mark.property
@given(
    scores=st.lists(_finite, min_size=2, max_size=6),
    data=st.data(),
)
@settings(max_examples=40)
def test_mc_metrics__permuting_candidate_rows__does_not_change_result(
    scores: list[float],
    data: st.DataObject,
) -> None:
    """MC-006: candidate-order permutation leaves MC1/MC2 unchanged."""
    n = len(scores)
    truthful_index = data.draw(st.integers(min_value=0, max_value=n - 1))
    # Build false indices as all others for MC2 split
    truthful_scores = [scores[truthful_index]]
    false_scores = [s for i, s in enumerate(scores) if i != truthful_index]
    if not false_scores:
        return
    base_mc1 = mc1_evaluate(
        candidate_scores=scores,
        truthful_indices=[truthful_index],
        mc1_tie_policy="fail_and_report",
    )
    base_mc2 = mc2_truthful_mass(
        truthful_scores=truthful_scores,
        false_scores=false_scores,
    )
    order = list(range(n))
    perm = data.draw(st.permutations(order))
    permuted_scores = [scores[i] for i in perm]
    # Map truthful index through permutation
    new_truthful = perm.index(truthful_index)
    perm_mc1 = mc1_evaluate(
        candidate_scores=permuted_scores,
        truthful_indices=[new_truthful],
        mc1_tie_policy="fail_and_report",
    )
    perm_truthful_scores = [permuted_scores[new_truthful]]
    perm_false_scores = [
        s for i, s in enumerate(permuted_scores) if i != new_truthful
    ]
    perm_mc2 = mc2_truthful_mass(
        truthful_scores=perm_truthful_scores,
        false_scores=perm_false_scores,
    )
    assert perm_mc1.success == base_mc1.success
    assert perm_mc1.n_mc1_top_ties == base_mc1.n_mc1_top_ties
    assert perm_mc2 == pytest.approx(base_mc2, abs=1e-9, rel=1e-9)
