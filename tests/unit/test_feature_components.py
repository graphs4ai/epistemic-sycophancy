"""Feature-selection loss components and their frozen subsets (Phase F)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from epistemic_sycophancy.feature_selection import selection_component_prompts
from epistemic_sycophancy.objective.losses import baseline_relative_hinge

_TOY_COMPONENTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "feature_selection"
    / "toy_components.py"
)
_spec = importlib.util.spec_from_file_location(
    "toy_components_unit", _TOY_COMPONENTS_PATH
)
assert _spec is not None and _spec.loader is not None
_toy_components = importlib.util.module_from_spec(_spec)
# Register before executing so the fixture's dataclasses can resolve their
# defining module.
sys.modules[_spec.name] = _toy_components
_spec.loader.exec_module(_toy_components)
frozen_partition = _toy_components.frozen_partition
prompt_rows = _toy_components.prompt_rows


@pytest.mark.unit
@pytest.mark.parametrize(
    ("component", "expected_condition", "expected_questions"),
    [
        ("resistance", "IB", {"q1", "q3"}),
        ("recovery", "CB", {"q2"}),
        ("neutral_surrogate", "N", {"q1", "q2", "q3"}),
        ("correct_surrogate", "CB", {"q1", "q3"}),
    ],
)
def test_feature_components__use_correct_conditions_and_frozen_question_subsets(
    component: str,
    expected_condition: str,
    expected_questions: set[str],
) -> None:
    """FEAT-010: each component reads one condition over one frozen subset."""
    partition = frozen_partition()

    selected = selection_component_prompts(
        component=component,
        prompt_rows=prompt_rows(),
        partition=partition,
    )

    assert {row.condition for row in selected} == {expected_condition}
    assert {row.question_id for row in selected} == expected_questions
    assert set(selected) == {
        row
        for row in prompt_rows()
        if row.condition == expected_condition
        and row.question_id in expected_questions
    }


@pytest.mark.unit
def test_feature_components__baseline_relative_hinges__have_zero_null_gradient() -> None:
    """FEAT-011: hinge [M0 - M(β) - δ]_+ is flat at β=0 when δ > 0.

    At the null intervention M(β)=M0, so the hinge is exactly zero and its
    local gradient vanishes. Using these hinges for null-intervention ranking
    would produce an all-zero ranking and is therefore forbidden.
    """
    delta_n = 0.25
    delta_c = 0.10
    baseline_neutral = 2.0
    baseline_correct = 1.5

    margin = torch.tensor(baseline_neutral, dtype=torch.float64, requires_grad=True)
    hinge = baseline_relative_hinge(
        baseline_margin=baseline_neutral,
        current_margin=margin,
        delta=delta_n,
    )
    assert float(hinge.item()) == 0.0
    (grad,) = torch.autograd.grad(hinge, margin, allow_unused=True)
    assert grad is None or float(grad.item()) == 0.0

    margin_c = torch.tensor(baseline_correct, dtype=torch.float64, requires_grad=True)
    hinge_c = baseline_relative_hinge(
        baseline_margin=baseline_correct,
        current_margin=margin_c,
        delta=delta_c,
    )
    assert float(hinge_c.item()) == 0.0
    (grad_c,) = torch.autograd.grad(hinge_c, margin_c, allow_unused=True)
    assert grad_c is None or float(grad_c.item()) == 0.0

    # Outside the flat region the hinge is informative (sanity on the formula).
    dropped = torch.tensor(
        baseline_neutral - delta_n - 0.5, dtype=torch.float64, requires_grad=True
    )
    hinge_dropped = baseline_relative_hinge(
        baseline_margin=baseline_neutral,
        current_margin=dropped,
        delta=delta_n,
    )
    assert float(hinge_dropped.item()) == pytest.approx(0.5)
    (grad_dropped,) = torch.autograd.grad(hinge_dropped, dropped)
    assert float(grad_dropped.item()) == pytest.approx(-1.0)


@pytest.mark.unit
def test_feature_selection__separate_backward_components__do_not_mix_gradients() -> None:
    """FEAT-023: a resistance ranking excludes recovery/neutral/correct terms."""
    from epistemic_sycophancy.feature_selection import isolate_component_jacobian

    resistance = {(0, 1): 2.0, (0, 2): -1.0}
    recovery = {(0, 1): 9.0, (0, 2): 9.0}
    neutral = {(0, 1): -3.0, (0, 2): 4.0}
    correct = {(0, 1): 5.0, (0, 2): 5.0}

    isolated = isolate_component_jacobian(
        component="resistance",
        component_jacobians={
            "resistance": resistance,
            "recovery": recovery,
            "neutral_surrogate": neutral,
            "correct_surrogate": correct,
        },
    )
    assert isolated == resistance
    assert isolated[(0, 1)] != recovery[(0, 1)]

    with pytest.raises(ValueError, match="composite"):
        isolate_component_jacobian(
            component="resistance+recovery",
            component_jacobians={
                "resistance": resistance,
                "recovery": recovery,
            },
        )
