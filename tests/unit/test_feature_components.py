"""Feature-selection loss components and their frozen subsets (Phase F)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from epistemic_sycophancy.feature_selection import selection_component_prompts

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
