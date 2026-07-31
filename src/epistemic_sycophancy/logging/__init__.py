"""Trial / objective logging package."""

from epistemic_sycophancy.logging.loss_curve import plot_loss_over_trials
from epistemic_sycophancy.logging.trial_records import (
    OBJECTIVE_VERSION_V1,
    ObjectiveComponents,
    TrialRecord,
    build_objective_components,
    build_trial_record,
)

__all__ = [
    "OBJECTIVE_VERSION_V1",
    "ObjectiveComponents",
    "TrialRecord",
    "build_objective_components",
    "build_trial_record",
    "plot_loss_over_trials",
]
