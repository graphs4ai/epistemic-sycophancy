"""Trial / objective logging package + operational pipeline logger (DEC-089)."""

from epistemic_sycophancy.logging.loss_curve import plot_loss_over_trials
from epistemic_sycophancy.logging.pipeline import (
    PIPELINE_LOGGER_NAME,
    configure_pipeline_logging,
    get_pipeline_logger,
    log_audit,
    log_progress,
    log_stage_end,
    log_stage_start,
)
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
    "PIPELINE_LOGGER_NAME",
    "TrialRecord",
    "build_objective_components",
    "build_trial_record",
    "configure_pipeline_logging",
    "get_pipeline_logger",
    "log_audit",
    "log_progress",
    "log_stage_end",
    "log_stage_start",
    "plot_loss_over_trials",
]
