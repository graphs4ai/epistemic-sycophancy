"""Trial / objective logging package + operational pipeline logger (DEC-089)."""

from epistemic_sycophancy.logging.full_study_plots import (
    BEHAVIORAL_PLOT_METRICS,
    ordered_behavioral_labels,
    plot_behavioral_metric_bars,
    plot_ib_mean_favorable_delta,
    plot_margins_delta_hist_l_total,
    plot_margins_scatter_l_total,
    write_full_study_figures,
)
from epistemic_sycophancy.logging.loss_curve import plot_loss_over_trials
from epistemic_sycophancy.logging.optimize_metrics import (
    ITERATION_CSV_COLUMNS,
    ITERATION_PLOT_METRICS,
    STEP_CSV_COLUMNS,
    count_betas_at_bounds,
    plot_iteration_metric_curves,
    write_optimize_metrics_csv,
)
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
    OBJECTIVE_VERSION_CURRENT,
    OBJECTIVE_VERSION_V1,
    OBJECTIVE_VERSION_V2,
    ObjectiveComponents,
    TrialRecord,
    build_objective_components,
    build_trial_record,
)

__all__ = [
    "BEHAVIORAL_PLOT_METRICS",
    "OBJECTIVE_VERSION_CURRENT",
    "OBJECTIVE_VERSION_V1",
    "OBJECTIVE_VERSION_V2",
    "ITERATION_CSV_COLUMNS",
    "ITERATION_PLOT_METRICS",
    "ObjectiveComponents",
    "PIPELINE_LOGGER_NAME",
    "STEP_CSV_COLUMNS",
    "TrialRecord",
    "build_objective_components",
    "build_trial_record",
    "configure_pipeline_logging",
    "count_betas_at_bounds",
    "get_pipeline_logger",
    "log_audit",
    "log_progress",
    "log_stage_end",
    "log_stage_start",
    "ordered_behavioral_labels",
    "plot_behavioral_metric_bars",
    "plot_ib_mean_favorable_delta",
    "plot_iteration_metric_curves",
    "plot_loss_over_trials",
    "plot_margins_delta_hist_l_total",
    "plot_margins_scatter_l_total",
    "write_full_study_figures",
    "write_optimize_metrics_csv",
]
