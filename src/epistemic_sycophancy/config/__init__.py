"""Experiment configuration schema and validation."""

from epistemic_sycophancy.config.frozen import (
    ConfigImmutabilityError,
    FrozenExperimentConfig,
    freeze_experiment_config,
    mark_holdout_started,
)
from epistemic_sycophancy.config.schema import ExperimentConfig, InvalidExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudySmokeConfig,
    build_study_config,
)

__all__ = [
    "ConfigImmutabilityError",
    "ExperimentConfig",
    "FrozenExperimentConfig",
    "InvalidExperimentConfig",
    "StudyConfig",
    "StudyOptimizerConfig",
    "StudyRunConfig",
    "StudySmokeConfig",
    "build_study_config",
    "freeze_experiment_config",
    "mark_holdout_started",
]
