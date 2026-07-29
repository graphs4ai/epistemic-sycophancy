"""Experiment configuration schema and validation."""

from epistemic_sycophancy.config.frozen import (
    ConfigImmutabilityError,
    FrozenExperimentConfig,
    freeze_experiment_config,
    mark_holdout_started,
)
from epistemic_sycophancy.config.schema import ExperimentConfig, InvalidExperimentConfig

__all__ = [
    "ConfigImmutabilityError",
    "ExperimentConfig",
    "FrozenExperimentConfig",
    "InvalidExperimentConfig",
    "freeze_experiment_config",
    "mark_holdout_started",
]
