"""Behavioral metrics and baseline partitions."""

from epistemic_sycophancy.metrics.baseline_partition import (
    BaselinePartition,
    BaselinePartitionArtifact,
    build_baseline_partition,
    freeze_baseline_partition_artifact,
    select_partition_for_evaluation,
)
from epistemic_sycophancy.metrics.behavioral import (
    BehavioralMetrics,
    compute_behavioral_metrics,
    is_truthful_margin,
)
from epistemic_sycophancy.metrics.exceptions import DegenerateBaselineError

__all__ = [
    "BaselinePartition",
    "BaselinePartitionArtifact",
    "BehavioralMetrics",
    "DegenerateBaselineError",
    "build_baseline_partition",
    "compute_behavioral_metrics",
    "freeze_baseline_partition_artifact",
    "is_truthful_margin",
    "select_partition_for_evaluation",
]
