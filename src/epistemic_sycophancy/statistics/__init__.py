"""Statistics package: question-cluster bootstrap."""

from epistemic_sycophancy.statistics.cluster_bootstrap import (
    BootstrapSelectivityResult,
    PairedClusterResample,
    bootstrap_selectivity_interval,
    paired_cluster_resample,
    sample_question_clusters,
)

__all__ = [
    "BootstrapSelectivityResult",
    "PairedClusterResample",
    "bootstrap_selectivity_interval",
    "paired_cluster_resample",
    "sample_question_clusters",
]
