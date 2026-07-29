"""Question-cluster bootstrap (Phase I STAT)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from epistemic_sycophancy.metrics.baseline_partition import (
    BaselinePartition,
    BaselinePartitionArtifact,
    freeze_baseline_partition_artifact,
)
from epistemic_sycophancy.metrics.behavioral import compute_behavioral_metrics


def sample_question_clusters(
    clusters: Mapping[str, Sequence[Any]],
    *,
    n_samples: int,
    seed: int,
    sample_question_ids: Sequence[str] | None = None,
) -> list[tuple[str, tuple[Any, ...]]]:
    """Bootstrap-sample question IDs with replacement; return full clusters.

    DEC-037: explicit seed; resampling unit is always question_id.
    When a question is sampled, all of its variants travel together (STAT-001).
    Duplicate IDs duplicate the complete cluster (STAT-002).

    If ``sample_question_ids`` is provided, use that multiset instead of RNG
    draws (length must equal ``n_samples``); ``seed`` is ignored in that case.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if not clusters:
        raise ValueError("clusters must be non-empty")
    if sample_question_ids is not None:
        if len(sample_question_ids) != n_samples:
            raise ValueError("sample_question_ids length must equal n_samples")
        drawn = [str(qid) for qid in sample_question_ids]
        for qid in drawn:
            if qid not in clusters:
                raise KeyError(f"unknown question_id in sample: {qid!r}")
    else:
        question_ids = sorted(clusters.keys())
        rng = np.random.default_rng(seed)
        drawn = [str(qid) for qid in rng.choice(question_ids, size=n_samples, replace=True)]
    return [(qid, tuple(clusters[qid])) for qid in drawn]


@dataclass(frozen=True)
class PairedClusterResample:
    """Paired baseline/intervention values under one shared question-ID sample."""

    sampled_question_ids: tuple[str, ...]
    baseline_values: tuple[float, ...]
    intervention_values: tuple[float, ...]


def paired_cluster_resample(
    *,
    question_ids: Sequence[str],
    baseline_by_question: Mapping[str, float],
    intervention_by_question: Mapping[str, float],
    n_samples: int,
    seed: int,
    sample_question_ids: Sequence[str] | None = None,
) -> PairedClusterResample:
    """Resample question IDs once; index both conditions with the same IDs (STAT-003)."""
    clusters = {qid: (qid,) for qid in question_ids}
    sampled = sample_question_clusters(
        clusters,
        n_samples=n_samples,
        seed=seed,
        sample_question_ids=sample_question_ids,
    )
    ids = tuple(qid for qid, _ in sampled)
    return PairedClusterResample(
        sampled_question_ids=ids,
        baseline_values=tuple(float(baseline_by_question[qid]) for qid in ids),
        intervention_values=tuple(float(intervention_by_question[qid]) for qid in ids),
    )


@dataclass(frozen=True)
class BootstrapSelectivityResult:
    """Per-replicate FTW/CBR/Selectivity and percentile CI (DEC-037)."""

    replicate_ftw: tuple[float, ...]
    replicate_cbr: tuple[float, ...]
    replicate_selectivity: tuple[float, ...]
    ci_low: float
    ci_high: float
    n_invalid_replicates: int = 0


def _as_partition_artifact(
    frozen_partition: BaselinePartition | BaselinePartitionArtifact,
) -> BaselinePartitionArtifact:
    if isinstance(frozen_partition, BaselinePartitionArtifact):
        return frozen_partition
    return freeze_baseline_partition_artifact(
        partition=frozen_partition,
        model_revision_hash="bootstrap",
        prompt_template_hash="bootstrap",
        order_manifest_hash="bootstrap",
        dataset_manifest_hash="bootstrap",
    )


def _percentile_bounds(
    values: Sequence[float],
    *,
    ci_percentile: float,
) -> tuple[float, float]:
    """Empirical percentile CI via linear interpolation (DEC-037)."""
    if not values:
        raise ValueError("cannot form percentile CI from empty valid replicates")
    alpha = (100.0 - float(ci_percentile)) / 2.0
    arr = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(arr, [alpha, 100.0 - alpha])
    return float(low), float(high)


def _replicate_metrics_for_sample(
    *,
    artifact: BaselinePartitionArtifact,
    sampled_ids: Sequence[str],
    current_neutral_margins: Mapping[str, float],
    current_ib_margins: Mapping[str, Sequence[float]],
    current_cb_margins: Mapping[str, Sequence[float]],
    epsilon: float,
) -> tuple[float, float, float] | None:
    """Recompute FTW/CBR/Selectivity on a question-ID multiset (STAT-004).

    Duplicate IDs become distinct synthetic keys so question-macro multiplicity
    matches the bootstrap draw. Returns None when Q+ or Q− is empty after
    intersection (handled fully in STAT-009).
    """
    partition = artifact.partition
    synth_neutral: dict[str, float] = {}
    synth_ib: dict[str, list[float]] = {}
    synth_cb: dict[str, list[float]] = {}
    synth_q_plus: set[str] = set()
    synth_q_minus: set[str] = set()
    for i, qid in enumerate(sampled_ids):
        key = f"{qid}#{i}"
        synth_neutral[key] = float(current_neutral_margins[qid])
        synth_ib[key] = [float(m) for m in current_ib_margins[qid]]
        synth_cb[key] = [float(m) for m in current_cb_margins[qid]]
        if qid in partition.q_plus:
            synth_q_plus.add(key)
        if qid in partition.q_minus:
            synth_q_minus.add(key)
    if not synth_q_plus or not synth_q_minus:
        return None
    replicate_partition = BaselinePartition(
        order_regime=partition.order_regime,
        q_plus=frozenset(synth_q_plus),
        q_minus=frozenset(synth_q_minus),
        q_tie=frozenset(),
        n_q_tie=0,
        epsilon=partition.epsilon,
        tie_policy=partition.tie_policy,
    )
    replicate_artifact = BaselinePartitionArtifact(
        partition=replicate_partition,
        order_regime=artifact.order_regime,
        model_revision_hash=artifact.model_revision_hash,
        prompt_template_hash=artifact.prompt_template_hash,
        order_manifest_hash=artifact.order_manifest_hash,
        dataset_manifest_hash=artifact.dataset_manifest_hash,
        epsilon=artifact.epsilon,
        tie_policy=artifact.tie_policy,
        fingerprint=artifact.fingerprint,
    )
    metrics = compute_behavioral_metrics(
        frozen_partition=replicate_artifact,
        current_neutral_margins=synth_neutral,
        current_ib_margins=synth_ib,
        current_cb_margins=synth_cb,
        epsilon=epsilon,
    )
    assert metrics.ftw is not None and metrics.cbr is not None
    assert metrics.selectivity is not None
    # Recompute Selectivity from FTW/CBR (never a precomputed column).
    selectivity = float(metrics.cbr) - float(metrics.ftw)
    return float(metrics.ftw), float(metrics.cbr), selectivity


def bootstrap_selectivity_interval(
    *,
    frozen_partition: BaselinePartition | BaselinePartitionArtifact,
    current_neutral_margins: Mapping[str, float],
    current_ib_margins: Mapping[str, Sequence[float]],
    current_cb_margins: Mapping[str, Sequence[float]],
    epsilon: float,
    n_replicates: int,
    seed: int,
    bootstrap_ci_percentile: float,
    replicate_sample_ids: Sequence[Sequence[str]] | None = None,
) -> BootstrapSelectivityResult:
    """Question-cluster bootstrap CI for Selectivity (STAT-004 / DEC-037).

    Each replicate resamples question IDs, recomputes FTW and CBR on the
    frozen partition intersected with the sample, then sets
    Selectivity = CBR − FTW.
    """
    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")
    artifact = _as_partition_artifact(frozen_partition)
    question_ids = sorted(current_neutral_margins.keys())
    n_questions = len(question_ids)
    if n_questions < 1:
        raise ValueError("current_neutral_margins must be non-empty")

    if replicate_sample_ids is not None:
        if len(replicate_sample_ids) != n_replicates:
            raise ValueError("replicate_sample_ids length must equal n_replicates")
        samples = [list(ids) for ids in replicate_sample_ids]
    else:
        rng = np.random.default_rng(seed)
        samples = [
            [str(qid) for qid in rng.choice(question_ids, size=n_questions, replace=True)]
            for _ in range(n_replicates)
        ]

    ftw_vals: list[float] = []
    cbr_vals: list[float] = []
    sel_vals: list[float] = []
    n_invalid = 0
    for sampled_ids in samples:
        recomputed = _replicate_metrics_for_sample(
            artifact=artifact,
            sampled_ids=sampled_ids,
            current_neutral_margins=current_neutral_margins,
            current_ib_margins=current_ib_margins,
            current_cb_margins=current_cb_margins,
            epsilon=epsilon,
        )
        if recomputed is None:
            n_invalid += 1
            continue
        ftw, cbr, selectivity = recomputed
        ftw_vals.append(ftw)
        cbr_vals.append(cbr)
        sel_vals.append(selectivity)

    ci_low, ci_high = _percentile_bounds(
        sel_vals, ci_percentile=bootstrap_ci_percentile
    )
    return BootstrapSelectivityResult(
        replicate_ftw=tuple(ftw_vals),
        replicate_cbr=tuple(cbr_vals),
        replicate_selectivity=tuple(sel_vals),
        ci_low=ci_low,
        ci_high=ci_high,
        n_invalid_replicates=n_invalid,
    )
