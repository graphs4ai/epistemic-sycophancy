"""Pool artifact load + in-memory StudyConfig overlay (ORCH-028 / DEC-073)."""

from __future__ import annotations

import json
from pathlib import Path

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.feature_selection.pool import CommonFeaturePool


def load_common_pool_artifact(path: str | Path) -> CommonFeaturePool:
    """Load ``common_pool.json`` written by feature_selection dispatch.

    DEC-085 / FSC-006: require ``schema_version: 2`` with provenance. Stale
    neutral-only (v1) pools are rejected so optimize forces a re-``run-fs``.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != 2:
        raise ValueError(
            f"stale common_pool artifact at {path}: schema_version={version!r} "
            "(require 2 with nominator provenance); re-run feature_selection "
            "(run-fs) before optimize (DEC-085)"
        )
    if "provenance" not in payload:
        raise ValueError(
            f"stale common_pool artifact at {path}: missing provenance; "
            "re-run feature_selection (run-fs) before optimize (DEC-085)"
        )
    feature_ids = tuple(
        (int(pair[0]), int(pair[1])) for pair in payload["feature_ids"]
    )
    scales = tuple(float(x) for x in payload["feature_scales"])
    if len(feature_ids) != len(scales):
        raise ValueError(
            f"pool feature_ids length {len(feature_ids)} != scales {len(scales)}"
        )
    return CommonFeaturePool(feature_ids=feature_ids, scales=scales)


def study_with_selected_pool(
    study: StudyConfig,
    pool: CommonFeaturePool,
) -> StudyConfig:
    """Return a new StudyConfig with experiment feature fields from ``pool``."""
    exp = study.experiment
    new_exp = ExperimentConfig(
        tau=float(exp.tau),
        lambda_n=float(exp.lambda_n),
        lambda_c=float(exp.lambda_c),
        lambda_beta=float(exp.lambda_beta),
        delta_n=float(exp.delta_n),
        delta_c=float(exp.delta_c),
        w_r=float(exp.w_r),
        w_u=float(exp.w_u),
        beta_lower=float(exp.beta_lower),
        beta_upper=float(exp.beta_upper),
        feature_ids=pool.feature_ids,
        feature_scales=pool.scales,
        coefficient_length=len(pool.feature_ids),
        tie_policy=exp.tie_policy,
        tie_band_epsilon=exp.tie_band_epsilon,
        mc1_tie_policy=exp.mc1_tie_policy,
        invalid_row_policy=exp.invalid_row_policy,
        multi_token_candidate_scoring=exp.multi_token_candidate_scoring,
        ro_manifest_selection=exp.ro_manifest_selection,
        continuation_A=exp.continuation_A,
        continuation_B=exp.continuation_B,
        continuation_include_eos=exp.continuation_include_eos,
        attribution_scope=exp.attribution_scope,
        pool_eligibility_override=exp.pool_eligibility_override,
        pool_quota_per_list=exp.pool_quota_per_list,
    )
    return StudyConfig(stack=study.stack, experiment=new_exp, run=study.run)
