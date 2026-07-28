"""Configuration schema and validation."""

from __future__ import annotations

import math
from collections.abc import Sequence


class InvalidExperimentConfig(Exception):
    """Raised when an experiment configuration violates a required invariant."""


class ExperimentConfig:
    """Validated experiment configuration for Phase A CFG invariants."""

    def __init__(
        self,
        *,
        tau: float,
        lambda_n: float,
        lambda_c: float,
        lambda_beta: float,
        delta_n: float,
        delta_c: float,
        w_r: float,
        w_u: float,
        beta_lower: float,
        beta_upper: float,
        feature_ids: Sequence[object],
        feature_scales: Sequence[float],
        coefficient_length: int,
        tie_policy: object,
        tie_band_epsilon: object,
        mc1_tie_policy: object,
        invalid_row_policy: object,
        multi_token_candidate_scoring: object,
        ro_manifest_selection: object,
        continuation_A: object,
        continuation_B: object,
        continuation_include_eos: object,
        attribution_scope: object,
        pool_eligibility_override: object,
        pool_quota_per_list: object,
    ) -> None:
        if tau <= 0:
            raise InvalidExperimentConfig(
                f"tau must be strictly positive; got {tau!r}"
            )
        for name, value in (
            ("lambda_n", lambda_n),
            ("lambda_c", lambda_c),
            ("lambda_beta", lambda_beta),
            ("delta_n", delta_n),
            ("delta_c", delta_c),
        ):
            if value < 0:
                raise InvalidExperimentConfig(
                    f"{name} must be nonnegative; got {value!r}"
                )
        if w_r < 0 or w_u < 0:
            raise InvalidExperimentConfig(
                f"behavioral weights must be nonnegative; got w_r={w_r!r}, w_u={w_u!r}"
            )
        if w_r + w_u != 1.0:
            raise InvalidExperimentConfig(
                f"behavioral weights must sum to 1; got w_r={w_r!r}, w_u={w_u!r}"
            )
        if not (beta_lower <= beta_upper <= 0):
            raise InvalidExperimentConfig(
                "suppression-only bounds require beta_lower <= beta_upper <= 0; "
                f"got beta_lower={beta_lower!r}, beta_upper={beta_upper!r}"
            )

        ids = tuple(feature_ids)
        scales = tuple(feature_scales)
        if len(ids) != len(set(ids)):
            raise InvalidExperimentConfig(
                f"selected feature IDs must be unique; got {ids!r}"
            )
        if not (len(ids) == len(scales) == coefficient_length):
            raise InvalidExperimentConfig(
                "feature IDs, scales, and coefficient_length must match; "
                f"got n_ids={len(ids)}, n_scales={len(scales)}, "
                f"coefficient_length={coefficient_length}"
            )
        for scale in scales:
            if not math.isfinite(scale) or scale <= 0:
                raise InvalidExperimentConfig(
                    f"feature scales must be finite and strictly positive; got {scale!r}"
                )

        for name, policy in (
            ("tie_policy", tie_policy),
            ("tie_band_epsilon", tie_band_epsilon),
            ("mc1_tie_policy", mc1_tie_policy),
            ("invalid_row_policy", invalid_row_policy),
            ("multi_token_candidate_scoring", multi_token_candidate_scoring),
            ("ro_manifest_selection", ro_manifest_selection),
            ("continuation_A", continuation_A),
            ("continuation_B", continuation_B),
            ("continuation_include_eos", continuation_include_eos),
            ("attribution_scope", attribution_scope),
            ("pool_eligibility_override", pool_eligibility_override),
            ("pool_quota_per_list", pool_quota_per_list),
        ):
            if policy is None:
                raise InvalidExperimentConfig(
                    f"{name} must be explicit; hidden defaults are forbidden"
                )
        if not isinstance(tie_band_epsilon, (int, float)) or isinstance(
            tie_band_epsilon, bool
        ):
            raise InvalidExperimentConfig(
                f"tie_band_epsilon must be a finite nonnegative float; "
                f"got {tie_band_epsilon!r}"
            )
        if not math.isfinite(float(tie_band_epsilon)) or float(tie_band_epsilon) < 0.0:
            raise InvalidExperimentConfig(
                f"tie_band_epsilon must be a finite nonnegative float; "
                f"got {tie_band_epsilon!r}"
            )
        if mc1_tie_policy != "fail_and_report":
            raise InvalidExperimentConfig(
                "DEC-014 requires mc1_tie_policy='fail_and_report'; "
                f"got {mc1_tie_policy!r}"
            )
        if continuation_A != "A" or continuation_B != "B":
            raise InvalidExperimentConfig(
                "DEC-010 requires continuation_A='A' and continuation_B='B'; "
                f"got continuation_A={continuation_A!r}, continuation_B={continuation_B!r}"
            )
        if continuation_include_eos is not False:
            raise InvalidExperimentConfig(
                "DEC-010 requires continuation_include_eos=False; "
                f"got {continuation_include_eos!r}"
            )
        if not isinstance(pool_eligibility_override, bool):
            raise InvalidExperimentConfig(
                "pool_eligibility_override must be an explicit bool; "
                f"got {pool_eligibility_override!r}"
            )
        if not isinstance(pool_quota_per_list, int) or isinstance(
            pool_quota_per_list, bool
        ):
            raise InvalidExperimentConfig(
                "pool_quota_per_list must be an explicit positive int; "
                f"got {pool_quota_per_list!r}"
            )
        if pool_quota_per_list <= 0:
            raise InvalidExperimentConfig(
                "pool_quota_per_list must be an explicit positive int; "
                f"got {pool_quota_per_list!r}"
            )

        self.tau = tau
        self.lambda_n = lambda_n
        self.lambda_c = lambda_c
        self.lambda_beta = lambda_beta
        self.delta_n = delta_n
        self.delta_c = delta_c
        self.w_r = w_r
        self.w_u = w_u
        self.beta_lower = beta_lower
        self.beta_upper = beta_upper
        self.feature_ids = ids
        self.feature_scales = scales
        self.coefficient_length = coefficient_length
        self.tie_policy = tie_policy
        self.tie_band_epsilon = float(tie_band_epsilon)
        self.mc1_tie_policy = mc1_tie_policy
        self.invalid_row_policy = invalid_row_policy
        self.multi_token_candidate_scoring = multi_token_candidate_scoring
        self.ro_manifest_selection = ro_manifest_selection
        self.continuation_A = continuation_A
        self.continuation_B = continuation_B
        self.continuation_include_eos = continuation_include_eos
        self.attribution_scope = attribution_scope
        self.pool_eligibility_override = pool_eligibility_override
        self.pool_quota_per_list = pool_quota_per_list
