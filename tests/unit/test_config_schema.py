"""Configuration schema validation tests (Phase A)."""

from __future__ import annotations

import inspect
import math

import pytest

from epistemic_sycophancy.config import ExperimentConfig, InvalidExperimentConfig


def _valid_config_kwargs(**overrides: object) -> dict[str, object]:
    """Minimal valid kwargs for the fields implemented so far."""
    kwargs: dict[str, object] = {
        "tau": 1.0,
        "lambda_n": 0.0,
        "lambda_c": 0.0,
        "lambda_beta": 0.0,
        "delta_n": 0.0,
        "delta_c": 0.0,
        "w_r": 0.5,
        "w_u": 0.5,
        "beta_lower": -2.0,
        "beta_upper": 0.0,
        "feature_ids": (10, 20),
        "feature_scales": (1.0, 2.0),
        "coefficient_length": 2,
        "tie_policy": "merge_into_q_minus",
        "tie_band_epsilon": 1e-6,
        "mc1_tie_policy": "fail_and_report",
        "invalid_row_policy": "fail_trial",
        "multi_token_candidate_scoring": "sum_log_probs",
        "ro_manifest_selection": "primary_single",
        "continuation_A": "A",
        "continuation_B": "B",
        "continuation_include_eos": False,
        "attribution_scope": "last_prompt_token",
        "pool_eligibility_override": False,
        "pool_quota_per_list": 8,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.unit
def test_config__tau_nonpositive__raises_validation_error() -> None:
    """CFG-001: reject tau <= 0; accept finite tau > 0."""
    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(**_valid_config_kwargs(tau=0.0))

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(**_valid_config_kwargs(tau=-1.0))

    config = ExperimentConfig(**_valid_config_kwargs(tau=1.0))
    assert config.tau == 1.0


@pytest.mark.unit
def test_config__negative_penalty_or_tolerance__raises_validation_error() -> None:
    """CFG-002: lambda_n, lambda_c, lambda_beta, delta_n, delta_c must be >= 0."""
    for field in ("lambda_n", "lambda_c", "lambda_beta", "delta_n", "delta_c"):
        with pytest.raises(InvalidExperimentConfig):
            ExperimentConfig(**_valid_config_kwargs(**{field: -0.1}))

    config = ExperimentConfig(**_valid_config_kwargs())
    assert config.lambda_n == 0.0
    assert config.lambda_c == 0.0
    assert config.lambda_beta == 0.0
    assert config.delta_n == 0.0
    assert config.delta_c == 0.0


@pytest.mark.unit
def test_config__behavior_weights__are_nonnegative_and_normalized() -> None:
    """CFG-003: w_r >= 0, w_u >= 0, and w_r + w_u == 1."""
    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(**_valid_config_kwargs(w_r=-0.1, w_u=1.1))

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(**_valid_config_kwargs(w_r=1.1, w_u=-0.1))

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(**_valid_config_kwargs(w_r=0.6, w_u=0.6))

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(**_valid_config_kwargs(w_r=0.2, w_u=0.3))

    config = ExperimentConfig(**_valid_config_kwargs(w_r=0.0, w_u=1.0))
    assert config.w_r == 0.0
    assert config.w_u == 1.0


@pytest.mark.unit
def test_config__suppression_only_bounds__cannot_include_positive_beta() -> None:
    """CFG-004: suppression mode requires beta_lower <= beta_upper <= 0."""
    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(**_valid_config_kwargs(beta_lower=-1.0, beta_upper=0.5))

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(**_valid_config_kwargs(beta_lower=0.1, beta_upper=0.2))

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(**_valid_config_kwargs(beta_lower=0.0, beta_upper=-1.0))

    config = ExperimentConfig(
        **_valid_config_kwargs(beta_lower=-2.0, beta_upper=0.0)
    )
    assert config.beta_lower == -2.0
    assert config.beta_upper == 0.0
    assert config.coefficient_mode == "suppression"


@pytest.mark.unit
def test_config__bidirectional_mode__allows_positive_beta_upper() -> None:
    """CFG-004b / DEC-105: bidirectional allows beta_upper > 0 when lower <= upper."""
    with pytest.raises(InvalidExperimentConfig, match="coefficient_mode"):
        ExperimentConfig(
            **_valid_config_kwargs(coefficient_mode="excitation")
        )

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(
            **_valid_config_kwargs(
                coefficient_mode="bidirectional",
                beta_lower=0.5,
                beta_upper=-0.5,
            )
        )

    config = ExperimentConfig(
        **_valid_config_kwargs(
            coefficient_mode="bidirectional",
            beta_lower=-2.0,
            beta_upper=2.0,
        )
    )
    assert config.coefficient_mode == "bidirectional"
    assert config.beta_lower == -2.0
    assert config.beta_upper == 2.0

    # Suppression mode still rejects the same bounds.
    with pytest.raises(InvalidExperimentConfig, match="suppression-only"):
        ExperimentConfig(
            **_valid_config_kwargs(
                coefficient_mode="suppression",
                beta_lower=-2.0,
                beta_upper=2.0,
            )
        )


@pytest.mark.unit
def test_projected_adam__bidirectional_bounds__clamps_to_box() -> None:
    """OPT-003b / DEC-105: Adam accepts beta_upper > 0 and clamps into the box."""
    import torch

    from epistemic_sycophancy.optimization.projected_adam import ProjectedAdam

    beta = torch.nn.Parameter(torch.tensor([3.0, -3.0], dtype=torch.float64))
    adam = ProjectedAdam(
        beta=beta,
        adam_lr=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_microbatch_questions=1,
        beta_lower=-2.0,
        beta_upper=2.0,
    )
    with torch.no_grad():
        beta.grad = torch.zeros_like(beta)
    adam.step()
    values = beta.detach().tolist()
    assert all(-2.0 <= v <= 2.0 for v in values)
    assert values[0] == pytest.approx(2.0)
    assert values[1] == pytest.approx(-2.0)


@pytest.mark.unit
def test_config__feature_scales__must_be_finite_and_positive() -> None:
    """CFG-005: finite positive scales; unique IDs; matching lengths."""
    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(
            **_valid_config_kwargs(feature_scales=(0.0, 2.0))
        )

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(
            **_valid_config_kwargs(feature_scales=(-1.0, 2.0))
        )

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(
            **_valid_config_kwargs(feature_scales=(math.nan, 2.0))
        )

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(
            **_valid_config_kwargs(feature_scales=(math.inf, 2.0))
        )

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(
            **_valid_config_kwargs(feature_ids=(10, 10))
        )

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(
            **_valid_config_kwargs(
                feature_ids=(10, 20, 30),
                feature_scales=(1.0, 2.0),
                coefficient_length=2,
            )
        )

    with pytest.raises(InvalidExperimentConfig):
        ExperimentConfig(
            **_valid_config_kwargs(
                feature_ids=(10, 20),
                feature_scales=(1.0, 2.0),
                coefficient_length=3,
            )
        )

    config = ExperimentConfig(**_valid_config_kwargs())
    assert config.feature_ids == (10, 20)
    assert config.feature_scales == (1.0, 2.0)
    assert config.coefficient_length == 2


@pytest.mark.unit
def test_config__tie_and_invalid_row_policies__must_be_explicit() -> None:
    """CFG-006: unresolved policies have no hidden defaults and reject None."""
    policy_fields = (
        "tie_policy",
        "tie_band_epsilon",
        "mc1_tie_policy",
        "invalid_row_policy",
        "multi_token_candidate_scoring",
        "ro_manifest_selection",
        "continuation_A",
        "continuation_B",
        "continuation_include_eos",
        "attribution_scope",
        "pool_eligibility_override",
        "pool_quota_per_list",
    )
    signature = inspect.signature(ExperimentConfig.__init__)
    for field in policy_fields:
        assert field in signature.parameters
        assert signature.parameters[field].default is inspect.Parameter.empty

    for field in policy_fields:
        with pytest.raises(InvalidExperimentConfig):
            ExperimentConfig(**_valid_config_kwargs(**{field: None}))

    config = ExperimentConfig(**_valid_config_kwargs())
    assert config.tie_policy == "merge_into_q_minus"
    assert config.tie_band_epsilon == 1e-6
    assert config.mc1_tie_policy == "fail_and_report"
    assert config.invalid_row_policy == "fail_trial"
    assert config.multi_token_candidate_scoring == "sum_log_probs"
    assert config.ro_manifest_selection == "primary_single"
    assert config.continuation_A == "A"
    assert config.continuation_B == "B"
    assert config.continuation_include_eos is False
