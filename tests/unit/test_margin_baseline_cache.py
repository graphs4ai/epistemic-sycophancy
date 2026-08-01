"""PERF-BASELINE-001: cache frozen β=0 N/CB margins across payload builds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyFsCoverageConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _study(*, artifact_dir: str) -> StudyConfig:
    return StudyConfig(
        stack=ExperimentStackConfig(
            model=ModelSpec(
                hf_id="google/gemma-3-4b-it",
                revision="093f9f388b31de276ce2de164bdc2081324b9767",
                tokenizer_revision="093f9f388b31de276ce2de164bdc2081324b9767",
                dtype="bfloat16",
                device_policy="cuda_required",
            ),
            sae=SaeSiteSpec(
                release="gemma-scope-2-4b-it-res",
                site="resid_post",
                width="width_65k",
                l0="l0_medium",
                layers=(17,),
            ),
            hooks=HookSpec(
                token_scope="last_prompt_token",
                resolver_id="gemma3_resid_post",
                k=None,
            ),
        ),
        experiment=ExperimentConfig(
            tau=1.0,
            lambda_n=0.0,
            lambda_c=0.0,
            lambda_beta=0.01,
            delta_n=0.0,
            delta_c=0.0,
            w_r=0.5,
            w_u=0.5,
            beta_lower=-2.0,
            beta_upper=0.0,
            feature_ids=((17, 1), (17, 2)),
            feature_scales=(1.0, 1.0),
            coefficient_length=2,
            tie_policy="merge_into_q_minus",
            tie_band_epsilon=1e-6,
            mc1_tie_policy="fail_and_report",
            invalid_row_policy="fail_trial",
            multi_token_candidate_scoring="sum_log_probs",
            ro_manifest_selection="primary_single",
            continuation_A="A",
            continuation_B="B",
            continuation_include_eos=False,
            attribution_scope="last_prompt_token",
            pool_eligibility_override=False,
            pool_quota_per_list=8,
        ),
        run=StudyRunConfig(
            artifact_dir=artifact_dir,
            order_regime="CF",
            feature_chunk_size=1024,
            prompt_batch_size=1,
            fs_coverage=StudyFsCoverageConfig(question_ids=("q1", "q2")),
            optimizer=StudyOptimizerConfig(
                kind="projected_adam",
                adam_lr=0.1,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                adam_microbatch_questions=1,
            ),
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=20,
                n_questions=2,
            ),
        ),
    )


def _counting_scorer() -> tuple[
    Any,
    list[tuple[str, tuple[float, ...]]],
]:
    calls: list[tuple[str, tuple[float, ...]]] = []

    def margin_scorer(
        *,
        belief_condition: str,
        question_ids: Sequence[str],
        beta: Sequence[float],
    ) -> Mapping[str, Any]:
        beta_t = tuple(float(b) for b in beta)
        calls.append((belief_condition, beta_t))
        if belief_condition == "N":
            return {qid: 1.0 if qid == "q1" else -0.5 for qid in question_ids}
        if belief_condition == "IB":
            return {qid: (0.25,) for qid in question_ids}
        if belief_condition == "CB":
            return {qid: (0.75,) for qid in question_ids}
        raise ValueError(belief_condition)

    return margin_scorer, calls


def _zero_call_count(
    calls: Sequence[tuple[str, tuple[float, ...]]],
    *,
    belief: str,
    coefficient_length: int,
) -> int:
    zero = (0.0,) * coefficient_length
    return sum(1 for cond, beta in calls if cond == belief and beta == zero)


@pytest.mark.unit
def test_adapters__margin_baseline_cache__reuses_beta0_n_and_cb(
    tmp_path: Path,
) -> None:
    """PERF-BASELINE-001: N@0 and CB@0 scored once per qid tuple; reused later."""
    from epistemic_sycophancy.runner.adapters.margins import (
        MarginBaselineCache,
        build_margin_payload,
    )

    study = _study(artifact_dir=str(tmp_path / "art"))
    scorer, calls = _counting_scorer()
    cache = MarginBaselineCache(
        scorer=scorer,
        coefficient_length=int(study.experiment.coefficient_length),
    )
    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}
    qids = ("q1", "q2")
    m = int(study.experiment.coefficient_length)

    payload1 = build_margin_payload(
        study,
        object(),
        beta=(-0.5, -1.0),
        question_ids=qids,
        partitions=partitions,
        margin_scorer=scorer,
        baseline_cache=cache,
    )
    assert payload1["baseline_neutral_margins"]["q1"] == pytest.approx(1.0)
    assert payload1["baseline_cb_margins"]["q2"] == (0.75,)
    assert _zero_call_count(calls, belief="N", coefficient_length=m) == 1
    assert _zero_call_count(calls, belief="CB", coefficient_length=m) == 1

    n_zero_before = _zero_call_count(calls, belief="N", coefficient_length=m)
    cb_zero_before = _zero_call_count(calls, belief="CB", coefficient_length=m)
    calls_before = len(calls)

    payload2 = build_margin_payload(
        study,
        object(),
        beta=(-1.25, -0.25),
        question_ids=qids,
        partitions=partitions,
        margin_scorer=scorer,
        baseline_cache=cache,
    )
    assert payload2["baseline_neutral_margins"] == payload1["baseline_neutral_margins"]
    assert payload2["baseline_cb_margins"] == payload1["baseline_cb_margins"]
    assert _zero_call_count(calls, belief="N", coefficient_length=m) == n_zero_before
    assert _zero_call_count(calls, belief="CB", coefficient_length=m) == cb_zero_before
    # Current-β N/IB/CB still scored on the second payload.
    assert len(calls) == calls_before + 3
    assert ("N", (-1.25, -0.25)) in calls[calls_before:]
    assert ("IB", (-1.25, -0.25)) in calls[calls_before:]
    assert ("CB", (-1.25, -0.25)) in calls[calls_before:]


@pytest.mark.unit
def test_adapters__margin_baseline_cache__different_qid_tuple__separate_entry(
    tmp_path: Path,
) -> None:
    """PERF-BASELINE-001: distinct question-ID tuples fill separate baselines."""
    from epistemic_sycophancy.runner.adapters.margins import (
        MarginBaselineCache,
        build_margin_payload,
    )

    study = _study(artifact_dir=str(tmp_path / "art"))
    scorer, calls = _counting_scorer()
    cache = MarginBaselineCache(
        scorer=scorer,
        coefficient_length=int(study.experiment.coefficient_length),
    )
    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}
    m = int(study.experiment.coefficient_length)

    build_margin_payload(
        study,
        object(),
        beta=(-0.5, -1.0),
        question_ids=("q1", "q2"),
        partitions=partitions,
        margin_scorer=scorer,
        baseline_cache=cache,
    )
    assert _zero_call_count(calls, belief="N", coefficient_length=m) == 1
    assert _zero_call_count(calls, belief="CB", coefficient_length=m) == 1

    build_margin_payload(
        study,
        object(),
        beta=(-0.5, -1.0),
        question_ids=("q1",),
        partitions={"q_plus": frozenset({"q1"}), "q_minus": frozenset()},
        margin_scorer=scorer,
        baseline_cache=cache,
    )
    assert _zero_call_count(calls, belief="N", coefficient_length=m) == 2
    assert _zero_call_count(calls, belief="CB", coefficient_length=m) == 2


@pytest.mark.unit
def test_adapters__objective_and_grad__share_baseline_cache__scores_beta0_once(
    tmp_path: Path,
) -> None:
    """PERF-BASELINE-002: one shared cache fills N@0/CB@0 once across grad+objective."""
    import torch

    from epistemic_sycophancy.runner.adapters.margins import MarginBaselineCache
    from epistemic_sycophancy.runner.adapters.objective import (
        build_grad_fn,
        build_objective_fn,
    )

    study = _study(artifact_dir=str(tmp_path / "art"))
    scorer, calls = _counting_scorer()
    m = int(study.experiment.coefficient_length)
    cache = MarginBaselineCache(scorer=scorer, coefficient_length=m)
    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}
    eligible = ("q1", "q2")

    ones = torch.ones(m, dtype=torch.float64)

    def margin_jacobian_fn(*, beta, question_ids, partitions):
        del beta, question_ids, partitions
        return {
            "ib_margin_jac": {"q1": [ones.clone()], "q2": [ones.clone()]},
            "cb_margin_jac": {"q1": [ones.clone()], "q2": [ones.clone()]},
            "neutral_margin_jac": {"q1": ones.clone(), "q2": ones.clone()},
        }

    objective_fn = build_objective_fn(
        study,
        stack=object(),
        partitions=partitions,
        margin_scorer=scorer,
        baseline_cache=cache,
    )
    grad_fn = build_grad_fn(
        study,
        stack=object(),
        partitions=partitions,
        margin_scorer=scorer,
        margin_jacobian_fn=margin_jacobian_fn,
        baseline_cache=cache,
    )

    beta = (-0.5, -1.0)
    _ = grad_fn(beta, eligible)
    assert _zero_call_count(calls, belief="N", coefficient_length=m) == 1
    assert _zero_call_count(calls, belief="CB", coefficient_length=m) == 1

    _ = objective_fn(beta, eligible)
    assert _zero_call_count(calls, belief="N", coefficient_length=m) == 1
    assert _zero_call_count(calls, belief="CB", coefficient_length=m) == 1
