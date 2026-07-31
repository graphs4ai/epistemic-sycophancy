"""ORCH-LOG-007: tqdm on optimize — Adam fixed-total batch bars; CMA trial bars."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudySmokeConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _adam_study(artifact_dir: str) -> StudyConfig:
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
            feature_ids=((17, 1),),
            feature_scales=(1.0,),
            coefficient_length=1,
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
            smoke=StudySmokeConfig(question_ids=("q1",)),
            optimizer=StudyOptimizerConfig(
                kind="projected_adam",
                adam_lr=0.1,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                adam_microbatch_questions=1,
                max_steps=1,
            ),
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=3,
                question_ids=("qo1",),
            ),
        ),
    )


def _cma_study(artifact_dir: str) -> StudyConfig:
    study = _adam_study(artifact_dir)
    object.__setattr__(study.experiment, "feature_ids", ((17, 1), (17, 2)))
    object.__setattr__(study.experiment, "feature_scales", (1.0, 1.0))
    object.__setattr__(study.experiment, "coefficient_length", 2)
    object.__setattr__(
        study.run,
        "optimizer",
        StudyOptimizerConfig(
            kind="cmaes",
            cma_seed=0,
            adam_lr=0.1,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            adam_microbatch_questions=1,
            max_steps=1,
        ),
    )
    object.__setattr__(
        study.run,
        "optimize",
        StudyOptimizeConfig(
            budget_match_on="n_objective_evals",
            n_trials=2,
            population_size=4,
            question_ids=("qo1",),
        ),
    )
    return study


class _RecordingBar:
    """Minimal tqdm stand-in that records totals, updates, and postfix."""

    instances: list[_RecordingBar] = []

    def __init__(self, iterable: Any = None, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self._items = list(iterable) if iterable is not None else []
        self.total = kwargs.get("total")
        self.n = 0
        self.postfixes: list[dict[str, Any]] = []
        self.total_history: list[int | None] = [self.total]
        type(self).instances.append(self)

    def __iter__(self):
        yield from self._items

    def update(self, n: int = 1) -> None:
        self.n += int(n)

    def refresh(self) -> None:
        return None

    def set_postfix(self, ordered_dict: Any = None, **kwargs: Any) -> None:
        del ordered_dict
        self.postfixes.append(dict(kwargs))

    def close(self) -> None:
        return None


@pytest.mark.unit
def test_optimize__adam_steps__fixed_total_bar_never_shrinks_percentage(
    tmp_path: Path,
) -> None:
    """ORCH-LOG-007c: Adam step bar total is fixed; ticks only advance n."""
    from epistemic_sycophancy.runner.optimize import run_optimize_dispatch
    from epistemic_sycophancy.runner.progress import tick_prompt_batch

    _RecordingBar.instances.clear()
    study = _adam_study(str(tmp_path / "art"))
    fixed_total = 5

    def objective_fn(beta, question_ids):
        del beta, question_ids
        for _ in range(3):
            tick_prompt_batch()
        return 1.0

    def grad_fn(beta, question_ids):
        del beta, question_ids
        for _ in range(2):
            tick_prompt_batch()
        return (1.0,)

    with patch("epistemic_sycophancy.runner.progress.tqdm", _RecordingBar):
        run_optimize_dispatch(
            study=study,
            freeze_status="unsealed",
            identity_passed=True,
            optimization_question_ids=("qo1",),
            objective_fn=objective_fn,
            grad_fn=grad_fn,
            beta_init=(0.0,),
            adam_step_batch_total=fixed_total,
        )

    assert len(_RecordingBar.instances) == 3
    for step_idx, bar in enumerate(_RecordingBar.instances):
        assert bar.kwargs.get("desc") == f"adam step {step_idx + 1}/3"
        assert bar.kwargs.get("unit") == "batch"
        assert bar.kwargs.get("total") == fixed_total
        assert bar.total == fixed_total
        assert bar.n == fixed_total
        assert bar.postfixes
        assert "l_total" in bar.postfixes[-1]


@pytest.mark.unit
def test_margin_batch__ticks_fixed_total_adam_step_bar() -> None:
    """ORCH-LOG-007c: margin_batch ticks a pre-sized Adam step bar."""
    from epistemic_sycophancy.runner.adapters.margin_batch import (
        compute_margin_projection_batch,
    )
    from epistemic_sycophancy.runner.progress import adam_step_batch_progress
    from tests.unit.test_margin_batch_prompt_microbatch import _Stack

    _RecordingBar.instances.clear()
    stack = _Stack()
    texts = ("p0", "p1", "p2")
    with patch("epistemic_sycophancy.runner.progress.tqdm", _RecordingBar):
        with adam_step_batch_progress(step=0, n_steps=1, total=3):
            compute_margin_projection_batch(
                stack,
                layer=17,
                texts=texts,
                question_ids=("q0", "q1", "q2"),
                continuation_token_ids_A=(0,),
                continuation_token_ids_B=(1,),
                truthful_labels=("A", "B", "A"),
                prompt_batch_size=1,
            )
    assert len(_RecordingBar.instances) == 1
    bar = _RecordingBar.instances[0]
    assert bar.kwargs.get("total") == 3
    assert bar.total == 3
    assert bar.n == 3


@pytest.mark.unit
def test_optimize__cma_trials__tqdm_bar_updates_each_iteration(
    tmp_path: Path,
) -> None:
    """ORCH-LOG-007: CMA-ES wraps n_trials with tqdm and posts l_total."""
    from epistemic_sycophancy.runner.optimize import run_optimize_dispatch

    _RecordingBar.instances.clear()
    study = _cma_study(str(tmp_path / "art"))

    def objective_fn(beta, question_ids):
        del question_ids
        return float(sum(x * x for x in beta))

    with patch("epistemic_sycophancy.runner.optimize.tqdm", _RecordingBar):
        run_optimize_dispatch(
            study=study,
            freeze_status="unsealed",
            identity_passed=True,
            optimization_question_ids=("qo1",),
            objective_fn=objective_fn,
            grad_fn=None,
            beta_init=(0.0, 0.0),
        )

    assert len(_RecordingBar.instances) == 1
    bar = _RecordingBar.instances[0]
    assert bar.kwargs.get("total") == 2
    assert bar.kwargs.get("desc") == "optimize"
    assert len(bar.postfixes) == 2
    assert "l_total" in bar.postfixes[-1]
