"""ORCH-022: build_jacobian_fn + build_scale_fn (DEC-060 / DEC-061)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudySmokeConfig,
)
from epistemic_sycophancy.feature_selection.projected_gradient import (
    coefficient_jacobian,
    project_residual_gradient,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec
from epistemic_sycophancy.stack.scales import scales_for_layer_feature_keys


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
            lambda_n=1.0,
            lambda_c=1.0,
            lambda_beta=0.01,
            delta_n=0.0,
            delta_c=0.0,
            w_r=0.5,
            w_u=0.5,
            beta_lower=-2.0,
            beta_upper=0.0,
            feature_ids=(),
            feature_scales=(),
            coefficient_length=0,
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
            order_regimes=("CF",),
            feature_chunk_size=2,
            prompt_batch_size=2,
            smoke=StudySmokeConfig(question_ids=("q_fs_1", "q_fs_2")),
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
                max_steps=20,
                n_questions=2,
            ),
        ),
    )


@pytest.mark.unit
def test_adapters__build_jacobian_and_scale__projected_signed_j_matches_toy_reference(
    tmp_path: Path,
) -> None:
    """ORCH-022: projected J = s * 1[z>0] * h; scales = decoder_norm > 0."""
    from epistemic_sycophancy.runner.adapters.jacobian import build_jacobian_fn
    from epistemic_sycophancy.runner.adapters.scales import build_scale_fn

    layer = 17
    # Decoder rows: f0=[3,0], f1=[0,4], f2=[0,0] invalid would fail scale; use nonzero.
    decoder = torch.tensor(
        [[3.0, 0.0], [0.0, 4.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    # Two prompts (one per question): latents and residual grads.
    latents = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.5, 0.0]],
        dtype=torch.float64,
    )
    residual_grads = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0]],
        dtype=torch.float64,
    )
    question_ids = ("q_fs_1", "q_fs_2")

    sae = SimpleNamespace(decoder_weight=decoder, layer=layer)
    stack = SimpleNamespace(
        saes={layer: sae},
        device=torch.device("cpu"),
        # Adapter uses this hook for unit/toy FS projection inputs (DEC-060).
        fs_projection_batch=lambda **kwargs: {
            "layer": layer,
            "residual_gradients": residual_grads,
            "latents": latents,
            "question_ids": list(question_ids),
        },
    )
    study = _study(artifact_dir=str(tmp_path / "art"))

    scale_fn = build_scale_fn(study, stack)
    keys = [(layer, 0), (layer, 1), (layer, 2)]
    scales_map = dict(scale_fn(keys))
    expected_scales = scales_for_layer_feature_keys(
        keys=keys, saes=stack.saes, scale_source="decoder_norm"
    )
    for key, scale in zip(keys, expected_scales, strict=True):
        assert scales_map[key] == pytest.approx(scale)
        assert scale > 0.0

    # Hand reference: raw projection then coefficient_jacobian, mean over prompts.
    feature_scales = torch.tensor(
        [scales_map[(layer, 0)], scales_map[(layer, 1)], scales_map[(layer, 2)]],
        dtype=torch.float64,
    )
    raw = project_residual_gradient(
        gradient=residual_grads,
        decoder=decoder,
        feature_chunk_size=study.run.feature_chunk_size,
    )
    per_prompt = coefficient_jacobian(
        raw_projection=raw,
        latents=latents,
        feature_scales=feature_scales,
    )
    expected_j = per_prompt.mean(dim=0)  # equal question weight, one prompt each

    jacobian_fn = build_jacobian_fn(study, stack)
    signed = dict(jacobian_fn(order_regime="CF", question_ids=question_ids))
    for fid in range(3):
        assert signed[(layer, fid)] == pytest.approx(float(expected_j[fid].item()))
