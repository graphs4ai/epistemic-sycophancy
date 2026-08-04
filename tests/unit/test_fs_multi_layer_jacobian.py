"""FSC-011: feature-selection Jacobians cover every configured SAE layer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
                layers=(17, 22),
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
            order_regime="CF",
            feature_chunk_size=2,
            prompt_batch_size=2,
            fs_coverage=StudyFsCoverageConfig(question_ids=("q_fs_1",)),
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
                max_steps=1,
            ),
        ),
    )


@pytest.mark.unit
def test_adapters__build_jacobian_fn__multi_layer_stack__merges_per_layer_maps(
    tmp_path: Path,
) -> None:
    """FSC-011: FS jacobian_fn keys both configured layers, not only layers[0].

    Distinct per-layer residual grads → distinct signed J at (17,0) vs (22,0).
    """
    from epistemic_sycophancy.runner.adapters.jacobian import build_jacobian_fn

    # Shared 1-feature SAE geometry; layer identity comes from residual grad scale.
    decoder = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    latents = torch.tensor([[1.0]], dtype=torch.float64)
    question_ids = ("q_fs_1",)
    called_layers: list[int] = []

    def fs_projection_batch(**kwargs):
        layer = int(kwargs["layer"])
        called_layers.append(layer)
        # g_17 = [2,0] → raw h=[2]; g_22 = [3,0] → raw h=[3]; z>0, s=1 → J=h.
        scale = 2.0 if layer == 17 else 3.0
        residual_grads = torch.tensor([[scale, 0.0]], dtype=torch.float64)
        return {
            "layer": layer,
            "residual_gradients": residual_grads,
            "latents": latents,
            "question_ids": list(question_ids),
        }

    sae17 = SimpleNamespace(decoder_weight=decoder, layer=17)
    sae22 = SimpleNamespace(decoder_weight=decoder, layer=22)
    stack = SimpleNamespace(
        saes={17: sae17, 22: sae22},
        device=torch.device("cpu"),
        fs_projection_batch=fs_projection_batch,
    )
    study = _study(artifact_dir=str(tmp_path / "art"))
    jacobian_fn = build_jacobian_fn(study, stack)
    signed = dict(jacobian_fn(order_regime="CF", question_ids=question_ids))

    assert set(called_layers) == {17, 22}
    assert signed[(17, 0)] == pytest.approx(2.0)
    assert signed[(22, 0)] == pytest.approx(3.0)
    assert (17, 0) in signed and (22, 0) in signed
