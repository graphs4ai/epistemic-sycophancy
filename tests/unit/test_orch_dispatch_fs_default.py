"""ORCH-028: FS defaults + pool → study overlay (DEC-073)."""

from __future__ import annotations

import json
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
    StudyFsCoverageConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.runner.adapters.pool import (
    load_common_pool_artifact,
    study_with_selected_pool,
)
from epistemic_sycophancy.runner.cli import dispatch_stage
from epistemic_sycophancy.runner.identity import clear_stack_cache
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "adapters"


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
            fs_coverage=StudyFsCoverageConfig(question_ids=("q_fs_1", "q_fs_2")),
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


@pytest.mark.unit
def test_dispatch__feature_selection__builds_jacobian_scale_and_persists_pool_for_optimize(
    tmp_path: Path,
) -> None:
    """ORCH-028: None jacobian/scale → adapters; pool overlay sets coefficient_length."""
    clear_stack_cache()
    art = tmp_path / "art"
    study = _study(artifact_dir=str(art))
    layer = 17
    decoder = torch.tensor(
        [[3.0, 0.0], [0.0, 4.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    # Positive J on feature 0 so pool is nonempty.
    latents = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    residual_grads = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    qids = ("q_fs_1", "q_fs_2")
    sae = SimpleNamespace(decoder_weight=decoder, layer=layer)
    stack = SimpleNamespace(
        saes={layer: sae},
        device=torch.device("cpu"),
        fs_projection_batch=lambda **kwargs: {
            "layer": layer,
            "residual_gradients": residual_grads,
            "latents": latents,
            "question_ids": list(qids),
        },
    )

    result = dispatch_stage(
        "feature_selection",
        study=study,
        freeze_status="unsealed",
        stack_loader=lambda _s: stack,
        jacobian_fn=None,
        scale_fn=None,
        corpus_jsonl_paths=(FIXTURE_ROOT / "processed_mc0_tiny.jsonl",),
        split_manifest_path=FIXTURE_ROOT / "split_manifest_tiny.csv",
    )
    assert result.ok
    pool_path = Path(result.artifacts["pool"])
    assert pool_path.is_file()
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    assert payload["pool_size"] >= 1
    pool = load_common_pool_artifact(pool_path)
    overlaid = study_with_selected_pool(study, pool)
    assert overlaid.experiment.coefficient_length == len(pool.feature_ids)
    assert overlaid.experiment.coefficient_length >= 1
    assert overlaid.experiment.feature_ids == pool.feature_ids
