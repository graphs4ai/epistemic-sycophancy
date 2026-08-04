"""GRAD-015: multi-layer selected-pool ∂M/∂β (DEC-054 scatter into length-m row)."""

from __future__ import annotations

from pathlib import Path

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


def _load_toy():
    import importlib.util

    toy_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "feature_selection"
        / "toy_gradients.py"
    )
    spec = importlib.util.spec_from_file_location("toy_gradients_grad015", toy_path)
    assert spec is not None and spec.loader is not None
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    return toy


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
            lambda_n=0.0,
            lambda_c=0.0,
            lambda_beta=0.0,
            delta_n=0.0,
            delta_c=0.0,
            w_r=1.0,
            w_u=0.0,
            beta_lower=-2.0,
            beta_upper=0.0,
            # Pool order ascending (layer, feature_id): one key per SAE.
            # FEAT-004 slice: L17 feat0 → J=4.0; L22 feat2 → J=0.5 at β=0.
            feature_ids=((17, 0), (22, 2)),
            feature_scales=(2.0, 0.5),
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
            fs_coverage=StudyFsCoverageConfig(question_ids=("q1",)),
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


def _baseline_batch(toy, *, layer: int) -> dict:
    g = toy.spec_gradient()
    return {
        "layer": layer,
        "residual_gradients": g.unsqueeze(0),
        "latents": toy.spec_latents().unsqueeze(0),
        "decoder": toy.spec_decoder(),
        "feature_scales": toy.spec_scales(),
        "question_ids": ["q1"],
    }


@pytest.mark.unit
def test_margin_jacobian__multi_layer_pool__scatters_per_layer_into_length_m_row(
    tmp_path: Path,
) -> None:
    """GRAD-015: ∂M/∂β length-m row fills from each SAE; no single-layer reject.

    Hand-derived (FEAT-004 / GRAD-003): same decoder/latents/grads on both layers.
    Pool [(17,0),(22,2)] at β=0 → [4.0, 0.5].
    """
    from epistemic_sycophancy.runner.adapters.margin_jacobian import (
        build_margin_jacobian_fn,
    )

    toy = _load_toy()
    study = _study(artifact_dir=str(tmp_path / "art"))
    called_layers: list[int] = []

    class _FakeStack:
        def margin_projection_batch(
            self,
            *,
            belief_condition: str,
            question_ids: tuple[str, ...],
            beta: tuple[float, ...],
            layer: int,
        ):
            del beta, belief_condition
            called_layers.append(int(layer))
            batch = _baseline_batch(toy, layer=int(layer))
            batch["question_ids"] = list(question_ids)
            return batch

    jac_fn = build_margin_jacobian_fn(study, _FakeStack())
    maps = jac_fn(
        beta=(0.0, 0.0),
        question_ids=("q1",),
        partitions={"q_plus": frozenset({"q1"}), "q_minus": frozenset()},
    )
    row = maps["neutral_margin_jac"]["q1"]
    assert row.dtype == torch.float64
    assert row.shape == (2,)
    assert row.tolist() == [4.0, 0.5]
    # One projection per pool layer per belief (N/IB/CB) → {17,22} × 3.
    assert set(called_layers) == {17, 22}
    assert called_layers.count(17) == 3
    assert called_layers.count(22) == 3
