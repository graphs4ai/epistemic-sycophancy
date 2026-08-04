"""GRAD-014: live-β activity mask is 1[z + sβ > 0], not stale 1[z > 0]."""

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

DTYPE = torch.float64
FD_STEP = 1e-8
FD_ATOL = 1e-6
FD_RTOL = 1e-6


def _load_toy():
    import importlib.util

    toy_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "feature_selection"
        / "toy_gradients.py"
    )
    spec = importlib.util.spec_from_file_location("toy_gradients_grad014", toy_path)
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
            lambda_beta=0.0,
            delta_n=0.0,
            delta_c=0.0,
            w_r=1.0,
            w_u=0.0,
            beta_lower=-2.0,
            beta_upper=0.0,
            # FEAT-004 selected slice: features 0 and 2.
            feature_ids=((17, 0), (17, 2)),
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
                max_steps=5,
                question_ids=("q1",),
            ),
        ),
    )


def _baseline_batch(toy) -> dict[str, object]:
    """Unshifted JumpReLU latents + residual grads (production del-β pathology)."""
    return {
        "layer": 17,
        "residual_gradients": toy.spec_gradient().unsqueeze(0).clone(),
        "latents": toy.spec_latents().unsqueeze(0).clone(),
        "decoder": toy.spec_decoder(),
        "feature_scales": toy.spec_scales(),
        "question_ids": ["q1"],
    }


def _linear_margin_from_beta(
    *,
    beta: torch.Tensor,
    latents: torch.Tensor,
    scales: torch.Tensor,
    raw_projection: torch.Tensor,
    selected_indices: tuple[int, ...],
) -> torch.Tensor:
    """M(β) = Σ_j (ReLU(z_j + s_j β_j) - z_j) h_j for selected j (linear head)."""
    margin = torch.zeros((), dtype=DTYPE)
    for pool_i, feat_i in enumerate(selected_indices):
        z_j = latents[feat_i]
        s_j = scales[feat_i]
        h_j = raw_projection[feat_i]
        z_prime = torch.relu(z_j + s_j * beta[pool_i])
        margin = margin + (z_prime - z_j) * h_j
    return margin


def _one_sided_fd_jacobian(
    *,
    beta: torch.Tensor,
    latents: torch.Tensor,
    scales: torch.Tensor,
    raw_projection: torch.Tensor,
    selected_indices: tuple[int, ...],
) -> torch.Tensor:
    """Feasible one-sided FD of linear M at β (DEC-021 spirit; ε into −β)."""
    m = len(selected_indices)
    jac = torch.zeros(m, dtype=DTYPE)
    base = _linear_margin_from_beta(
        beta=beta,
        latents=latents,
        scales=scales,
        raw_projection=raw_projection,
        selected_indices=selected_indices,
    )
    for j in range(m):
        stepped = beta.clone()
        stepped[j] = beta[j] - FD_STEP
        delta = _linear_margin_from_beta(
            beta=stepped,
            latents=latents,
            scales=scales,
            raw_projection=raw_projection,
            selected_indices=selected_indices,
        )
        jac[j] = (delta - base) / (-FD_STEP)
    return jac


@pytest.mark.unit
def test_margin_jacobian__suppression_past_relu_boundary__selected_derivative_is_zero(
    tmp_path: Path,
) -> None:
    """GRAD-014: once z + sβ ≤ 0, live ∂M/∂β_j is 0 (not stale 1[z>0]).

    FEAT-004 slice: z0=0.5, s0=2 → boundary at β0=-0.25.
    At β=(-0.3, 0), z0+s0β0=-0.1 ≤ 0 ⇒ J=[0, 0.5].
    Stale baseline mask would still yield J0=4.0.
    """
    from epistemic_sycophancy.runner.adapters.margin_jacobian import (
        build_margin_jacobian_fn,
    )

    toy = _load_toy()
    study = _study(artifact_dir=str(tmp_path / "art"))

    class _FakeStack:
        def margin_projection_batch(
            self,
            *,
            belief_condition: str,
            question_ids: tuple[str, ...],
            beta: tuple[float, ...],
        ):
            # Reproduce production fallback: ignore β, return unshifted latents.
            del beta, belief_condition
            batch = _baseline_batch(toy)
            batch["question_ids"] = list(question_ids)
            return batch

    jac_fn = build_margin_jacobian_fn(study, _FakeStack())
    maps = jac_fn(
        beta=(-0.3, 0.0),
        question_ids=("q1",),
        partitions={"q_plus": frozenset({"q1"}), "q_minus": frozenset()},
    )
    row = maps["neutral_margin_jac"]["q1"]
    assert row.tolist() == [0.0, 0.5]


@pytest.mark.unit
def test_margin_jacobian__live_beta_points__match_feasible_finite_difference(
    tmp_path: Path,
) -> None:
    """GRAD-014: production jac ≡ one-sided FD at β=0, interior, boundary, checkpoint.

    Linear-head reference M(β)=Σ_j(ReLU(z_j+s_jβ_j)-z_j)h_j makes residual
    grads β-independent, so only the shifted activity mask is discriminative.
    """
    from epistemic_sycophancy.feature_selection.projected_gradient import (
        project_residual_gradient,
    )
    from epistemic_sycophancy.runner.adapters.margin_jacobian import (
        build_margin_jacobian_fn,
    )

    toy = _load_toy()
    study = _study(artifact_dir=str(tmp_path / "art"))
    latents = toy.spec_latents()
    scales = toy.spec_scales()
    decoder = toy.spec_decoder()
    raw = project_residual_gradient(gradient=toy.spec_gradient(), decoder=decoder)
    selected = (0, 2)

    class _FakeStack:
        def margin_projection_batch(
            self,
            *,
            belief_condition: str,
            question_ids: tuple[str, ...],
            beta: tuple[float, ...],
        ):
            del beta, belief_condition
            batch = _baseline_batch(toy)
            batch["question_ids"] = list(question_ids)
            return batch

    jac_fn = build_margin_jacobian_fn(study, _FakeStack())
    # β=0; interior (−0.1); exact boundary (−0.25); past boundary / "checkpoint".
    probe_betas = (
        (0.0, 0.0),
        (-0.1, 0.0),
        (-0.25, 0.0),
        (-0.5, -0.1),
    )
    for beta_tuple in probe_betas:
        beta = torch.tensor(beta_tuple, dtype=DTYPE)
        # Feasibility: one-sided step must not re-open a clamped feature.
        for pool_i, feat_i in enumerate(selected):
            pre = float(latents[feat_i] + scales[feat_i] * beta[pool_i])
            if pre > 0.0:
                assert FD_STEP * float(scales[feat_i]) < pre

        expected = _one_sided_fd_jacobian(
            beta=beta,
            latents=latents,
            scales=scales,
            raw_projection=raw,
            selected_indices=selected,
        )
        maps = jac_fn(
            beta=beta_tuple,
            question_ids=("q1",),
            partitions={"q_plus": frozenset({"q1"}), "q_minus": frozenset()},
        )
        got = maps["neutral_margin_jac"]["q1"]
        assert torch.allclose(got, expected, atol=FD_ATOL, rtol=FD_RTOL), (
            f"beta={beta_tuple}: got={got.tolist()} expected={expected.tolist()}"
        )
