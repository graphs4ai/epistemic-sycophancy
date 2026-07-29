"""Toy end-to-end integration tests (Phase J E2E / spec §21)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.evaluation.toy_e2e import run_toy_e2e_baseline
from tests.fixtures.e2e.corpus import (
    GOLDEN_CF_BASELINE_LOGITS,
    GOLDEN_CF_BASELINE_MARGINS,
    GOLDEN_CF_CB_MARGINS,
    GOLDEN_CF_IB_MARGINS,
    GOLDEN_CF_NEUTRAL_MARGINS,
    GOLDEN_CBR,
    GOLDEN_FTW,
    GOLDEN_KNOWN_BETA_L_BEHAVIOR,
    GOLDEN_KNOWN_BETA_L_BETA,
    GOLDEN_KNOWN_BETA_L_CORRECT,
    GOLDEN_KNOWN_BETA_L_NEUTRAL,
    GOLDEN_KNOWN_BETA_L_RECOVER,
    GOLDEN_KNOWN_BETA_L_RESIST,
    GOLDEN_KNOWN_BETA_L_TOTAL,
    GOLDEN_KNOWN_BETA_Q1N_DELTA,
    GOLDEN_KNOWN_BETA_Q1N_LATENTS,
    GOLDEN_KNOWN_BETA_Q1N_LATENTS_PRIME,
    GOLDEN_KNOWN_BETA_Q1N_LOGITS,
    GOLDEN_NEUTRAL_ACCURACY,
    GOLDEN_N_Q_MINUS,
    GOLDEN_N_Q_PLUS,
    GOLDEN_PRA_ALL,
    GOLDEN_PRA_MEAN,
    GOLDEN_Q_MINUS,
    GOLDEN_Q_PLUS,
    GOLDEN_SELECTIVITY,
    KNOWN_BETA,
    KNOWN_SCALES,
    KNOWN_SELECTED,
)

@pytest.mark.integration
def test_e2e_toy__baseline__matches_hand_computed_logits_margins_partitions_and_metrics() -> (
    None
):
    """E2E-001: unintervened toy pipeline matches hand-computed CF goldens."""
    result = run_toy_e2e_baseline(order_regime="CF")

    for prompt_id, expected_logits in GOLDEN_CF_BASELINE_LOGITS.items():
        got = result.logits_by_prompt_id[prompt_id]
        assert got == pytest.approx(expected_logits, abs=1e-12, rel=1e-12)

    for prompt_id, expected_margin in GOLDEN_CF_BASELINE_MARGINS.items():
        assert result.margins_by_prompt_id[prompt_id] == pytest.approx(
            expected_margin, abs=1e-12, rel=1e-12
        )

    assert result.neutral_margins == pytest.approx(
        GOLDEN_CF_NEUTRAL_MARGINS, abs=1e-12, rel=1e-12
    )
    for qid, expected in GOLDEN_CF_IB_MARGINS.items():
        assert result.ib_margins[qid] == pytest.approx(expected, abs=1e-12, rel=1e-12)
    for qid, expected in GOLDEN_CF_CB_MARGINS.items():
        assert result.cb_margins[qid] == pytest.approx(expected, abs=1e-12, rel=1e-12)

    assert result.partition.q_plus == GOLDEN_Q_PLUS
    assert result.partition.q_minus == GOLDEN_Q_MINUS
    assert result.metrics.neutral_accuracy == pytest.approx(
        GOLDEN_NEUTRAL_ACCURACY, abs=1e-12, rel=1e-12
    )
    assert result.metrics.ftw == pytest.approx(GOLDEN_FTW, abs=1e-12, rel=1e-12)
    assert result.metrics.cbr == pytest.approx(GOLDEN_CBR, abs=1e-12, rel=1e-12)
    assert result.metrics.selectivity == pytest.approx(
        GOLDEN_SELECTIVITY, abs=1e-12, rel=1e-12
    )
    assert result.metrics.pra_mean == pytest.approx(
        GOLDEN_PRA_MEAN, abs=1e-12, rel=1e-12
    )
    assert result.metrics.pra_all == pytest.approx(
        GOLDEN_PRA_ALL, abs=1e-12, rel=1e-12
    )
    assert result.metrics.n_q_plus == GOLDEN_N_Q_PLUS
    assert result.metrics.n_q_minus == GOLDEN_N_Q_MINUS


@pytest.mark.integration
def test_e2e_toy__zero_beta__matches_unhooked_pipeline() -> None:
    """E2E-002: β=0 hooked path matches unhooked baseline logits and margins."""
    from epistemic_sycophancy.evaluation.toy_e2e import run_toy_e2e_with_beta

    unhooked = run_toy_e2e_baseline(order_regime="CF")
    hooked = run_toy_e2e_with_beta(
        order_regime="CF",
        beta=(0.0, 0.0, 0.0),
        selected_indices=(0, 1, 2),
        scales=(1.0, 1.0, 1.0),
    )
    assert hooked.logits_by_prompt_id == unhooked.logits_by_prompt_id
    assert hooked.margins_by_prompt_id == unhooked.margins_by_prompt_id
    assert hooked.neutral_margins == unhooked.neutral_margins
    assert hooked.ib_margins == unhooked.ib_margins
    assert hooked.cb_margins == unhooked.cb_margins


@pytest.mark.integration
def test_e2e_toy__known_beta__matches_hand_computed_latents_delta_logits_and_objective() -> (
    None
):
    """E2E-003: known β matches hand-computed latents, Δx, logits, and objective."""
    from epistemic_sycophancy.evaluation.toy_e2e import (
        evaluate_toy_e2e_objective,
        inspect_toy_e2e_prompt,
    )

    detail = inspect_toy_e2e_prompt(
        prompt_id="CF:q1:N:0",
        beta=KNOWN_BETA,
        selected_indices=KNOWN_SELECTED,
        scales=KNOWN_SCALES,
    )
    assert detail.latents == pytest.approx(
        GOLDEN_KNOWN_BETA_Q1N_LATENTS, abs=1e-12, rel=1e-12
    )
    assert detail.latents_prime == pytest.approx(
        GOLDEN_KNOWN_BETA_Q1N_LATENTS_PRIME, abs=1e-12, rel=1e-12
    )
    assert detail.residual_delta == pytest.approx(
        GOLDEN_KNOWN_BETA_Q1N_DELTA, abs=1e-12, rel=1e-12
    )
    assert detail.logits == pytest.approx(
        GOLDEN_KNOWN_BETA_Q1N_LOGITS, abs=1e-12, rel=1e-12
    )

    objective = evaluate_toy_e2e_objective(
        order_regime="CF",
        beta=KNOWN_BETA,
        selected_indices=KNOWN_SELECTED,
        scales=KNOWN_SCALES,
        tau=1.0,
        w_r=0.5,
        w_u=0.5,
        delta_n=0.1,
        delta_c=0.1,
        lambda_n=0.1,
        lambda_c=0.1,
        lambda_beta=0.1,
    )
    assert objective.l_resist == pytest.approx(
        GOLDEN_KNOWN_BETA_L_RESIST, abs=1e-12, rel=1e-12
    )
    assert objective.l_recover == pytest.approx(
        GOLDEN_KNOWN_BETA_L_RECOVER, abs=1e-12, rel=1e-12
    )
    assert objective.l_behavior == pytest.approx(
        GOLDEN_KNOWN_BETA_L_BEHAVIOR, abs=1e-12, rel=1e-12
    )
    assert objective.l_neutral == pytest.approx(
        GOLDEN_KNOWN_BETA_L_NEUTRAL, abs=1e-12, rel=1e-12
    )
    assert objective.l_correct == pytest.approx(
        GOLDEN_KNOWN_BETA_L_CORRECT, abs=1e-12, rel=1e-12
    )
    assert objective.l_beta == pytest.approx(
        GOLDEN_KNOWN_BETA_L_BETA, abs=1e-12, rel=1e-12
    )
    assert objective.l_total == pytest.approx(
        GOLDEN_KNOWN_BETA_L_TOTAL, abs=1e-12, rel=1e-12
    )


@pytest.mark.integration
def test_e2e_toy__projected_gradient__matches_autograd_and_finite_difference() -> None:
    """E2E-004: projected J on DEC-046 prompt matches autograd and one-sided FD."""
    import torch

    from epistemic_sycophancy.evaluation.toy_e2e import (
        toy_e2e_prompt_coefficient_jacobian,
    )

    prompt_id = "CF:q1:N:0"
    selected = KNOWN_SELECTED
    scales = KNOWN_SCALES
    projected = toy_e2e_prompt_coefficient_jacobian(
        prompt_id=prompt_id,
        selected_indices=selected,
        scales=scales,
    )
    autograd_j, fd_j = _reference_autograd_and_fd_jacobian(
        prompt_id=prompt_id,
        selected_indices=selected,
        scales=scales,
    )
    assert projected == pytest.approx(autograd_j, abs=1e-8, rel=1e-6)
    assert projected == pytest.approx(fd_j, abs=1e-6, rel=1e-6)


def _reference_autograd_and_fd_jacobian(
    *,
    prompt_id: str,
    selected_indices: tuple[int, ...],
    scales: tuple[float, ...],
) -> tuple[list[float], list[float]]:
    """Independent autograd + DEC-021 one-sided FD for the E2E prompt loss."""
    import torch

    from epistemic_sycophancy.evaluation.toy_e2e import build_dec046_corpus
    from epistemic_sycophancy.intervention.sae_delta import apply_additive_sae_delta

    row = next(r for r in build_dec046_corpus() if r.prompt_id == prompt_id)
    residual = torch.tensor(row.residual_last, dtype=torch.float64)
    w_dec = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64)
    w_enc = torch.tensor(
        [[0.5, 0.0], [0.0, 0.25], [0.25, 0.25]], dtype=torch.float64
    )
    b_enc = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    head = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    scales_t = torch.tensor(list(scales), dtype=torch.float64)
    for param in (w_dec, w_enc, b_enc, head, scales_t):
        param.requires_grad_(False)

    beta = torch.zeros(len(selected_indices), dtype=torch.float64, requires_grad=True)
    intervened = apply_additive_sae_delta(
        residual=residual,
        selected_indices=list(selected_indices),
        scales=scales_t,
        beta=beta,
        encoder_weight=w_enc,
        encoder_bias=b_enc,
        decoder_weight=w_dec,
    )
    logits = head @ intervened
    margin = logits[0] - logits[1]
    loss = torch.nn.functional.softplus(-margin)
    autograd_j = torch.autograd.grad(loss, beta)[0]

    # Activity guard for FD (DEC-021).
    latents = torch.relu(residual @ w_enc.T + b_enc)
    fd_step = 1e-8
    for idx, scale in zip(selected_indices, scales):
        if float(latents[idx]) > 0:
            assert fd_step * float(scale) < float(latents[idx])

    def loss_at(beta_vec: list[float]) -> float:
        with torch.no_grad():
            intervened = apply_additive_sae_delta(
                residual=residual,
                selected_indices=list(selected_indices),
                scales=list(scales),
                beta=beta_vec,
                encoder_weight=w_enc,
                encoder_bias=b_enc,
                decoder_weight=w_dec,
            )
            logits = head @ intervened
            margin = logits[0] - logits[1]
            return float(torch.nn.functional.softplus(-margin).item())

    base = loss_at([0.0] * len(selected_indices))
    fd_j: list[float] = []
    for j in range(len(selected_indices)):
        beta_m = [0.0] * len(selected_indices)
        beta_m[j] = -fd_step
        fd_j.append((loss_at(beta_m) - base) / (-fd_step))
    return [float(v) for v in autograd_j.tolist()], fd_j


@pytest.mark.integration
def test_e2e_toy__row_order_and_batch_size__do_not_change_results() -> None:
    """E2E-005: permuting rows or changing batch size leaves objective unchanged."""
    from epistemic_sycophancy.evaluation.toy_e2e import (
        evaluate_toy_e2e_objective,
        evaluate_toy_e2e_objective_batched,
    )

    kwargs = dict(
        order_regime="CF",
        beta=KNOWN_BETA,
        selected_indices=KNOWN_SELECTED,
        scales=KNOWN_SCALES,
        tau=1.0,
        w_r=0.5,
        w_u=0.5,
        delta_n=0.1,
        delta_c=0.1,
        lambda_n=0.1,
        lambda_c=0.1,
        lambda_beta=0.1,
    )
    reference = evaluate_toy_e2e_objective(**kwargs)
    permuted = evaluate_toy_e2e_objective_batched(
        **kwargs, row_permutation=(3, 0, 5, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13), batch_size=14
    )
    batched = evaluate_toy_e2e_objective_batched(**kwargs, batch_size=3)
    for result in (permuted, batched):
        assert result.l_total == pytest.approx(reference.l_total, abs=1e-12, rel=1e-12)
        assert result.l_resist == pytest.approx(reference.l_resist, abs=1e-12, rel=1e-12)
        assert result.l_recover == pytest.approx(reference.l_recover, abs=1e-12, rel=1e-12)


@pytest.mark.integration
def test_e2e_toy__projected_adam__reduces_toy_total_loss_without_violating_bounds() -> (
    None
):
    """E2E-006: projected Adam reduces DEC-046 objective and stays in bounds."""
    from epistemic_sycophancy.evaluation.toy_e2e import run_toy_e2e_projected_adam

    result = run_toy_e2e_projected_adam(
        order_regime="IF",
        selected_indices=KNOWN_SELECTED,
        scales=KNOWN_SCALES,
        beta0=(0.0, 0.0, 0.0),
        n_steps=20,
        adam_lr=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_microbatch_questions=1,
        beta_lower=-2.0,
        beta_upper=0.0,
        tau=1.0,
        w_r=0.5,
        w_u=0.5,
        delta_n=0.1,
        delta_c=0.1,
        lambda_n=0.1,
        lambda_c=0.1,
        lambda_beta=0.01,
    )
    assert result.l_final < result.l_initial
    assert all(-2.0 <= v <= 0.0 for v in result.beta_final)
    assert all(-2.0 <= v <= 0.0 for beta in result.beta_trajectory for v in beta)


@pytest.mark.integration
def test_e2e_toy__cross_order__uses_correct_prompts_and_partitions_for_all_nine_cells() -> (
    None
):
    """E2E-007: 3×3 matrix uses eval-order prompts and frozen partitions."""
    from epistemic_sycophancy.evaluation.toy_e2e import run_toy_e2e_cross_order_matrix
    from epistemic_sycophancy.metrics.baseline_partition import (
        freeze_baseline_partition_artifact,
    )

    betas = {
        "CF": list(KNOWN_BETA),
        "IF": list(KNOWN_BETA),
        "RO": list(KNOWN_BETA),
    }
    cells = run_toy_e2e_cross_order_matrix(
        betas_by_optimized_under=betas,
        selected_indices=KNOWN_SELECTED,
        scales=KNOWN_SCALES,
    )
    assert len(cells) == 9
    baselines = {
        order: run_toy_e2e_baseline(order_regime=order) for order in ("CF", "IF", "RO")
    }
    for cell in cells:
        eval_baseline = baselines[cell.evaluated_under]
        assert cell.n_q_plus == len(eval_baseline.partition.q_plus)
        assert cell.n_q_minus == len(eval_baseline.partition.q_minus)
        assert cell.beta == tuple(betas[cell.optimized_under])
        expected_fp = freeze_baseline_partition_artifact(
            partition=eval_baseline.partition,
            model_revision_hash="toy-e2e-dec046",
            prompt_template_hash="toy-e2e-dec046",
            order_manifest_hash=f"toy-e2e-{cell.evaluated_under}",
            dataset_manifest_hash="toy-e2e-dec046",
        ).fingerprint
        assert cell.baseline_partition_fingerprint == expected_fp
        assert cell.evaluation_order_manifest_hash == f"toy-e2e-{cell.evaluated_under}"
        assert cell.optimization_order_manifest_hash == f"toy-e2e-{cell.optimized_under}"


@pytest.mark.integration
def test_e2e_toy__cross_order__uses_correct_prompts_and_partitions_for_all_nine_cells() -> (
    None
):
    """E2E-007: 3×3 matrix uses eval-order prompts and frozen partitions."""
    from epistemic_sycophancy.evaluation.toy_e2e import run_toy_e2e_cross_order_matrix
    from epistemic_sycophancy.metrics.baseline_partition import (
        freeze_baseline_partition_artifact,
    )

    betas = {
        "CF": list(KNOWN_BETA),
        "IF": list(KNOWN_BETA),
        "RO": list(KNOWN_BETA),
    }
    cells = run_toy_e2e_cross_order_matrix(
        betas_by_optimized_under=betas,
        selected_indices=KNOWN_SELECTED,
        scales=KNOWN_SCALES,
    )
    assert len(cells) == 9
    baselines = {
        order: run_toy_e2e_baseline(order_regime=order) for order in ("CF", "IF", "RO")
    }
    for cell in cells:
        eval_baseline = baselines[cell.evaluated_under]
        assert cell.n_q_plus == len(eval_baseline.partition.q_plus)
        assert cell.n_q_minus == len(eval_baseline.partition.q_minus)
        assert cell.beta == tuple(betas[cell.optimized_under])
        expected_fp = freeze_baseline_partition_artifact(
            partition=eval_baseline.partition,
            model_revision_hash="toy-e2e-dec046",
            prompt_template_hash="toy-e2e-dec046",
            order_manifest_hash=f"toy-e2e-{cell.evaluated_under}",
            dataset_manifest_hash="toy-e2e-dec046",
        ).fingerprint
        assert cell.baseline_partition_fingerprint == expected_fp
        assert cell.evaluation_order_manifest_hash == f"toy-e2e-{cell.evaluated_under}"
        assert cell.optimization_order_manifest_hash == f"toy-e2e-{cell.optimized_under}"
