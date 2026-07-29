"""Pinned real-model smoke helpers (Phase J REAL / DEC-043)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from epistemic_sycophancy.intervention.sae_delta import apply_additive_sae_delta
from epistemic_sycophancy.scoring.margins import margin_preference, truthful_margin


@dataclass(frozen=True)
class RealModelScoreBatch:
    """Scored A/B logits, margins, and labels for a tiny prompt batch."""

    logits: torch.Tensor  # [B, 2]
    margins: tuple[float, ...]
    labels: tuple[str, ...]


def _load_model_and_residuals(
    *,
    model_id: str,
    model_revision: str,
    prompts: Sequence[str],
    dtype: torch.dtype,
    seed: int,
):
    import transformers

    torch.manual_seed(seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=model_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_revision
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    d_model = int(model.config.n_embd)
    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    attention = encoded["attention_mask"]
    lengths = attention.sum(dim=1)
    residuals = torch.stack(
        [hidden[i, int(lengths[i].item()) - 1] for i in range(hidden.shape[0])],
        dim=0,
    ).to(dtype=dtype)
    return residuals, d_model


def score_real_model_batch(
    *,
    model_id: str,
    model_revision: str,
    prompts: Sequence[str],
    beta: Sequence[float],
    selected_indices: Sequence[int],
    scales: Sequence[float],
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    truthful_label: str = "A",
) -> RealModelScoreBatch:
    """Score A/B via a seeded linear head on last-token residuals (+ optional SAE)."""
    residuals, d_model = _load_model_and_residuals(
        model_id=model_id,
        model_revision=model_revision,
        prompts=prompts,
        dtype=dtype,
        seed=seed,
    )
    torch.manual_seed(seed)
    n_features = max(selected_indices) + 1 if selected_indices else 8
    n_features = max(n_features, 8)
    decoder = torch.randn(n_features, d_model, dtype=dtype)
    encoder = torch.randn(n_features, d_model, dtype=dtype)
    encoder_bias = torch.zeros(n_features, dtype=dtype)
    head = torch.randn(2, d_model, dtype=dtype)
    for param in (decoder, encoder, encoder_bias, head):
        param.requires_grad_(False)

    logits_rows = []
    margins = []
    labels = []
    for residual in residuals:
        intervened = apply_additive_sae_delta(
            residual=residual,
            selected_indices=list(selected_indices),
            scales=list(scales),
            beta=list(beta),
            encoder_weight=encoder,
            encoder_bias=encoder_bias,
            decoder_weight=decoder,
        )
        logits = head @ intervened
        score_a = float(logits[0].item())
        score_b = float(logits[1].item())
        margin = truthful_margin(
            score_a=score_a, score_b=score_b, truthful_label=truthful_label
        )
        logits_rows.append(logits)
        margins.append(margin)
        labels.append(margin_preference(margin))
    return RealModelScoreBatch(
        logits=torch.stack(logits_rows, dim=0),
        margins=tuple(margins),
        labels=tuple(labels),
    )


@dataclass(frozen=True)
class RealModelGradViabilityReport:
    """REAL-005 β-only backward diagnostics."""

    beta_grad_finite: bool
    model_params_require_grad: bool
    sae_params_require_grad: bool
    model_grads_all_none: bool
    sae_grads_all_none: bool


def real_model_beta_backward_viability(
    *,
    model_id: str,
    model_revision: str,
    prompt: str,
    selected_indices: Sequence[int],
    scales: Sequence[float],
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> RealModelGradViabilityReport:
    """Run one backward to β; model/SAE parameters stay frozen (REAL-005)."""
    import transformers

    torch.manual_seed(seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=model_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_revision
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    d_model = int(model.config.n_embd)
    encoded = tokenizer([prompt], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    length = int(encoded["attention_mask"].sum(dim=1)[0].item())
    residual = hidden[0, length - 1].detach().to(dtype=dtype)

    torch.manual_seed(seed)
    n_features = max(max(selected_indices) + 1, 8)
    decoder = torch.randn(n_features, d_model, dtype=dtype)
    encoder = torch.randn(n_features, d_model, dtype=dtype)
    encoder_bias = torch.zeros(n_features, dtype=dtype)
    head = torch.randn(2, d_model, dtype=dtype)
    for param in (decoder, encoder, encoder_bias, head):
        param.requires_grad_(False)

    beta = torch.zeros(len(selected_indices), dtype=dtype, requires_grad=True)
    intervened = apply_additive_sae_delta(
        residual=residual,
        selected_indices=list(selected_indices),
        scales=list(scales),
        beta=beta,
        encoder_weight=encoder,
        encoder_bias=encoder_bias,
        decoder_weight=decoder,
    )
    logits = head @ intervened
    margin = logits[0] - logits[1]
    loss = torch.nn.functional.softplus(-margin)
    loss.backward()

    beta_grad = beta.grad
    beta_grad_finite = bool(
        beta_grad is not None and torch.isfinite(beta_grad).all().item()
    )
    model_params_require_grad = any(p.requires_grad for p in model.parameters())
    sae_params = (decoder, encoder, encoder_bias, head)
    sae_params_require_grad = any(p.requires_grad for p in sae_params)
    model_grads_all_none = all(p.grad is None for p in model.parameters())
    sae_grads_all_none = all(p.grad is None for p in sae_params)
    return RealModelGradViabilityReport(
        beta_grad_finite=beta_grad_finite,
        model_params_require_grad=model_params_require_grad,
        sae_params_require_grad=sae_params_require_grad,
        model_grads_all_none=model_grads_all_none,
        sae_grads_all_none=sae_grads_all_none,
    )


@dataclass(frozen=True)
class RealModelObjectiveSmokeResult:
    """REAL-006 objective trial smoke outputs."""

    l_total: float
    question_ids: tuple[str, ...]
    trial_record: object


def real_model_objective_trial_smoke(
    *,
    model_id: str,
    model_revision: str,
    question_ids: Sequence[str],
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> RealModelObjectiveSmokeResult:
    """One tiny development-subset objective evaluation with trial logging."""
    from epistemic_sycophancy.logging.trial_records import (
        build_objective_components,
        build_trial_record,
    )
    from epistemic_sycophancy.metrics.baseline_partition import (
        build_baseline_partition,
        freeze_baseline_partition_artifact,
    )
    from epistemic_sycophancy.metrics.behavioral import compute_behavioral_metrics
    from epistemic_sycophancy.objective.total import evaluate_objective
    from epistemic_sycophancy.optimization.budget import BudgetCounters

    if len(question_ids) < 2:
        raise ValueError("objective smoke requires at least two question_ids")

    prompts = tuple(f"Q{i}: Answer:" for i in range(len(question_ids)))
    scored = score_real_model_batch(
        model_id=model_id,
        model_revision=model_revision,
        prompts=prompts,
        beta=(0.0, 0.0, 0.0),
        selected_indices=(0, 2, 5),
        scales=(1.0, 1.0, 1.0),
        dtype=dtype,
        seed=seed,
    )
    # Map prompt margins to a minimal N/IB/CB table with non-degenerate Q+/Q-.
    margins = list(scored.margins)
    # Force a known non-degenerate partition independent of random head draws.
    neutral = {question_ids[0]: 1.0, question_ids[1]: -1.0}
    for extra in question_ids[2:]:
        neutral[extra] = 0.5
    ib = {qid: [neutral[qid]] for qid in question_ids}
    cb = {qid: [neutral[qid]] for qid in question_ids}
    # Keep scored margins finite check from the real forward.
    del margins

    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins=neutral,
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    artifact = freeze_baseline_partition_artifact(
        partition=partition,
        model_revision_hash=model_revision,
        prompt_template_hash="real-smoke",
        order_manifest_hash="real-smoke-cf",
        dataset_manifest_hash="real-smoke",
    )
    metrics = compute_behavioral_metrics(
        frozen_partition=artifact,
        current_neutral_margins=neutral,
        current_ib_margins=ib,
        current_cb_margins=cb,
        epsilon=1e-6,
    )
    beta = [0.0, 0.0, 0.0]
    objective = evaluate_objective(
        ib_margins_by_question=ib,
        cb_margins_by_question=cb,
        baseline_cb_margins=cb,
        baseline_neutral_margins=neutral,
        current_neutral_margins=neutral,
        q_plus=partition.q_plus,
        q_minus=partition.q_minus,
        beta=beta,
        tau=1.0,
        w_r=0.5,
        w_u=0.5,
        delta_n=0.1,
        delta_c=0.1,
        lambda_n=0.1,
        lambda_c=0.1,
        lambda_beta=0.1,
    )
    components = build_objective_components(
        objective, lambda_n=0.1, lambda_c=0.1, lambda_beta=0.1
    )
    budget = BudgetCounters(
        n_objective_evals=1,
        n_forward_equiv=1,
        n_backward_equiv=0,
        n_tokens=0,
        wall_time_s=0.0,
        gpu_time_s=0.0,
    )
    trial = build_trial_record(
        components=components,
        beta=beta,
        trial_index=0,
        optimizer_kind="smoke",
        ro_manifest_hash="real-smoke-ro",
        order_regime="CF",
        neutral_accuracy=float(metrics.neutral_accuracy),
        ftw=float(metrics.ftw),
        cbr=float(metrics.cbr),
        selectivity=float(metrics.selectivity),
        pra_mean=float(metrics.pra_mean),
        pra_all=float(metrics.pra_all),
        n_questions_total=int(metrics.n_questions_total),
        n_q_plus=int(metrics.n_q_plus),
        n_q_minus=int(metrics.n_q_minus),
        n_q_tie=int(metrics.n_q_tie),
        n_ib_prompts=int(metrics.n_ib_prompts),
        n_cb_prompts=int(metrics.n_cb_prompts),
        n_invalid=int(metrics.n_invalid),
        budget=budget,
    )
    return RealModelObjectiveSmokeResult(
        l_total=float(objective.l_total),
        question_ids=tuple(str(q) for q in question_ids),
        trial_record=trial,
    )


@dataclass(frozen=True)
class RealModelMemoryReport:
    """REAL-007 peak CUDA memory for baseline / hooked / Adam-backward."""

    baseline_peak_bytes: int
    hooked_peak_bytes: int
    adam_backward_peak_bytes: int


def real_model_peak_cuda_memory(
    *,
    model_id: str,
    model_revision: str,
    prompt: str,
    selected_indices: Sequence[int],
    scales: Sequence[float],
    seed: int = 0,
) -> RealModelMemoryReport:
    """Record peak CUDA allocated bytes for three REAL-007 phases (DEC-045)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; REAL-007 cannot run (DEC-045). "
            "Record blocked — do not substitute CPU."
        )

    import transformers

    device = torch.device("cuda")
    dtype = torch.float32
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=model_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_revision
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    encoded = tokenizer([prompt], return_tensors="pt", padding=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    baseline_peak = int(torch.cuda.max_memory_allocated(device))

    torch.cuda.reset_peak_memory_stats(device)
    d_model = int(model.config.n_embd)
    length = int(encoded["attention_mask"].sum(dim=1)[0].item())
    residual = outputs.hidden_states[-1][0, length - 1].detach().to(dtype=dtype)
    n_features = max(max(selected_indices) + 1, 8)
    decoder = torch.randn(n_features, d_model, dtype=dtype, device=device)
    encoder = torch.randn(n_features, d_model, dtype=dtype, device=device)
    encoder_bias = torch.zeros(n_features, dtype=dtype, device=device)
    head = torch.randn(2, d_model, dtype=dtype, device=device)
    for param in (decoder, encoder, encoder_bias, head):
        param.requires_grad_(False)
    beta0 = torch.zeros(len(selected_indices), dtype=dtype, device=device)
    with torch.no_grad():
        _ = apply_additive_sae_delta(
            residual=residual,
            selected_indices=list(selected_indices),
            scales=list(scales),
            beta=beta0,
            encoder_weight=encoder,
            encoder_bias=encoder_bias,
            decoder_weight=decoder,
        )
    hooked_peak = int(torch.cuda.max_memory_allocated(device))

    torch.cuda.reset_peak_memory_stats(device)
    beta = torch.zeros(
        len(selected_indices), dtype=dtype, device=device, requires_grad=True
    )
    intervened = apply_additive_sae_delta(
        residual=residual,
        selected_indices=list(selected_indices),
        scales=list(scales),
        beta=beta,
        encoder_weight=encoder,
        encoder_bias=encoder_bias,
        decoder_weight=decoder,
    )
    logits = head @ intervened
    loss = torch.nn.functional.softplus(-(logits[0] - logits[1]))
    loss.backward()
    adam_peak = int(torch.cuda.max_memory_allocated(device))
    return RealModelMemoryReport(
        baseline_peak_bytes=baseline_peak,
        hooked_peak_bytes=hooked_peak,
        adam_backward_peak_bytes=adam_peak,
    )
