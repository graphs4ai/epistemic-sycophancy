"""SAE identity integration tests (Phase E SAE-008/009/013)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from epistemic_sycophancy.intervention.hooks import (
    apply_delta_with_token_scope,
    build_token_scope_mask,
)
from epistemic_sycophancy.intervention.sae_delta import apply_additive_sae_delta
from epistemic_sycophancy.scoring.margins import margin_preference, truthful_margin

_TOY_SAE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "intervention" / "toy_sae.py"
)
_spec = importlib.util.spec_from_file_location("toy_sae_identity", _TOY_SAE_PATH)
assert _spec is not None and _spec.loader is not None
_toy_sae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_toy_sae)
decoder_weight = _toy_sae.decoder_weight
imperfect_encoder_params = _toy_sae.imperfect_encoder_params
logit_head_weight = _toy_sae.logit_head_weight
toy_logits_from_residual = _toy_sae.toy_logits_from_residual


def _hooked_residual_last_token(
    residual: torch.Tensor,
    *,
    prompt_lengths: list[int],
    selected_indices: list[int],
    scales: list[float],
    beta: list[float],
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    token_scope: str,
) -> torch.Tensor:
    """Apply per-row additive SAE delta at configured token scope."""
    batch_size, seq_len, d_model = residual.shape
    delta = torch.zeros_like(residual)
    for batch_index, prompt_length in enumerate(prompt_lengths):
        last = residual[batch_index, prompt_length - 1]
        intervened = apply_additive_sae_delta(
            residual=last,
            selected_indices=selected_indices,
            scales=scales,
            beta=beta,
            encoder_weight=encoder_weight,
            encoder_bias=encoder_bias,
            decoder_weight=decoder_weight,
        )
        # Place per-token delta only at last prompt token; mask enforces scope.
        delta[batch_index, prompt_length - 1] = intervened - last
    mask = build_token_scope_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        prompt_lengths=prompt_lengths,
        token_scope=token_scope,
    )
    return apply_delta_with_token_scope(residual=residual, delta=delta, mask=mask)


@pytest.mark.integration
def test_intervention__beta_zero__matches_unmodified_logits() -> None:
    """SAE-008: hook@β=0 logits match unhooked logits (DEC-017 bf16 tols)."""
    dtype = torch.bfloat16
    prompt_lengths = [3]
    residual = torch.tensor(
        [
            [
                [0.5, -0.25],
                [1.0, 0.5],
                [0.75, -1.0],
            ]
        ],
        dtype=dtype,
    )
    w_dec = decoder_weight(dtype=dtype)
    w_enc, b_enc = imperfect_encoder_params(dtype=dtype)
    head = logit_head_weight(dtype=dtype)

    unhooked = toy_logits_from_residual(
        residual, head_weight=head, prompt_lengths=prompt_lengths
    )
    hooked_residual = _hooked_residual_last_token(
        residual,
        prompt_lengths=prompt_lengths,
        selected_indices=[0, 1, 2],
        scales=[1.0, 1.0, 1.0],
        beta=[0.0, 0.0, 0.0],
        encoder_weight=w_enc,
        encoder_bias=b_enc,
        decoder_weight=w_dec,
        token_scope="last_prompt_token",
    )
    hooked = toy_logits_from_residual(
        hooked_residual, head_weight=head, prompt_lengths=prompt_lengths
    )
    assert torch.allclose(hooked, unhooked, atol=5e-3, rtol=1e-4)


@pytest.mark.integration
def test_intervention__beta_zero__matches_unmodified_margins_and_labels() -> None:
    """SAE-009: β=0 margins within tol; A/B decisions exact (DEC-001/017)."""
    dtype = torch.bfloat16
    prompt_lengths = [3, 3]
    residual = torch.tensor(
        [
            [
                [0.5, -0.25],
                [1.0, 0.5],
                [0.75, -1.0],
            ],
            [
                [-0.5, 1.0],
                [0.25, 0.5],
                [-0.5, 1.25],
            ],
        ],
        dtype=dtype,
    )
    w_dec = decoder_weight(dtype=dtype)
    w_enc, b_enc = imperfect_encoder_params(dtype=dtype)
    head = logit_head_weight(dtype=dtype)
    truthful_labels = ["A", "B"]  # CF then IF

    unhooked_logits = toy_logits_from_residual(
        residual, head_weight=head, prompt_lengths=prompt_lengths
    )
    hooked_residual = _hooked_residual_last_token(
        residual,
        prompt_lengths=prompt_lengths,
        selected_indices=[0, 1, 2],
        scales=[1.0, 1.0, 1.0],
        beta=[0.0, 0.0, 0.0],
        encoder_weight=w_enc,
        encoder_bias=b_enc,
        decoder_weight=w_dec,
        token_scope="last_prompt_token",
    )
    hooked_logits = toy_logits_from_residual(
        hooked_residual, head_weight=head, prompt_lengths=prompt_lengths
    )

    for row in range(2):
        score_a_u = float(unhooked_logits[row, 0].item())
        score_b_u = float(unhooked_logits[row, 1].item())
        score_a_h = float(hooked_logits[row, 0].item())
        score_b_h = float(hooked_logits[row, 1].item())
        margin_u = truthful_margin(
            score_a=score_a_u,
            score_b=score_b_u,
            truthful_label=truthful_labels[row],
        )
        margin_h = truthful_margin(
            score_a=score_a_h,
            score_b=score_b_h,
            truthful_label=truthful_labels[row],
        )
        assert margin_h == pytest.approx(margin_u, abs=5e-3, rel=1e-4)
        assert margin_preference(margin_h) == margin_preference(margin_u)
