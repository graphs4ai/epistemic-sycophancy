"""Deterministic toy linear SAE fixtures (DEC-016)."""

from __future__ import annotations

import torch

D_MODEL = 2
N_FEATURES = 3

# Decoder rows (no unit-normalization): f0=[1,0], f1=[0,2], f2=[1,1]
DECODER_ROWS: list[list[float]] = [
    [1.0, 0.0],
    [0.0, 2.0],
    [1.0, 1.0],
]


def decoder_weight(*, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Return W_dec with shape [n_features, d_model]."""
    return torch.tensor(DECODER_ROWS, dtype=dtype)


def encode(
    residual: torch.Tensor,
    *,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
) -> torch.Tensor:
    """Encode residual with linear map + ReLU: z = ReLU(x W_enc^T + b)."""
    return torch.relu(residual @ encoder_weight.T + encoder_bias)


def decode(latents: torch.Tensor, *, decoder_weight: torch.Tensor) -> torch.Tensor:
    """Linear decode: x_hat = z @ W_dec."""
    return latents @ decoder_weight


def imperfect_encoder_params(
    *, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed imperfect encoder so decode(encode(x)) != x for typical x.

    W_enc is not a left inverse of W_dec; bias breaks zero-centered reconstruction.
    """
    encoder_weight = torch.tensor(
        [
            [0.5, 0.0],
            [0.0, 0.25],
            [0.25, 0.25],
        ],
        dtype=dtype,
    )
    encoder_bias = torch.tensor([0.1, -0.2, 0.05], dtype=dtype)
    return encoder_weight, encoder_bias


def logit_head_weight(*, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Fixed residual → logits [A, B] map, shape [2, d_model]."""
    return torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=dtype,
    )


def toy_logits_from_residual(
    residual: torch.Tensor,
    *,
    head_weight: torch.Tensor,
    prompt_lengths: list[int],
) -> torch.Tensor:
    """Score A/B logits from the last prompt token residual.

    residual: [B, T, D]; returns [B, 2] logits.
    """
    batch_logits = []
    for batch_index, prompt_length in enumerate(prompt_lengths):
        last = residual[batch_index, prompt_length - 1]
        batch_logits.append(head_weight @ last)
    return torch.stack(batch_logits, dim=0)
