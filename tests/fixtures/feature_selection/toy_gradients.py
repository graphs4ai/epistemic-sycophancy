"""Deterministic float64 fixtures for Phase F gradient work (DEC-020)."""

from __future__ import annotations

import torch

DTYPE = torch.float64

# Three rendered prompts of unequal length inside a padded [B, T] batch.
PROMPT_LENGTHS: tuple[int, ...] = (3, 5, 2)
SEQ_LEN = 5
D_MODEL = 2


def _content_vector(row: int, position_in_prompt: int) -> list[float]:
    """Distinct residual content so every (row, position) is identifiable."""
    value = 10.0 * (row + 1) + position_in_prompt
    return [value, -value]


def final_token_contents(*, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Hand-listed residual content at each prompt's final token, [B, D]."""
    return torch.tensor(
        [
            _content_vector(row, length - 1)
            for row, length in enumerate(PROMPT_LENGTHS)
        ],
        dtype=dtype,
    )


def right_padded_batch(
    *, dtype: torch.dtype = DTYPE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded residual [B, T, D] and attention mask [B, T]."""
    residual = torch.zeros(len(PROMPT_LENGTHS), SEQ_LEN, D_MODEL, dtype=dtype)
    mask = torch.zeros(len(PROMPT_LENGTHS), SEQ_LEN, dtype=torch.bool)
    for row, length in enumerate(PROMPT_LENGTHS):
        for position in range(length):
            residual[row, position] = torch.tensor(
                _content_vector(row, position), dtype=dtype
            )
            mask[row, position] = True
    return residual, mask


def left_padded_batch(
    *, dtype: torch.dtype = DTYPE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-padded residual [B, T, D] and attention mask [B, T].

    Carries the same prompt content as :func:`right_padded_batch`, so the
    selected final-token states must agree between the two paddings.
    """
    residual = torch.zeros(len(PROMPT_LENGTHS), SEQ_LEN, D_MODEL, dtype=dtype)
    mask = torch.zeros(len(PROMPT_LENGTHS), SEQ_LEN, dtype=torch.bool)
    for row, length in enumerate(PROMPT_LENGTHS):
        offset = SEQ_LEN - length
        for position in range(length):
            residual[row, offset + position] = torch.tensor(
                _content_vector(row, position), dtype=dtype
            )
            mask[row, offset + position] = True
    return residual, mask
