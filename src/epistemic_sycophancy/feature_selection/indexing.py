"""Prompt-specific final-token indexing (Phase F FEAT-001)."""

from __future__ import annotations

import torch


def final_prompt_token_index(*, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return t* for each row: its own last non-padding position.

    Shapes:
      attention_mask: [batch, seq_len] bool or 0/1
      returns:        [batch] long

    Works for both left and right padding because it locates the last
    unmasked position of each individual rendered prompt.
    """
    mask = attention_mask.to(torch.int64)
    seq_len = mask.shape[-1]
    reversed_first_active = torch.argmax(mask.flip(-1), dim=-1)
    return seq_len - 1 - reversed_first_active


def select_final_token_states(
    *,
    residual: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather the residual state at each prompt's t*.

    Shapes:
      residual:       [batch, seq_len, d_model]
      attention_mask: [batch, seq_len]
      returns:        [batch, d_model]
    """
    index = final_prompt_token_index(attention_mask=attention_mask)
    gather_index = index.view(-1, 1, 1).expand(-1, 1, residual.shape[-1])
    return residual.gather(1, gather_index).squeeze(1)
