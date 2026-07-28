"""SAE hook token-scope and batch tests (Phase E)."""

from __future__ import annotations

import pytest
import torch

from epistemic_sycophancy.intervention.hooks import (
    apply_delta_with_token_scope,
    build_token_scope_mask,
)


@pytest.mark.integration
def test_hook__configured_token_scope__modifies_only_intended_positions() -> None:
    """SAE-010: last / all / last-k prompt scopes; pad and answer positions unchanged."""
    dtype = torch.bfloat16
    batch_size = 2
    seq_len = 6
    d_model = 2
    prompt_lengths = [4, 3]  # answers/pad occupy indices >= prompt_length
    residual = torch.zeros(batch_size, seq_len, d_model, dtype=dtype)
    delta = torch.ones(batch_size, seq_len, d_model, dtype=dtype)

    # last_prompt_token
    mask_last = build_token_scope_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        prompt_lengths=prompt_lengths,
        token_scope="last_prompt_token",
    )
    out_last = apply_delta_with_token_scope(
        residual=residual, delta=delta, mask=mask_last
    )
    assert torch.equal(out_last[0, 3], torch.ones(d_model, dtype=dtype))
    assert torch.equal(out_last[0, 0], torch.zeros(d_model, dtype=dtype))
    assert torch.equal(out_last[0, 4], torch.zeros(d_model, dtype=dtype))  # answer/pad
    assert torch.equal(out_last[1, 2], torch.ones(d_model, dtype=dtype))
    assert torch.equal(out_last[1, 3], torch.zeros(d_model, dtype=dtype))

    # all_prompt_tokens
    mask_all = build_token_scope_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        prompt_lengths=prompt_lengths,
        token_scope="all_prompt_tokens",
    )
    out_all = apply_delta_with_token_scope(
        residual=residual, delta=delta, mask=mask_all
    )
    assert torch.equal(out_all[0, :4], torch.ones(4, d_model, dtype=dtype))
    assert torch.equal(out_all[0, 4:], torch.zeros(2, d_model, dtype=dtype))
    assert torch.equal(out_all[1, :3], torch.ones(3, d_model, dtype=dtype))
    assert torch.equal(out_all[1, 3:], torch.zeros(3, d_model, dtype=dtype))

    # last_k_prompt_tokens with k=2
    mask_k = build_token_scope_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        prompt_lengths=prompt_lengths,
        token_scope="last_k_prompt_tokens",
        k=2,
    )
    out_k = apply_delta_with_token_scope(residual=residual, delta=delta, mask=mask_k)
    # batch 0 prompt len 4 → positions 2,3
    assert torch.equal(out_k[0, 2], torch.ones(d_model, dtype=dtype))
    assert torch.equal(out_k[0, 3], torch.ones(d_model, dtype=dtype))
    assert torch.equal(out_k[0, 1], torch.zeros(d_model, dtype=dtype))
    assert torch.equal(out_k[0, 4], torch.zeros(d_model, dtype=dtype))
    # batch 1 prompt len 3 → positions 1,2
    assert torch.equal(out_k[1, 1], torch.ones(d_model, dtype=dtype))
    assert torch.equal(out_k[1, 2], torch.ones(d_model, dtype=dtype))
    assert torch.equal(out_k[1, 0], torch.zeros(d_model, dtype=dtype))
    assert torch.equal(out_k[1, 3], torch.zeros(d_model, dtype=dtype))
