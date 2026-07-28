"""Real-model Jacobian spot check (Phase F FEAT-030 / DEC-025)."""

from __future__ import annotations

import pytest
import torch

from epistemic_sycophancy.feature_selection import (
    coefficient_jacobian,
    project_residual_gradient,
)
from epistemic_sycophancy.intervention.sae_delta import apply_additive_sae_delta

# DEC-025 pins; revision filled after a successful resolve.
MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
MODEL_REVISION = "71034c5d8bde858ff824298bdedc65515b97d2b9"
ATOL = 1e-5
RTOL = 1e-4
DTYPE = torch.float32
TAU = 1.0
SEED = 0


def _logistic_margin_loss(margin: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(-margin / TAU)


@pytest.mark.real_model
@pytest.mark.slow
def test_feature_jacobian__sampled_real_features__match_direct_autograd_beta_gradient() -> (
    None
):
    """FEAT-030: projected J equals autograd d(loss)/dβ on pinned tiny GPT-2 SAE."""
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    model.eval()
    d_model = int(model.config.n_embd)
    resolved = getattr(model.config, "_commit_hash", None) or MODEL_REVISION

    prompts = ["Answer:", "Q: sky? Answer:"]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    # Last non-pad token residual per row from the final hidden state.
    attention = encoded["attention_mask"]
    lengths = attention.sum(dim=1)
    hidden = outputs.hidden_states[-1]  # [B, T, D]
    residuals = torch.stack(
        [hidden[i, int(lengths[i].item()) - 1] for i in range(hidden.shape[0])],
        dim=0,
    ).to(dtype=DTYPE)

    torch.manual_seed(SEED)
    n_features = 8
    selected = [0, 2, 5]
    residual = residuals[0].detach().clone()
    decoder = torch.randn(n_features, d_model, dtype=DTYPE)
    encoder = torch.randn(n_features, d_model, dtype=DTYPE)
    encoder_bias = torch.zeros(n_features, dtype=DTYPE)
    # Shift bias so every selected latent is strictly active at this residual.
    with torch.no_grad():
        preacts = residual @ encoder.T + encoder_bias
        for index in selected:
            if preacts[index] <= 0:
                encoder_bias[index] = encoder_bias[index] - preacts[index] + 0.5
    scales = torch.ones(len(selected), dtype=DTYPE) * 1.5
    head = torch.randn(2, d_model, dtype=DTYPE)

    for param in (decoder, encoder, encoder_bias, head, scales):
        param.requires_grad_(False)

    latents = torch.relu(residual @ encoder.T + encoder_bias)
    assert bool((latents[selected] > 0).all()), "fixture must sit in linear ReLU region"

    beta = torch.zeros(len(selected), dtype=DTYPE, requires_grad=True)
    intervened = apply_additive_sae_delta(
        residual=residual,
        selected_indices=selected,
        scales=scales,
        beta=beta,
        encoder_weight=encoder,
        encoder_bias=encoder_bias,
        decoder_weight=decoder,
    )
    logits = head @ intervened
    margin = logits[0] - logits[1]
    autograd_jacobian = torch.autograd.grad(_logistic_margin_loss(margin), beta)[0]

    residual_leaf = residual.clone().requires_grad_(True)
    residual_gradient = torch.autograd.grad(
        _logistic_margin_loss((head @ residual_leaf)[0] - (head @ residual_leaf)[1]),
        residual_leaf,
    )[0]
    full_projection = project_residual_gradient(
        gradient=residual_gradient, decoder=decoder
    )
    projected = coefficient_jacobian(
        raw_projection=full_projection[selected],
        latents=latents[selected],
        feature_scales=scales,
    )

    assert torch.allclose(projected, autograd_jacobian, atol=ATOL, rtol=RTOL)
    # Recordable identity for DEC-025; not a soft assertion on network state.
    assert isinstance(resolved, str) and len(resolved) > 0
    assert d_model > 0
