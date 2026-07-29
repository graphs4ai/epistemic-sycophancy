"""WIRE-002: nonzero-β hooked residuals differ from unhooked on real stack."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec
from epistemic_sycophancy.stack.intervention_stack import load_stack

_PROMPTS = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "real_model"
    / "gemma_smoke_prompts.json"
)
_SAE_PIN = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "real_model"
    / "gemmascope2_4b_it_resid_post_pin.json"
)
_MODEL_PIN = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "real_model"
    / "gemma3_continuation_token_ids.json"
)


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_stack__nonzero_beta__hooked_residuals_differ_from_unhooked() -> None:
    """WIRE-002: nonzero β changes residuals vs unhooked on Gemma+GemmaScope2."""
    if not torch.cuda.is_available():
        pytest.fail(
            "CUDA unavailable; WIRE-002 must be recorded blocked (DEC-047), never faked on CPU"
        )
    prompts_meta = json.loads(_PROMPTS.read_text())
    sae_pin = json.loads(_SAE_PIN.read_text())
    model_pin = json.loads(_MODEL_PIN.read_text())
    # Single layer smoke for speed.
    layer = 17

    cfg = ExperimentStackConfig(
        model=ModelSpec(
            hf_id=model_pin["model_id"],
            revision=model_pin["revision"],
            tokenizer_revision=model_pin["tokenizer_revision"],
            dtype="bfloat16",
            device_policy="cuda_required",
        ),
        sae=SaeSiteSpec(
            release=sae_pin["release"],
            site=sae_pin["site"],
            width=sae_pin["width"],
            l0=sae_pin["l0"],
            layers=(layer,),
        ),
        hooks=HookSpec(
            token_scope=prompts_meta["token_scope"],
            resolver_id="gemma3_resid_post",
            k=None,
        ),
    )
    stack = load_stack(cfg)
    raw_prompt = prompts_meta["prompts"][0]
    if prompts_meta["use_chat_template"]:
        text = stack.tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = raw_prompt

    unhooked = stack.capture_layer_residuals(texts=[text], layers=(layer,))
    encoded = stack.tokenizer([text], return_tensors="pt", padding=True)
    prompt_lengths = [int(encoded["attention_mask"].sum(dim=1)[0].item())]
    # Pick an active JumpReLU feature at the last prompt token.
    handle = stack.saes[layer]
    residual_vec = unhooked[layer][0, prompt_lengths[0] - 1].to(dtype=torch.float32)
    w_enc = handle.sae.W_enc.detach().to(dtype=torch.float32, device=residual_vec.device)
    b_enc = handle.sae.b_enc.detach().to(dtype=torch.float32, device=residual_vec.device)
    if w_enc.shape[0] == residual_vec.shape[-1]:
        pre = residual_vec @ w_enc + b_enc
    else:
        pre = residual_vec @ w_enc.T + b_enc
    threshold = handle.sae.threshold.detach().to(
        dtype=torch.float32, device=residual_vec.device
    )
    latents = pre * (pre > threshold)
    active = torch.nonzero(latents > 0, as_tuple=False).flatten()
    assert active.numel() > 0, "expected at least one active JumpReLU latent"
    feature_id = int(active[0].item())
    selected_keys = ((layer, feature_id),)
    scales = (1.0,)
    beta = (-1.0,)

    from epistemic_sycophancy.stack.hooks import install_multi_layer_hooks

    with install_multi_layer_hooks(
        model=stack.model,
        resolver_id=stack.config.hooks.resolver_id,
        saes=stack.saes,
        selected_keys=selected_keys,
        scales=scales,
        beta=beta,
        token_scope=stack.config.hooks.token_scope,
        prompt_lengths=prompt_lengths,
        k=stack.config.hooks.k,
    ):
        hooked = stack.capture_layer_residuals(texts=[text], layers=(layer,))

    assert not torch.equal(unhooked[layer], hooked[layer]), (
        "nonzero β must change hooked residuals relative to unhooked"
    )
