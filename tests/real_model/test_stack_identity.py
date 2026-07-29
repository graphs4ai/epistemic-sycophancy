"""RUN-006: multi-layer β=0 identity with InterventionStack hooks."""

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
def test_stack__multi_layer_hooks_beta_zero__residuals_match_unhooked() -> None:
    """RUN-006: with multi-layer hooks and β=0, residuals equal unhooked forward."""
    prompts_meta = json.loads(_PROMPTS.read_text())
    sae_pin = json.loads(_SAE_PIN.read_text())
    model_pin = json.loads(_MODEL_PIN.read_text())
    layers = tuple(prompts_meta["layers"])

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
            layers=layers,
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

    unhooked = stack.capture_layer_residuals(texts=[text], layers=layers)
    selected_keys = tuple((layer, 0) for layer in layers)
    scales = tuple(1.0 for _ in layers)
    beta = tuple(0.0 for _ in layers)
    with stack.install_hooks(
        selected_keys=selected_keys,
        scales=scales,
        beta=beta,
    ):
        hooked = stack.capture_layer_residuals(texts=[text], layers=layers)

    for layer in layers:
        assert torch.equal(unhooked[layer], hooked[layer]), (
            f"β=0 identity failed at layer {layer}"
        )
