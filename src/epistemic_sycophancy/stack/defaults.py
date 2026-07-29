"""Default first-study ExperimentStackConfig (Phase K RUN-015)."""

from __future__ import annotations

from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec

# DEC-049 / DEC-051 first-study pins.
_DEFAULT_MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
_DEFAULT_SAE_RELEASE = "gemma-scope-2-4b-it-res"
_DEFAULT_LAYERS = (9, 17, 22, 29)


def first_study_stack_config() -> ExperimentStackConfig:
    """Config factory: Gemma-3-4B-IT + all four resid_post width_65k/l0_medium layers."""
    return ExperimentStackConfig(
        model=ModelSpec(
            hf_id="google/gemma-3-4b-it",
            revision=_DEFAULT_MODEL_REVISION,
            tokenizer_revision=_DEFAULT_MODEL_REVISION,
            dtype="bfloat16",
            device_policy="cuda_required",
        ),
        sae=SaeSiteSpec(
            release=_DEFAULT_SAE_RELEASE,
            site="resid_post",
            width="width_65k",
            l0="l0_medium",
            layers=_DEFAULT_LAYERS,
        ),
        hooks=HookSpec(
            token_scope="last_prompt_token",
            resolver_id="gemma3_resid_post",
            k=None,
        ),
    )
