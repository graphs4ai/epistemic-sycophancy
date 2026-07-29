"""Experiment stack config validation (Phase K RUN-001)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import (
    GEMMASCOPE2_4B_IT_RESID_POST_SUBSET_LAYERS,
    InvalidSaeSiteSpec,
    SaeSiteSpec,
)
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


@pytest.mark.unit
def test_stack_config__sae_site_spec__rejects_empty_duplicate_or_unknown_layers() -> None:
    """RUN-001: layers nonempty, unique, ⊆ known resid_post subset; width/L0 explicit."""
    with pytest.raises(InvalidSaeSiteSpec):
        SaeSiteSpec(
            release="gemma-scope-2-4b-it-resid_post",
            site="resid_post",
            width="width_65k",
            l0="l0_medium",
            layers=(),
        )

    with pytest.raises(InvalidSaeSiteSpec):
        SaeSiteSpec(
            release="gemma-scope-2-4b-it-resid_post",
            site="resid_post",
            width="width_65k",
            l0="l0_medium",
            layers=(9, 9, 17),
        )

    with pytest.raises(InvalidSaeSiteSpec):
        SaeSiteSpec(
            release="gemma-scope-2-4b-it-resid_post",
            site="resid_post",
            width="width_65k",
            l0="l0_medium",
            layers=(9, 12),
        )

    with pytest.raises(InvalidSaeSiteSpec):
        SaeSiteSpec(
            release="gemma-scope-2-4b-it-resid_post",
            site="resid_post",
            width=None,  # type: ignore[arg-type]
            l0="l0_medium",
            layers=(9,),
        )

    with pytest.raises(InvalidSaeSiteSpec):
        SaeSiteSpec(
            release="gemma-scope-2-4b-it-resid_post",
            site="resid_post",
            width="width_65k",
            l0=None,  # type: ignore[arg-type]
            layers=(9,),
        )

    valid = SaeSiteSpec(
        release="gemma-scope-2-4b-it-resid_post",
        site="resid_post",
        width="width_65k",
        l0="l0_medium",
        layers=(9, 17, 22, 29),
    )
    assert valid.layers == (9, 17, 22, 29)
    assert set(valid.layers) <= GEMMASCOPE2_4B_IT_RESID_POST_SUBSET_LAYERS
    assert valid.width == "width_65k"
    assert valid.l0 == "l0_medium"

    single = SaeSiteSpec(
        release="gemma-scope-2-4b-it-resid_post",
        site="resid_post",
        width="width_65k",
        l0="l0_medium",
        layers=(17,),
    )
    assert single.layers == (17,)

    model = ModelSpec(
        hf_id="google/gemma-3-4b-it",
        revision="pending",
        tokenizer_revision="pending",
        dtype="bfloat16",
        device_policy="cuda_if_available",
    )
    hooks = HookSpec(
        token_scope="last_prompt_token",
        resolver_id="gemma3_resid_post",
        k=None,
    )
    stack = ExperimentStackConfig(model=model, sae=valid, hooks=hooks)
    assert stack.sae.layers == (9, 17, 22, 29)
    assert stack.hooks.token_scope == "last_prompt_token"
