"""WIRE-003: InterventionStack.install_hooks passes real prompt_lengths."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec
from epistemic_sycophancy.stack.intervention_stack import InterventionStack


@pytest.mark.unit
def test_intervention_stack__install_hooks__passes_real_prompt_lengths() -> None:
    """WIRE-003: install_hooks requires and forwards prompt_lengths (not [1])."""
    cfg = ExperimentStackConfig(
        model=ModelSpec(
            hf_id="google/gemma-3-4b-it",
            revision="093f9f388b31de276ce2de164bdc2081324b9767",
            tokenizer_revision="093f9f388b31de276ce2de164bdc2081324b9767",
            dtype="bfloat16",
            device_policy="cuda_required",
        ),
        sae=SaeSiteSpec(
            release="gemma-scope-2-4b-it-res",
            site="resid_post",
            width="width_65k",
            l0="l0_medium",
            layers=(17,),
        ),
        hooks=HookSpec(
            token_scope="last_prompt_token",
            resolver_id="gemma3_resid_post",
            k=None,
        ),
    )
    stack = InterventionStack(
        config=cfg,
        loaded=SimpleNamespace(model=SimpleNamespace(), tokenizer=None, device="cpu"),
        saes={17: SimpleNamespace()},
    )
    captured: dict[str, Any] = {}

    @contextmanager
    def fake_install(**kwargs: Any) -> Iterator[None]:
        captured.update(kwargs)
        yield

    with patch(
        "epistemic_sycophancy.stack.intervention_stack.install_multi_layer_hooks",
        fake_install,
    ):
        with stack.install_hooks(
            selected_keys=((17, 0),),
            scales=(1.0,),
            beta=(0.0,),
            prompt_lengths=(12, 8),
        ):
            pass

    assert captured["prompt_lengths"] == (12, 8)
    assert captured["prompt_lengths"] != [1]
