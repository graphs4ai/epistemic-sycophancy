"""RUN-015: config swap single-layer vs all-four without code forks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec
from epistemic_sycophancy.stack.defaults import first_study_stack_config
from epistemic_sycophancy.stack.intervention_stack import load_stack


@pytest.mark.unit
def test_stack__config_swap__single_layer_subset_vs_all_four_loads_without_code_fork() -> None:
    """RUN-015: same load_stack path for layers=[17] and all-four default."""
    calls: list[tuple[int, ...]] = []

    def fake_load_model(spec: ModelSpec):
        return SimpleNamespace(
            model=SimpleNamespace(),
            tokenizer=SimpleNamespace(),
            device=SimpleNamespace(type="cpu"),
            model_id=spec.hf_id,
            revision=spec.revision,
            tokenizer_revision=spec.tokenizer_revision,
            dtype="bfloat16",
        )

    def fake_load_sae_stack(*, spec: SaeSiteSpec, device: str, dtype: str):
        del device, dtype
        calls.append(spec.layers)
        return {layer: SimpleNamespace(layer=layer) for layer in spec.layers}

    default_cfg = first_study_stack_config()
    assert default_cfg.sae.layers == (9, 17, 22, 29)
    assert default_cfg.sae.width == "width_65k"
    assert default_cfg.sae.l0 == "l0_medium"

    single = ExperimentStackConfig(
        model=default_cfg.model,
        sae=SaeSiteSpec(
            release=default_cfg.sae.release,
            site=default_cfg.sae.site,
            width=default_cfg.sae.width,
            l0=default_cfg.sae.l0,
            layers=(17,),
        ),
        hooks=default_cfg.hooks,
    )

    stack_all = load_stack(
        default_cfg,
        _load_model=fake_load_model,
        _load_sae_stack=fake_load_sae_stack,
    )
    stack_one = load_stack(
        single,
        _load_model=fake_load_model,
        _load_sae_stack=fake_load_sae_stack,
    )
    assert calls == [(9, 17, 22, 29), (17,)]
    assert set(stack_all.saes.keys()) == {9, 17, 22, 29}
    assert set(stack_one.saes.keys()) == {17}
