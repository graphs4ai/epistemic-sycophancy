"""RUN-003/004: GemmaScope2 resid_post SAE load via sae-lens."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_sycophancy.sae.load import load_sae
from epistemic_sycophancy.sae.spec import SaeSiteSpec

_PIN = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "real_model"
    / "gemmascope2_4b_it_resid_post_pin.json"
)


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_sae__load_single_layer__d_model_matches_and_decoder_width_recorded() -> None:
    """RUN-003: load one layer SAE; d_in matches model; decoder width recorded."""
    pin = json.loads(_PIN.read_text())
    spec = SaeSiteSpec(
        release=pin["release"],
        site=pin["site"],
        width=pin["width"],
        l0=pin["l0"],
        layers=(9,),
    )
    handle = load_sae(spec=spec, layer=9, device="cuda", dtype="bfloat16")
    assert handle.layer == 9
    assert handle.sae_id == pin["layer_9_sae_id"]
    assert handle.release == pin["release"]
    assert handle.d_in == pin["expected_d_in"] == pin["model_hidden_size"]
    assert handle.d_sae == pin["expected_d_sae"]
    assert handle.decoder_width == pin["expected_d_in"]
    assert tuple(handle.decoder_weight.shape) == (
        pin["expected_d_sae"],
        pin["expected_d_in"],
    )
