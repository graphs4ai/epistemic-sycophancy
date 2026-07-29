"""Identity stage: β=0 residual identity on smoke prompts (ORCH-001 / DEC-064)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.load_study import study_config_fingerprint
from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.stack.config import ExperimentStackConfig
from epistemic_sycophancy.stack.intervention_stack import InterventionStack, load_stack

# Process-local stack cache (DEC-064/080): one InterventionStack per stack config.
_STACK_CACHE: dict[str, Any] = {}


def clear_stack_cache() -> None:
    """Clear the process-local stack cache (tests / fingerprint change)."""
    _STACK_CACHE.clear()


def stack_config_fingerprint(stack: ExperimentStackConfig) -> str:
    """Stable hash of model+SAE+hooks only (DEC-080); ignores experiment/run."""
    payload = {
        "model": {
            "hf_id": stack.model.hf_id,
            "revision": stack.model.revision,
            "tokenizer_revision": stack.model.tokenizer_revision,
            "dtype": stack.model.dtype,
            "device_policy": stack.model.device_policy,
        },
        "sae": {
            "release": stack.sae.release,
            "site": stack.sae.site,
            "width": stack.sae.width,
            "l0": stack.sae.l0,
            "layers": list(stack.sae.layers),
        },
        "hooks": {
            "token_scope": stack.hooks.token_scope,
            "resolver_id": stack.hooks.resolver_id,
            "k": stack.hooks.k,
        },
    }
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(material.encode("utf-8")).hexdigest()


def resolve_stack(
    study: StudyConfig,
    *,
    stack_loader: Callable[[StudyConfig], Any] | None = None,
) -> Any:
    """Lazy-load and cache InterventionStack for this process (DEC-064/065/080)."""
    cache_key = stack_config_fingerprint(study.stack)
    if stack_loader is not None:
        # Explicit injection always wins (unit tests; DEC-065).
        stack = stack_loader(study)
        _STACK_CACHE[cache_key] = stack
        return stack
    if cache_key in _STACK_CACHE:
        return _STACK_CACHE[cache_key]
    stack = _default_stack_loader(study)
    _STACK_CACHE[cache_key] = stack
    return stack


def _default_stack_loader(study: StudyConfig) -> InterventionStack:
    return load_stack(study.stack)


def run_identity_stage(
    *,
    study: StudyConfig,
    stack: Any,
    smoke_texts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare unhooked vs β=0 hooked residuals on smoke prompts.

    Returns metrics including ``identity_passed`` and ``max_abs_diff``.
    """
    layers = tuple(int(layer) for layer in study.stack.sae.layers)
    texts = list(smoke_texts) if smoke_texts is not None else _default_smoke_texts()
    selected_keys = tuple((layer, 0) for layer in layers)
    scales = tuple(1.0 for _ in selected_keys)
    beta = tuple(0.0 for _ in selected_keys)
    prompt_lengths = _prompt_lengths(stack, texts)

    unhooked = stack.capture_layer_residuals(texts=texts, layers=layers)
    with stack.install_hooks(
        selected_keys=selected_keys,
        scales=scales,
        beta=beta,
        prompt_lengths=prompt_lengths,
    ):
        hooked = stack.capture_layer_residuals(texts=texts, layers=layers)

    max_abs_diff = 0.0
    for layer in layers:
        diff = (unhooked[layer].float() - hooked[layer].float()).abs().max().item()
        max_abs_diff = max(max_abs_diff, float(diff))

    identity_passed = max_abs_diff == 0.0 or max_abs_diff < 1e-6
    artifact_dir = Path(study.run.artifact_dir) / "identity"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "identity_result.json"
    payload = {
        "identity_passed": identity_passed,
        "max_abs_diff": max_abs_diff,
        "layers": list(layers),
        "n_prompts": len(texts),
        "study_yaml_fingerprint": study_config_fingerprint(study),
    }
    artifact_path.write_text(
        __import__("json").dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "identity_passed": identity_passed,
        "max_abs_diff": max_abs_diff,
        "artifacts": {"identity_result": str(artifact_path)},
        "payload": payload,
    }


def _default_smoke_texts() -> list[str]:
    return ["Smoke identity prompt A.", "Smoke identity prompt B."]


def _prompt_lengths(stack: Any, texts: Sequence[str]) -> tuple[int, ...]:
    tokenizer = getattr(stack, "tokenizer", None)
    if tokenizer is None:
        return tuple(max(1, len(text.split())) for text in texts)
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
    )
    if "attention_mask" in encoded:
        lengths = encoded["attention_mask"].sum(dim=-1).tolist()
        return tuple(int(x) for x in lengths)
    return tuple(int(encoded["input_ids"].shape[-1]) for _ in texts)
