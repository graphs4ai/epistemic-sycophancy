"""FSC-009: layer17_n2 multi-condition FS → IB/CB-active pool → nonzero optimize grad."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from epistemic_sycophancy.config.load_study import load_study_config
from epistemic_sycophancy.runner.cli import dispatch_stage
from epistemic_sycophancy.runner.identity import clear_stack_cache

CFG = Path("configs/smokes/layer17_n2.yaml")


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FSC-009 real_model gate")


def _pool_activity_on_fs_conditions(
    *,
    study,
    stack,
    pool_feature_ids: list[tuple[int, int]],
    question_ids: list[str],
    order_regime: str = "CF",
) -> dict[str, dict[str, int]]:
    """Count selected-feature activity 1[z>0] at last prompt token on FS N/IB/CB."""
    from epistemic_sycophancy.config.study import StudySmokeConfig
    from epistemic_sycophancy.runner.adapters.jacobian import (
        render_fs_multi_condition_rows,
    )
    from epistemic_sycophancy.runner.adapters.resolve import resolve_corpus_context
    from epistemic_sycophancy.sae.jumprelu_delta import jumprelu
    from epistemic_sycophancy.stack.resolver import resolve_resid_post_module

    corpus, split_ids, _ = resolve_corpus_context(study)
    by_condition = render_fs_multi_condition_rows(
        corpus_rows=corpus,
        smoke=StudySmokeConfig(question_ids=tuple(question_ids)),
        split_question_ids=split_ids,
        order_regime=order_regime,
    )
    layer = int(study.stack.sae.layers[0])
    sae = stack.saes[layer].sae
    enc_w = sae.W_enc.detach()
    enc_b = sae.b_enc.detach()
    threshold = sae.threshold.detach()
    selected = [fid for lyr, fid in pool_feature_ids if lyr == layer]
    out: dict[str, dict[str, int]] = {}
    for belief, rows in by_condition.items():
        if not rows:
            out[belief] = {"n_slots": 0, "n_active": 0, "n_prompts": 0}
            continue
        texts = [r.text for r in rows]
        encoded = stack.tokenizer(texts, return_tensors="pt", padding=True)
        encoded = {k: v.to(stack.device) for k, v in encoded.items()}
        captured: dict[str, object] = {}

        def hook(_m, _i, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured["residual"] = tensor.detach()
            return output

        module = resolve_resid_post_module(
            stack.model,
            layer=layer,
            resolver_id=stack.config.hooks.resolver_id,
        )
        handle = module.register_forward_hook(hook)
        try:
            with __import__("torch").no_grad():
                stack.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                )
        finally:
            handle.remove()
        residual = captured["residual"]
        assert residual is not None
        attn = encoded.get("attention_mask")
        n_active = 0
        n_slots = 0
        for i in range(residual.shape[0]):
            plen = int(attn[i].sum().item()) if attn is not None else residual.shape[1]
            last = residual[i, plen - 1, :].float()
            pre = last @ enc_w.float() + enc_b.float()
            z = jumprelu(pre, threshold.float())
            for fid in selected:
                n_slots += 1
                if float(z[int(fid)].item()) > 0.0:
                    n_active += 1
        out[belief] = {
            "n_slots": n_slots,
            "n_active": n_active,
            "n_prompts": len(rows),
        }
    return out


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__layer17_n2__fs_pool_active_on_ib_cb_and_optimize_moves_beta(
    tmp_path: Path,
) -> None:
    """FSC-009: multi-condition FS pool active on IB/CB; optimize moves β."""
    _require_cuda()
    clear_stack_cache()
    from dataclasses import replace

    from epistemic_sycophancy.config.study import StudyOptimizeConfig
    from epistemic_sycophancy.runner.identity import resolve_stack

    art = tmp_path / "art"
    study = load_study_config(CFG)
    study = replace(study, run=replace(study.run, artifact_dir=str(art)))

    assert dispatch_stage("identity", study=study, freeze_status="unsealed").ok
    assert dispatch_stage(
        "baseline_partitions", study=study, freeze_status="unsealed", score_fn=None
    ).ok
    fs_result = dispatch_stage(
        "feature_selection",
        study=replace(study, run=replace(study.run, order_regime="CF")),
        freeze_status="unsealed",
        jacobian_fn=None,
        scale_fn=None,
    )
    assert fs_result.ok
    pool_path = art / "feature_selection" / "common_pool.json"
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    assert payload.get("schema_version") == 2
    pool_ids = [(int(a), int(b)) for a, b in payload["feature_ids"]]
    assert pool_ids, "expected nonempty pool"

    stack = resolve_stack(study)
    activity = _pool_activity_on_fs_conditions(
        study=study,
        stack=stack,
        pool_feature_ids=pool_ids,
        question_ids=list(payload["question_ids"]),
        order_regime="CF",
    )
    diag_path = art / "feature_selection" / "pool_activity_diagnosis.json"
    diag_path.write_text(json.dumps(activity, indent=2, sort_keys=True) + "\n")
    if activity["IB"]["n_active"] < 1 or activity["CB"]["n_active"] < 1:
        pytest.fail(
            "FSC-009 blocked: pool features inactive on FS IB/CB. "
            f"activity={activity} (wrote {diag_path})"
        )

    opt_study = replace(
        study,
        run=replace(
            study.run,
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=5,
                n_questions=4,
            ),
        ),
    )
    try:
        result = dispatch_stage(
            "optimize",
            study=opt_study,
            freeze_status="unsealed",
            objective_fn=None,
            grad_fn=None,
            identity_passed=None,
        )
    except ValueError as exc:
        msg = str(exc)
        if "identically zero" in msg or "∂L/∂β" in msg:
            pytest.fail(
                "FSC-009: DEC-084 zero grad after multi-condition FS. "
                f"activity={activity}. Original: {msg}"
            )
        raise

    assert result.ok
    trials_path = art / "optimize" / "trials.jsonl"
    rows = [
        json.loads(line)
        for line in trials_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    moved = any(
        any(abs(float(b)) > 1e-8 for b in row["beta"])
        and all(math.isfinite(float(b)) for b in row["beta"])
        for row in rows
    )
    assert moved, f"FSC-009: no |β_i|>0 in trials; activity={activity}"
