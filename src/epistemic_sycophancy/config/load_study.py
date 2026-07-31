"""Load StudyConfig from YAML (Phase L CFGFILE-002 / DEC-057)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from epistemic_sycophancy.config.schema import (
    ExperimentConfig,
    InvalidExperimentConfig,
)
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudySmokeConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _require_mapping(payload: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise InvalidExperimentConfig(
            f"{label} must be a mapping; got {type(payload).__name__}"
        )
    return payload


def _require_key(mapping: dict[str, Any], key: str, *, label: str) -> Any:
    if key not in mapping:
        raise InvalidExperimentConfig(f"{label} missing required key {key!r}")
    return mapping[key]


def _parse_stack(raw: dict[str, Any]) -> ExperimentStackConfig:
    model_raw = _require_mapping(_require_key(raw, "model", label="stack"), label="stack.model")
    sae_raw = _require_mapping(_require_key(raw, "sae", label="stack"), label="stack.sae")
    hooks_raw = _require_mapping(
        _require_key(raw, "hooks", label="stack"), label="stack.hooks"
    )
    layers = _require_key(sae_raw, "layers", label="stack.sae")
    return ExperimentStackConfig(
        model=ModelSpec(
            hf_id=str(_require_key(model_raw, "hf_id", label="stack.model")),
            revision=str(_require_key(model_raw, "revision", label="stack.model")),
            tokenizer_revision=str(
                _require_key(model_raw, "tokenizer_revision", label="stack.model")
            ),
            dtype=str(_require_key(model_raw, "dtype", label="stack.model")),
            device_policy=str(
                _require_key(model_raw, "device_policy", label="stack.model")
            ),
        ),
        sae=SaeSiteSpec(
            release=str(_require_key(sae_raw, "release", label="stack.sae")),
            site=str(_require_key(sae_raw, "site", label="stack.sae")),
            width=str(_require_key(sae_raw, "width", label="stack.sae")),
            l0=str(_require_key(sae_raw, "l0", label="stack.sae")),
            layers=tuple(int(layer) for layer in layers),
        ),
        hooks=HookSpec(
            token_scope=str(
                _require_key(hooks_raw, "token_scope", label="stack.hooks")
            ),
            resolver_id=str(
                _require_key(hooks_raw, "resolver_id", label="stack.hooks")
            ),
            k=hooks_raw.get("k"),
        ),
    )


def _parse_feature_ids(raw_ids: Any) -> tuple[object, ...]:
    if raw_ids is None:
        raise InvalidExperimentConfig("experiment.feature_ids must be explicit")
    out: list[object] = []
    for item in raw_ids:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((int(item[0]), int(item[1])))
        else:
            out.append(int(item) if isinstance(item, (int, float)) and not isinstance(item, bool) else item)
    return tuple(out)


def _parse_experiment(raw: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        tau=float(_require_key(raw, "tau", label="experiment")),
        lambda_n=float(_require_key(raw, "lambda_n", label="experiment")),
        lambda_c=float(_require_key(raw, "lambda_c", label="experiment")),
        lambda_beta=float(_require_key(raw, "lambda_beta", label="experiment")),
        delta_n=float(_require_key(raw, "delta_n", label="experiment")),
        delta_c=float(_require_key(raw, "delta_c", label="experiment")),
        w_r=float(_require_key(raw, "w_r", label="experiment")),
        w_u=float(_require_key(raw, "w_u", label="experiment")),
        beta_lower=float(_require_key(raw, "beta_lower", label="experiment")),
        beta_upper=float(_require_key(raw, "beta_upper", label="experiment")),
        feature_ids=_parse_feature_ids(
            _require_key(raw, "feature_ids", label="experiment")
        ),
        feature_scales=tuple(
            float(s) for s in _require_key(raw, "feature_scales", label="experiment")
        ),
        coefficient_length=int(
            _require_key(raw, "coefficient_length", label="experiment")
        ),
        tie_policy=_require_key(raw, "tie_policy", label="experiment"),
        tie_band_epsilon=_require_key(raw, "tie_band_epsilon", label="experiment"),
        mc1_tie_policy=_require_key(raw, "mc1_tie_policy", label="experiment"),
        invalid_row_policy=_require_key(raw, "invalid_row_policy", label="experiment"),
        multi_token_candidate_scoring=_require_key(
            raw, "multi_token_candidate_scoring", label="experiment"
        ),
        ro_manifest_selection=_require_key(
            raw, "ro_manifest_selection", label="experiment"
        ),
        continuation_A=_require_key(raw, "continuation_A", label="experiment"),
        continuation_B=_require_key(raw, "continuation_B", label="experiment"),
        continuation_include_eos=_require_key(
            raw, "continuation_include_eos", label="experiment"
        ),
        attribution_scope=_require_key(raw, "attribution_scope", label="experiment"),
        pool_eligibility_override=_require_key(
            raw, "pool_eligibility_override", label="experiment"
        ),
        pool_quota_per_list=_require_key(
            raw, "pool_quota_per_list", label="experiment"
        ),
    )


def _parse_smoke(raw: dict[str, Any]) -> StudySmokeConfig:
    kwargs: dict[str, Any] = {}
    if "question_ids" in raw and raw["question_ids"] is not None:
        kwargs["question_ids"] = tuple(str(q) for q in raw["question_ids"])
    if "n_questions" in raw and raw["n_questions"] is not None:
        kwargs["n_questions"] = int(raw["n_questions"])
    if "split" in raw and raw["split"] is not None:
        kwargs["split"] = str(raw["split"])
    if "seed" in raw and raw["seed"] is not None:
        kwargs["seed"] = int(raw["seed"])
    return StudySmokeConfig(**kwargs)


def _parse_optimizer(raw: dict[str, Any]) -> StudyOptimizerConfig:
    kind = str(_require_key(raw, "kind", label="run.optimizer"))
    kwargs: dict[str, Any] = {
        "kind": kind,
        "max_steps": int(_require_key(raw, "max_steps", label="run.optimizer")),
    }
    for key in (
        "adam_lr",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "adam_microbatch_questions",
        "cma_seed",
    ):
        if key in raw and raw[key] is not None:
            value = raw[key]
            if key in {"adam_microbatch_questions", "cma_seed"}:
                kwargs[key] = int(value)
            else:
                kwargs[key] = float(value)
    return StudyOptimizerConfig(**kwargs)


def _parse_optimize(raw: dict[str, Any]) -> StudyOptimizeConfig:
    kwargs: dict[str, Any] = {
        "budget_match_on": str(
            _require_key(raw, "budget_match_on", label="run.optimize")
        ),
    }
    if "max_steps" in raw and raw["max_steps"] is not None:
        kwargs["max_steps"] = int(raw["max_steps"])
    if "n_trials" in raw and raw["n_trials"] is not None:
        kwargs["n_trials"] = int(raw["n_trials"])
    if "population_size" in raw and raw["population_size"] is not None:
        kwargs["population_size"] = int(raw["population_size"])
    if "n_questions" in raw and raw["n_questions"] is not None:
        kwargs["n_questions"] = int(raw["n_questions"])
    if "question_ids" in raw and raw["question_ids"] is not None:
        kwargs["question_ids"] = tuple(str(q) for q in raw["question_ids"])
    return StudyOptimizeConfig(**kwargs)


def _parse_run(raw: dict[str, Any]) -> StudyRunConfig:
    smoke_raw = _require_mapping(
        _require_key(raw, "smoke", label="run"), label="run.smoke"
    )
    opt_raw = _require_mapping(
        _require_key(raw, "optimizer", label="run"), label="run.optimizer"
    )
    optimize_raw = _require_mapping(
        _require_key(raw, "optimize", label="run"), label="run.optimize"
    )
    if "order_regimes" in raw:
        raise InvalidExperimentConfig(
            "run.order_regimes is no longer accepted; use singular "
            "run.order_regime in {'CF', 'IF', 'RO'} (DEC-087 / ORDER-EXP-001)"
        )
    return StudyRunConfig(
        artifact_dir=str(_require_key(raw, "artifact_dir", label="run")),
        order_regime=str(_require_key(raw, "order_regime", label="run")),
        feature_chunk_size=int(
            _require_key(raw, "feature_chunk_size", label="run")
        ),
        prompt_batch_size=int(_require_key(raw, "prompt_batch_size", label="run")),
        smoke=_parse_smoke(smoke_raw),
        optimizer=_parse_optimizer(opt_raw),
        optimize=_parse_optimize(optimize_raw),
    )


def study_config_from_mapping(payload: dict[str, Any]) -> StudyConfig:
    """Validate a raw mapping into StudyConfig."""
    stack_raw = _require_mapping(
        _require_key(payload, "stack", label="study"), label="stack"
    )
    experiment_raw = _require_mapping(
        _require_key(payload, "experiment", label="study"), label="experiment"
    )
    run_raw = _require_mapping(_require_key(payload, "run", label="study"), label="run")
    return StudyConfig(
        stack=_parse_stack(stack_raw),
        experiment=_parse_experiment(experiment_raw),
        run=_parse_run(run_raw),
    )


def load_study_config(path: str | Path) -> StudyConfig:
    """Parse YAML at ``path`` into a validated StudyConfig (DEC-057)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidExperimentConfig(f"cannot read study config {path}: {exc}") from exc
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InvalidExperimentConfig(f"invalid YAML in {path}: {exc}") from exc
    if payload is None:
        raise InvalidExperimentConfig(f"study config {path} is empty")
    mapping = _require_mapping(payload, label="study")
    return study_config_from_mapping(mapping)


def _fingerprint_payload(study: StudyConfig) -> dict[str, Any]:
    """Canonical JSON-serializable payload for fingerprinting."""
    exp = study.experiment
    feature_ids: list[Any] = []
    for fid in exp.feature_ids:
        if isinstance(fid, tuple):
            feature_ids.append([int(fid[0]), int(fid[1])])
        else:
            feature_ids.append(fid)
    smoke = study.run.smoke
    opt = study.run.optimizer
    optimize = study.run.optimize
    return {
        "stack": {
            "model": {
                "hf_id": study.stack.model.hf_id,
                "revision": study.stack.model.revision,
                "tokenizer_revision": study.stack.model.tokenizer_revision,
                "dtype": study.stack.model.dtype,
                "device_policy": study.stack.model.device_policy,
            },
            "sae": {
                "release": study.stack.sae.release,
                "site": study.stack.sae.site,
                "width": study.stack.sae.width,
                "l0": study.stack.sae.l0,
                "layers": list(study.stack.sae.layers),
            },
            "hooks": {
                "token_scope": study.stack.hooks.token_scope,
                "resolver_id": study.stack.hooks.resolver_id,
                "k": study.stack.hooks.k,
            },
        },
        "experiment": {
            "tau": exp.tau,
            "lambda_n": exp.lambda_n,
            "lambda_c": exp.lambda_c,
            "lambda_beta": exp.lambda_beta,
            "delta_n": exp.delta_n,
            "delta_c": exp.delta_c,
            "w_r": exp.w_r,
            "w_u": exp.w_u,
            "beta_lower": exp.beta_lower,
            "beta_upper": exp.beta_upper,
            "feature_ids": feature_ids,
            "feature_scales": list(exp.feature_scales),
            "coefficient_length": exp.coefficient_length,
            "tie_policy": exp.tie_policy,
            "tie_band_epsilon": exp.tie_band_epsilon,
            "mc1_tie_policy": exp.mc1_tie_policy,
            "invalid_row_policy": exp.invalid_row_policy,
            "multi_token_candidate_scoring": exp.multi_token_candidate_scoring,
            "ro_manifest_selection": exp.ro_manifest_selection,
            "continuation_A": exp.continuation_A,
            "continuation_B": exp.continuation_B,
            "continuation_include_eos": exp.continuation_include_eos,
            "attribution_scope": exp.attribution_scope,
            "pool_eligibility_override": exp.pool_eligibility_override,
            "pool_quota_per_list": exp.pool_quota_per_list,
        },
        "run": {
            "artifact_dir": study.run.artifact_dir,
            "order_regime": study.run.order_regime,
            "feature_chunk_size": study.run.feature_chunk_size,
            "prompt_batch_size": study.run.prompt_batch_size,
            "smoke": {
                "n_questions": smoke.n_questions,
                "split": smoke.split,
                "seed": smoke.seed,
                "question_ids": list(smoke.question_ids)
                if smoke.question_ids is not None
                else None,
            },
            "optimizer": {
                "kind": opt.kind,
                "max_steps": opt.max_steps,
                "adam_lr": opt.adam_lr,
                "adam_beta1": opt.adam_beta1,
                "adam_beta2": opt.adam_beta2,
                "adam_eps": opt.adam_eps,
                "adam_microbatch_questions": opt.adam_microbatch_questions,
                "cma_seed": opt.cma_seed,
            },
            "optimize": {
                "budget_match_on": optimize.budget_match_on,
                "max_steps": optimize.max_steps,
                "n_trials": optimize.n_trials,
                "population_size": optimize.population_size,
                "n_questions": optimize.n_questions,
                "question_ids": list(optimize.question_ids)
                if optimize.question_ids is not None
                else None,
            },
        },
    }


def study_config_fingerprint(study: StudyConfig) -> str:
    """SHA-256 hex of the canonical validated StudyConfig payload."""
    payload = _fingerprint_payload(study)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
