# Phase M ship gate — start first real experiment

Phase M is the **final implementation phase**. Researchers edit YAML only.

## Recommended ASAP path (DEC-067)

1. Start with single-layer smoke: `configs/smokes/layer17_n2.yaml`
   - `run.smoke`: N=2 FS subset
   - `run.optimizer.max_steps: 1` (opt_smoke only)
   - `run.optimize`: `max_steps: 20`, `n_questions: 4` (tiny non-smoke; DEC-068)
2. After that path is green, scale to `configs/first_study_gemma3_4b_resid_post_65k_medium.yaml`
   (4 layers; full optimization split when `run.optimize.n_questions` omitted).

## Commands

```bash
CFG=configs/smokes/layer17_n2.yaml

pixi run --environment test-cuda run-identity -- --config "$CFG"
pixi run --environment test-cuda run-baseline -- --config "$CFG"
pixi run --environment test-cuda run-fs -- --config "$CFG"
pixi run --environment test-cuda run-opt-smoke -- --config "$CFG"
pixi run --environment test-cuda run-optimize -- --config "$CFG"
pixi run --environment test-cuda run-freeze -- --config "$CFG"
pixi run --environment test-cuda run-study -- --config "$CFG"
```

Holdout remains sealed until `run-holdout` after freeze (DEC-071). Do not open holdout for checkpoint selection.

## Artifacts (DEC-070)

Under `run.artifact_dir`:

- `identity/`, `baseline/`, `feature_selection/`, `opt_smoke/`
- `optimize/trials.jsonl`, `optimize/checkpoints/`, `optimize/best_checkpoint.json`
- `freeze/frozen_experiment_config.json`
- `full_study/behavioral.json`, `full_study/cross_order_matrix.json`
- `holdout/` only after unlock

## Unit vs CUDA

- Unit tests inject `stack_loader` / `score_fn` / `jacobian_fn` / `objective_fn` (DEC-065).
- Real Gemma + GemmaScope2 requires `pixi run --environment test-cuda …`.
