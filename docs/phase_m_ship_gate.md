# Phase M ship gate — start first real experiment

Phase M is the **final implementation phase**. Researchers edit YAML only.

Phase **M.1** wires production adapters so bare CLI `--config` needs **no**
`score_fn` / `jacobian_fn` / `objective_fn` / `eval_payload` injector kwargs
(DEC-073…081). Unit tests may still inject fakes (DEC-065).

## Recommended ASAP path (DEC-067 / DEC-079)

1. Start with single-layer smoke: `configs/smokes/layer17_n2.yaml`
   - `run.smoke`: `n_questions: 32`, `split: feature_selection`, `seed: 0`
     (N=32 for non-degenerate baseline partitions on Gemma3-4B)
   - `run.optimizer.max_steps: 1` (opt_smoke only)
   - `run.optimize`: `max_steps: 20`, `n_questions: 4` (tiny non-smoke; DEC-068)
2. After that path is green, scale to `configs/first_study_gemma3_4b_resid_post_65k_medium.yaml`
   (4 layers; full optimization split when `run.optimize.n_questions` omitted).

## Commands (YAML-only; `test-cuda`)

No injector kwargs. Stack loads once per process (DEC-064/080). Pool after FS is
applied in-memory from `feature_selection/common_pool.json` (DEC-073).

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

Holdout remains sealed until `run-holdout` after freeze (DEC-071). Do not open
holdout for checkpoint selection.

## Real-model gates (supersede hollow ORCH-017/018)

Verified on `pixi run --environment test-cuda`:

| Spec | Behavior |
|---|---|
| ORCH-034 | identity via default stack |
| ORCH-035 | baseline partitions without `score_fn` |
| ORCH-036 | feature_selection writes `common_pool.json` |
| ORCH-037 | opt_smoke finite `l_total` via live belief-margin adapters |
| ORCH-038 | optimize writes `best_checkpoint.json` via `run.optimize` budget |

Unit e2e without injectors (fake `stack_loader` only): ORCH-033.

## Artifacts (DEC-070)

Under `run.artifact_dir`:

- `identity/`, `baseline/`, `feature_selection/`, `opt_smoke/`
- `optimize/trials.jsonl`, `optimize/checkpoints/`, `optimize/best_checkpoint.json`
- `freeze/frozen_experiment_config.json`
- `full_study/behavioral.json`, `full_study/cross_order_matrix.json`
- `holdout/` only after unlock

## Unit vs CUDA

- Unit tests may inject `stack_loader` and stage callables (DEC-065).
- Production ASAP path builds adapters from `StudyConfig` + `InterventionStack`
  when injectors are `None`.
- Real Gemma + GemmaScope2 requires `pixi run --environment test-cuda …`.
