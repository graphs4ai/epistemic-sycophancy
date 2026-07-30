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
| GRAD-007 | toy/integration: `build_grad_fn` + projected Adam moves β from 0 |
| GRAD-008 | real `layer17_n2` optimize: ≥1 `|β_i|>0` in trials (**required**; DEC-084 loud zero-grad is failure, not green — GRAD-011). Unblocked by multi-condition FS (FSC-009). |
| FSC-009 | real `layer17_n2`: FS pool features active on ≥1 FS IB and ≥1 FS CB; then optimize moves β |

### GRAD-FIX (DEC-084) — re-run optimize after fix

Production `build_grad_fn` must supply projected ∂M/∂β (not all-zero jac). After
GRAD-FIX, **re-run** `run-optimize` (and prefer deleting stale
`optimize/trials.jsonl`).

**Failure criterion:** flat all-zero β for every step with constant `l_total` is
**not** success evidence (pre-fix `artifacts/smokes/layer17_n2/optimize/trials.jsonl`
is obsolete). Accept only: β moves within bounds. A loud DEC-084 identically-zero /
non-finite grad error is a **failure diagnosis**, not a green gate (GRAD-011).

### FS-COMPONENTS (DEC-085) — re-run feature_selection before optimize

Production feature selection ranks **four** components from **N, IB, and CB** on the
FS split (resistance=IB∩Q+, recovery=CB∩Q−, neutral/correct surrogates). Pool
nomination remains DEC-019 (resistance/recovery only). Artifacts use
`schema_version: 2` with nominator provenance.

**Stale pools:** neutral-only / v1 `feature_selection/common_pool.json` files are
**rejected** on load. Always **re-run** `run-fs` before `run-optimize` after this
fix (or after any FS change). Do not reuse pre-FS-COMPONENTS pools.

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
