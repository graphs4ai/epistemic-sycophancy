# Phase M ship gate — start first real experiment

Phase M is the **final implementation phase**. Researchers edit YAML only.

Phase **M.1** wires production adapters so bare CLI `--config` needs **no**
`score_fn` / `jacobian_fn` / `objective_fn` / `eval_payload` injector kwargs
(DEC-073…081). Unit tests may still inject fakes (DEC-065).

**DEC-087:** each study is a **single** answer-order experiment
(`run.order_regime: CF|IF|RO`). Run CF, IF, and RO as three separate sealed
studies, then assemble the 3×3 matrix (DEC-088).

**DEC-093:** no `run.smoke` / `opt_smoke` stage. Coverage defaults to the **full
available split** when omitted. Limited runs opt in via `run.fs_coverage` and/or
`run.optimize.n_questions` (see `configs/dev/`).

## Recommended limited path (DEC-067 / DEC-079 / DEC-087 / DEC-093)

1. Start with single-layer limited YAML for **one** order, e.g.
   `configs/dev/layer17_n32_CF.yaml` (aliases: `layer17_n32.yaml` = CF)
   - `run.order_regime: CF` (or IF / RO via sibling YAMLs)
   - `run.fs_coverage`: `n_questions: 32`, `seed: 0` (explicit subset)
   - `run.optimize`: `max_steps: 20`, optional `n_questions` for a tiny optimize set
2. Repeat with `layer17_n32_IF.yaml` and `layer17_n32_RO.yaml` (distinct
   `artifact_dir`s).
3. Assemble cross-order matrix from the three sealed roots
   (`configs/dev/layer17_n32_cross_order.yaml` + `run_cross_order_assemble`).
4. After that path is green, scale to
   `configs/first_study_gemma3_4b_resid_post_65k_medium.yaml` (CF default;
   **no** `fs_coverage` → full feature_selection split)
   and the IF/RO siblings
   `configs/first_study_gemma3_4b_resid_post_65k_medium_{IF,RO}.yaml`.

## Commands (YAML-only; `test-cuda`)

No injector kwargs. Stack loads once per process (DEC-064/080). Pool after FS is
applied in-memory from `feature_selection/common_pool.json` (DEC-073).

```bash
# One order experiment (repeat with _IF / _RO)
CFG=configs/dev/layer17_n32_CF.yaml

pixi run --environment test-cuda run-identity -- --config "$CFG"
pixi run --environment test-cuda run-baseline -- --config "$CFG"
pixi run --environment test-cuda run-fs -- --config "$CFG"
pixi run --environment test-cuda run-optimize -- --config "$CFG"
pixi run --environment test-cuda run-freeze -- --config "$CFG"
pixi run --environment test-cuda run-study -- --config "$CFG"
```

After CF, IF, and RO are sealed, assemble the 3×3 (DEC-088) via
`run_cross_order_assemble` with `configs/dev/layer17_n32_cross_order.yaml`
sources (Python/API; not part of the per-study DEC-072 sequence).

Holdout remains sealed until `run-holdout` after freeze (DEC-071). Do not open
holdout for checkpoint selection.

## Real-model gates (supersede hollow ORCH-017/018)

Verified on `pixi run --environment test-cuda`:

| Spec | Behavior |
|---|---|
| ORCH-034 | identity via default stack |
| ORCH-035 | baseline partitions without `score_fn` (single `order_regime`) |
| ORCH-036 | feature_selection writes `common_pool.json` |
| ORCH-038 | optimize writes `best_checkpoint.json` via `run.optimize` budget |
| GRAD-007 | toy/integration: `build_grad_fn` + projected Adam moves β from 0 |
| GRAD-008 | real `layer17_n32` optimize: ≥1 `|β_i|>0` in trials (**required**; DEC-084 loud zero-grad is failure, not green — GRAD-011). Unblocked by multi-condition FS (FSC-009). |
| FSC-009 | real `layer17_n32`: FS pool features active on ≥1 FS IB and ≥1 FS CB; then optimize moves β |

### GRAD-FIX (DEC-084) — re-run optimize after fix

Production `build_grad_fn` must supply projected ∂M/∂β (not all-zero jac). After
GRAD-FIX, **re-run** `run-optimize` (and prefer deleting stale
`optimize/trials.jsonl`).

**Failure criterion:** flat all-zero β for every step with constant `l_total` is
**not** success evidence. Accept only: β moves within bounds. A loud DEC-084
identically-zero / non-finite grad error is a **failure diagnosis**, not a green
gate (GRAD-011).

### FS-COMPONENTS (DEC-085) — re-run feature_selection before optimize

Production feature selection ranks **four** components from **N, IB, and CB** on the
FS split for the study `order_regime`. Pool nomination remains DEC-019
(resistance/recovery only). Artifacts use `schema_version: 2` with nominator
provenance.

**Stale pools:** neutral-only / v1 `feature_selection/common_pool.json` files are
**rejected** on load. Always **re-run** `run-fs` before `run-optimize` after this
fix (or after any FS change).

Unit e2e without injectors (fake `stack_loader` only): ORCH-033.

## Artifacts (DEC-070 / DEC-087 / DEC-088 / DEC-093)

Per-study under `run.artifact_dir` (e.g. `artifacts/dev/layer17_n32_CF/`):

- `identity/`, `baseline/partition_{order}.json`, `feature_selection/`
- `optimize/trials.jsonl`, `optimize/checkpoints/`, `optimize/best_checkpoint.json`, `optimize/best_checkpoint_by_{metric}.json` (DEC-100)
- `freeze/frozen_experiment_config.json` (includes `order_regime`)
- `full_study/behavioral.json` (intervened best-β; single-order; **no** in-study 3×3)
- `full_study/behavioral_non_intervened.json` (β=0 comparison; same metric schema; DEC-098)
- `full_study/behavioral_best_by_{metric}.json` (per opt-split criterion; DEC-100)
- `holdout/` only after unlock

Cross-study assemble root (e.g. `artifacts/dev/layer17_n32_cross_order/`):

- `cross_order/sources.json`
- `cross_order/cross_order_matrix.json`

## Unit vs CUDA

- Unit tests may inject `stack_loader` and stage callables (DEC-065).
- Production path builds adapters from `StudyConfig` + `InterventionStack`
  when injectors are `None`.
- Real Gemma + GemmaScope2 requires `pixi run --environment test-cuda …`.

## Operational logging (DEC-089)

Stderr logger `epistemic_sycophancy.pipeline`. CLI `--log-level` (default INFO).

- Every stage: start + end with `elapsed_s` and artifact paths.
- Optimize: per-step/trial `progress=optimize_step` with `l_total`.
- Feature selection: per-component start, skip of empty components, pool size.
- Freeze seal / holdout unseal: WARNING `audit=…` (leakage-sensitive).
- Distinct from DEC-026/035 `trials.jsonl` / `TrialRecord` artifact schemas.
