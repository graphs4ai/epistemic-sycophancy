# epistemic-sycophancy

Library and experiment for **additive SAE interventions** against
**epistemic sycophancy**: models deferring to a user's incorrect beliefs even
when they would answer correctly under a neutral prompt.

The intervention is meant to reduce that deference while preserving
**correction selectivity** (still accept truthful corrections when the baseline
was wrong) and **baseline truthfulness** (do not degrade neutral or
correct-belief performance).

## Motivation

Belief-conditioned multiple-choice prompts expose a failure mode that ordinary
accuracy misses. Under an incorrect-belief context, the model may flip from a
truthful answer to the user's false claim. Under a correct-belief context, it
should keep or regain the truthful answer. The experimental question is whether
a sparse, interpretable residual-stream edit can improve resistance and recovery
without turning the model into a generic “always disagree” system.

Sparse autoencoders (SAEs) supply candidate features. This project selects a
small pool of those features from local loss Jacobians, then optimizes bounded
suppression coefficients $\beta$ under a composite objective. Evaluation is
behavioral and order-aware: frozen baseline partitions, question-macro metrics,
and a 3×3 cross-order matrix that separates semantic steering from answer-position
artifacts.

## Core concepts

**Semantic truthful margin.** For two-candidate prompts the margin is always

$$
M = s_{\text{truthful}} - s_{\text{incorrect}},
$$

independent of whether truth is labeled A or B. Swapping candidate positions and
their scores must preserve $M$. The two-candidate truthful probability is
$\sigma(M)$.

**Belief conditions and answer order.** Each original question yields neutral,
correct-belief, and incorrect-belief prompts. Answer order is one of CF
(correct-first), IF (incorrect-first), or RO (deterministic random order keyed
by seed and question id). Each study optimizes under a single order regime;
cross-order evaluation compares CF/IF/RO without refitting $\beta$.

**Frozen, order-specific baselines.** From unmodified *neutral* margins, each
order freezes $Q^+$ (baseline truthful) and $Q^-$ (baseline incorrect), plus
a configured near-tie policy. Those partitions are never recomputed after an
intervention and are reused for every downstream metric.

**Question-macro aggregation.** Score each concrete prompt, average within the
original question, then average across questions. Questions with more belief
paraphrases do not receive more weight.

**Additive SAE delta.** Selected features update as
$\alpha = s \odot \beta$, $z' = \mathrm{ReLU}(z+\alpha)$, then

$$
\Delta x = \mathrm{decode}(z') - \mathrm{decode}(z),\qquad
x' = x + \Delta x.
$$

At $\beta = 0$ the hook returns the original residual stream, not the SAE
reconstruction. Token scope (last token, last-$k$, or all prompt tokens) is
explicit configuration.

**Exact local coefficient Jacobian.** Feature ranking uses

$$
\frac{\partial \ell_u^{(p)}}{\partial \beta_{\ell,j}}\Big|_{\beta=0}
= s_{\ell,j}\,
\mathbf{1}\bigl[z_{\ell,j,t_p^{*}}^{(p)} \gt 0\bigr]
\,\langle g_{\ell,u}^{(p)}, d_{\ell,j}\rangle
$$

including decoder projection, prompt-specific ReLU activity, feature scale, and
question-macro weights. Suppression candidates are ranked by *signed* Jacobian:
positive $J$ predicts that suppressing the feature decreases loss. CF/IF/RO
rankings stay separate; a deterministic quota-union builds one common pool.

**Objective.** Optimization mixes resistance (incorrect belief on $Q^+$),
recovery (correct belief on $Q^-$), neutral and correct-belief preservation
hinges, and mean $|\beta|$. Feature selection uses non-flat preservation
surrogates because the optimizer hinges are locally flat at the null
intervention. Checkpoint selection uses validation metrics only; holdout stays
sealed until a freeze artifact unlocks it.

FTW, CBR, Selectivity, PRA, MC1, and MC2 are computed on
frozen partitions at the original-question level. Uncertainty uses
question-cluster bootstrap, not prompt-row resampling. Controls include
random-feature and shuffled-coefficient baselines.

## Pipeline overview

```text
data / prompts
        │
        ▼
   scoring (MC0 margins)
        │
        ├──► baseline partitions ──► behavioral metrics
        │
        ▼
   SAE intervention (additive residual δ)
        │
        ├──► feature Jacobians / rankings / common pool
        │
        ▼
   objective (resistance, recovery, hinges, β reg)
        │
        ├──► CMA-ES / projected Adam
        │
        ▼
   evaluation (cross-order, controls, bootstrap, gates)
```

Stage order for a sealed study:

```text
identity → baseline_partitions → feature_selection
→ optimize → freeze → full_study → holdout_eval
```

Holdout evaluation is last and requires an explicit unlock after freeze.

## Repository layout

```text
src/epistemic_sycophancy/   # library (importable package)
configs/                   # StudyConfig YAML (stack + experiment + run)
data/                      # TruthfulQA corpus build + frozen artifacts
tests/                     # unit / property / integration / real_model
docs/                      # TDD spec, decisions, progress log
AGENTS.md                  # contributor / agent implementation contract
```

| Package | Responsibility |
| --- | --- |
| `config` | Validated experiment config; freeze / holdout immutability |
| `data` | Manifest loading and integrity / leakage checks |
| `prompts` | Structured prompts, CF/IF/RO, continuation token contract |
| `scoring` | Margins, candidate scores, MC1/MC2, valid-answer mass |
| `metrics` | Frozen baseline partitions; FTW / CBR / Selectivity / PRA |
| `intervention` | Additive SAE δ and token-scope hooks |
| `feature_selection` | Jacobians, rankings, common pool, artifacts |
| `objective` | Logistic losses, question-macro means, full objective |
| `optimization` | CMA-ES, projected Adam, checkpoints, budgets |
| `statistics` | Question-cluster bootstrap |
| `evaluation` | Cross-order matrix, toy E2E, real-model check helpers |
| `controls` | Random features, shuffled coefficients |
| `reproducibility` | Holdout seal and phase gates |
| `logging` | Objective component and trial records |

## Reproducibility

### Environment

[Pixi](https://pixi.sh/) is the only environment and task manager
(`pyproject.toml` / `pixi.lock`).

```bash
pixi install
```

| Environment | Use |
| --- | --- |
| `default` | Runtime (CPU PyTorch) |
| `test` | Runtime + pytest, Hypothesis, transformers |
| `dev` | test + ruff, mypy |
| `cuda` / `test-cuda` | GPU PyTorch (linux-64-cuda) |

### Pixi tasks

Tasks are declared in `pyproject.toml`. Study stages take `--config <yaml>` after
`--`; use `--environment test-cuda` for real-model GPU runs.

**Study pipeline** (order: identity → baseline → feature selection → optimize →
freeze → full study → holdout):

| Task | What it does |
| --- | --- |
| `run-identity` | Sanity-check the SAE hook stack at $\beta = 0$: hooked residuals must match the unhooked residual stream. Later stages should not proceed if this fails. |
| `run-baseline` | Score neutral prompts and freeze order-specific partitions $Q^+$, $Q^-$, and ties. Those partitions stay fixed for the rest of the study. |
| `run-fs` | Gradient-based feature selection: per-order Jacobians for resistance / recovery / preservation surrogates, then a deterministic common feature pool under `feature_selection/`. |
| `run-optimize` | Fit SAE coefficients $\beta$ on the optimization split (CMA-ES or projected Adam) using the selected pool. Writes `optimize/best_checkpoint.json`. |
| `run-freeze` | Seal the study into a `FrozenExperimentConfig` under `freeze/`. Locks revisions, hashes, and selected features before validation / holdout. |
| `run-study` | Post-freeze evaluation on `behavior_validation` with the best checkpoint **and** a β=0 non-intervened comparison (same FTW/CBR/Selectivity/… schema). Writes `full_study/behavioral.json` + `full_study/behavioral_non_intervened.json`. Requires a sealed freeze; does **not** open holdout. |
| `run-holdout` | Final unlock: load holdout only after a sealed freeze, mark `holdout_started`, and write holdout artifacts. One-shot terminal stage. |

**Tests** (`test` / `test-cuda` environments):

| Task | What it does |
| --- | --- |
| `test-fast` | Fast gate: `pytest` excluding `real_model`, `slow`, and `gpu` markers. |
| `test` | Full `pytest` collection for the active environment. |
| `test-gpu` | CUDA env only: `pytest` for `real_model` or `gpu` markers. |

**Lint / typing** (`dev` environment):

| Task | What it does |
| --- | --- |
| `lint` | `ruff check .` |
| `format-check` | `ruff format --check .` |
| `typecheck` | `mypy src tests` |

### Run a study from YAML

Author a StudyConfig under `configs/` (`stack` + `experiment` + `run`). Bare
`--config` needs no injector kwargs: production adapters build scoring,
Jacobians, and the objective from the stack and processed corpus.

Coverage defaults to the **full available split**. Omit `run.fs_coverage` and
omit `run.optimize.n_questions` / `question_ids` unless you intentionally want a
subset (see `configs/dev/`).

```bash
# Limited layer-17 path (explicit fs_coverage / optimize.n_questions)
CFG=configs/dev/layer17_n32_CF.yaml
# alias: configs/dev/layer17_n32.yaml (= CF)

pixi run --environment test-cuda run-identity -- --config "$CFG"
pixi run --environment test-cuda run-baseline -- --config "$CFG"
pixi run --environment test-cuda run-fs -- --config "$CFG"
pixi run --environment test-cuda run-optimize -- --config "$CFG"
pixi run --environment test-cuda run-freeze -- --config "$CFG"
pixi run --environment test-cuda run-study -- --config "$CFG"      # sealed; no holdout
# holdout only after freeze + explicit unlock (DEC-071):
# pixi run --environment test-cuda run-holdout -- --config "$CFG" --freeze-status sealed
```

Each study is a **single** answer-order experiment (`run.order_regime: CF|IF|RO`).
Run CF, IF, and RO as three sealed studies, then assemble the 3×3 matrix.

Operational logs go to **stderr**: stage start/end with `elapsed_s`, optimize/FS
progress, and WARNING audits for freeze seal / holdout unseal. Optional
`--log-level DEBUG|INFO|WARNING|ERROR` (default `INFO`).

Adapter defaults, artifact layout, cross-order assemble, and real-model gates:
[`docs/phase_m_ship_gate.md`](docs/phase_m_ship_gate.md).

### Tests

```bash
# Fast gate (no real_model / slow / gpu)
pixi run --environment test test-fast

# Full pytest collection for the active environment
pixi run --environment test test

# One file or node
pixi run --environment test pytest tests/unit/test_scoring_margins.py -q

# Pinned tiny GPT-2 check (downloads the pinned revision on first run)
pixi run --environment test pytest -m real_model -q

# GPU memory / CUDA-marked tests (requires cuda env + hardware)
pixi run --environment test-cuda test-gpu
```

Markers: `unit`, `property`, `integration`, `real_model`, `slow`, `gpu`.

### Lint and typecheck

```bash
pixi run --environment dev lint
pixi run --environment dev format-check
pixi run --environment dev typecheck
```

### Frozen dataset

The corpus under `data/data_processed/` is the experiment input. Rebuild only
when intentionally regenerating artifacts; downstream work treats the current
manifest as frozen.

```bash
cd data
python -m src \
  --csv data_raw/TruthfulQA.csv \
  --mc-json data_raw/mc_task.json \
  --output-dir data_processed \
  --reports-dir reports \
  --split-seed 42

python -m pytest tests -q
```

Interactive inspection: [`data/get_data.ipynb`](data/get_data.ipynb).

**Split roles (seed 42):**

| Split | Questions | Formats | Role |
| --- | ---: | --- | --- |
| `feature_selection` | 316 | MC0 | Jacobian / feature ranking |
| `optimization` | 237 | MC0 | Coefficient optimization |
| `behavior_validation` | 118 | MC0, MC1, MC2 | Checkpoint / behavioral selection |
| `holdout_test_behavior` | 119 | MC0, MC1, MC2 | Final held-out evaluation |

MC1/MC2 rows are withheld from feature selection and optimization. Artifact
hashes live in `data/reports/split_summary.json`.

### Library entry points

After `pixi install` (or `pixi shell --environment test`):

```python
from epistemic_sycophancy.config import ExperimentConfig, freeze_experiment_config
from epistemic_sycophancy.scoring.margins import truthful_margin
from epistemic_sycophancy.metrics import build_baseline_partition, compute_behavioral_metrics
from epistemic_sycophancy.intervention.sae_delta import apply_additive_sae_delta
from epistemic_sycophancy.feature_selection import (
    coefficient_jacobian,
    rank_suppression_candidates,
    build_common_feature_pool,
)
from epistemic_sycophancy.objective.total import evaluate_objective
from epistemic_sycophancy.optimization import CMAESOptimizer, ProjectedAdam
from epistemic_sycophancy.evaluation import build_cross_order_matrix, run_toy_e2e_baseline
from epistemic_sycophancy.statistics import bootstrap_selectivity_interval
```

Toy end-to-end checks and real-model helpers live under `evaluation/` and are
exercised by the integration / `real_model` tests.

## Further reading

| Document | Contents |
| --- | --- |
| [`docs/epistemic_sycophancy_tdd_spec_v2.md`](docs/epistemic_sycophancy_tdd_spec_v2.md) | Equations, Spec IDs, numerical contracts |
| [`docs/decisions.md`](docs/decisions.md) | Frozen policies (ties, scopes, pools, pins, …) |
| [`docs/phase_m_ship_gate.md`](docs/phase_m_ship_gate.md) | YAML study pipeline, artifacts, real-model gates |
| [`docs/tdd-progress.md`](docs/tdd-progress.md) | Implemented Spec IDs and gate evidence |
| [`AGENTS.md`](AGENTS.md) | TDD workflow for contributors and coding agents |
