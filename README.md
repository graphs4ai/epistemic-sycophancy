# epistemic-sycophancy

Library and experiment scaffolding for **additive SAE interventions** against epistemic sycophancy: sparse-autoencoder steering meant to reduce deference to incorrect beliefs while preserving correction selectivity and baseline truthfulness.

## Phase M — YAML → CLI → optimize → freeze → full_study (final impl)

Author a StudyConfig YAML under `configs/` (`stack` + `experiment` + `run`).
Bare `--config` runs **without injector kwargs** — production adapters build
`score_fn` / Jacobians / objective from the stack + processed corpus.

**Coverage default = full split.** Omit `run.fs_coverage` and omit
`run.optimize.n_questions` / `question_ids` to use every available question in
the stage's split. Subsets are opt-in only (e.g. under `configs/dev/`).

Stage order (DEC-072 / DEC-093):

```text
identity → baseline_partitions → feature_selection
→ optimize → freeze → full_study → holdout_eval
```

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

Operational logs go to **stderr** (DEC-089): stage start/end with `elapsed_s`,
optimize/FS progress, and WARNING audits for freeze seal / holdout unseal.
Optional `--log-level DEBUG|INFO|WARNING|ERROR` (default `INFO`).

See [`docs/phase_m_ship_gate.md`](docs/phase_m_ship_gate.md) for adapter defaults,
single-order artifact layout, cross-order assemble, and ORCH-034…038 real_model
gates (`test-cuda`).

Pixi tasks: `run-identity`, `run-baseline`, `run-fs`, `run-optimize`, `run-freeze`, `run-study`, `run-holdout`. The legacy `"stage … ready"` CLI stub is **deprecated**; use `--config` real dispatch.

What you can also run today: the frozen dataset pipeline, Pixi test/lint tasks, and the Python APIs under `src/epistemic_sycophancy/`.

## How the system works

The codebase is organized as a layered mathematical pipeline. Each layer is a small package with pure or near-pure functions; model I/O and logging sit at the edges.

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

**Data and prompts.** Questions are split once at the original-question level (feature selection, optimization, behavior validation, holdout). Belief-conditioned MC0 prompts are built for neutral / correct / incorrect contexts. Answer order is one of CF (correct-first), IF (incorrect-first), or RO (deterministic random order keyed by seed and question id).

**Scoring.** For two-candidate prompts the semantic margin is always \(M = s_{\text{truthful}} - s_{\text{incorrect}}\), independent of which option is labeled A or B. Candidate scores are next-token logits or summed conditional log-probs. MC1/MC2 use the official multi-candidate formulas when those rows are present.

**Baselines and metrics.** From unmodified *neutral* margins, each order regime freezes \(Q^+\) (truthful) and \(Q^-\) (incorrect), with a configured near-tie band. Those partitions are never recomputed after an intervention. Behavioral rates (accuracy, FTW, CBR, Selectivity, PRA) are question-macro averages conditioned on the frozen subsets.

**Intervention.** Selected SAE features are updated with normalized coefficients \(\beta\): \(\alpha = s \odot \beta\), \(z' = \operatorname{ReLU}(z+\alpha)\), then an additive residual \(\Delta x = \operatorname{decode}(z') - \operatorname{decode}(z)\). At \(\beta = 0\) the hook returns the original residual stream, not the SAE reconstruction. Token scope (last token, last-\(k\), or all prompt tokens) is explicit.

**Feature selection.** Local coefficient Jacobians combine decoder projection, ReLU activity masks, and feature scales. Suppression candidates are ranked by *signed* Jacobian (positive \(J\) predicts that suppressing the feature decreases loss). CF/IF/RO rankings stay separate; a deterministic quota-union builds one common feature pool for all order-specific optimizers. Feature selection may only see the feature-selection split.

**Objective and optimization.** The scalar objective mixes resistance (incorrect belief on \(Q^+\)), recovery (correct belief on \(Q^-\)), neutral and correct-belief preservation hinges, and mean \(|\beta|\). Aggregation is always mean-within-question then mean-across-questions. Optimizers are CMA-ES (derivative-free, full-split trials) and projected Adam (\(\beta\)-only, bounds clamp). Checkpoint selection uses validation metrics only; holdout stays sealed until a frozen config artifact marks holdout started.

**Evaluation.** Cross-order evaluation fills a 3×3 matrix (optimize under × evaluate under CF/IF/RO) without refitting \(\beta\). Controls include random-feature and shuffled-coefficient baselines. Uncertainty uses question-cluster bootstrap (not prompt-row resampling).

## Repository layout

```text
src/epistemic_sycophancy/   # library (importable package)
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

## Setup

[Pixi](https://pixi.sh/) is the only environment and task manager (`pyproject.toml` / `pixi.lock`).

```bash
pixi install
```

| Environment | Use |
| --- | --- |
| `default` | Runtime (CPU PyTorch) |
| `test` | Runtime + pytest, Hypothesis, transformers |
| `dev` | test + ruff, mypy |
| `cuda` / `test-cuda` | GPU PyTorch (linux-64-cuda) |

## Available actions

### Run tests

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

### Rebuild or validate the dataset

The frozen corpus under `data/data_processed/` is the experiment input. Rebuild only when intentionally regenerating artifacts (downstream work treats the current manifest as frozen).

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

MC1/MC2 rows are withheld from feature selection and optimization. Artifact hashes live in `data/reports/split_summary.json`.

### Use the library from Python

After `pixi install` (or `pixi shell --environment test`), import packages directly. Typical entry points:

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

Toy end-to-end checks and real-model check helpers live under `evaluation/` and are exercised by the integration / `real_model` tests. Full-scale model training loops are not wired as a CLI yet; compose the APIs above (or follow the tests as usage examples).

## Further reading

| Document | Contents |
| --- | --- |
| [`docs/epistemic_sycophancy_tdd_spec_v2.md`](docs/epistemic_sycophancy_tdd_spec_v2.md) | Equations, Spec IDs, numerical contracts |
| [`docs/decisions.md`](docs/decisions.md) | Frozen policies (ties, scopes, pools, pins, …) |
| [`docs/tdd-progress.md`](docs/tdd-progress.md) | Implemented Spec IDs and gate evidence |
| [`AGENTS.md`](AGENTS.md) | TDD workflow for contributors and coding agents |
