# epistemic-sycophancy

Research codebase for SAE interventions against epistemic sycophancy: additive sparse-autoencoder steering that aims to reduce deference to incorrect beliefs while preserving correction selectivity and baseline truthfulness.

Implementation follows strict TDD. See [`AGENTS.md`](AGENTS.md) and [`docs/epistemic_sycophancy_tdd_spec_v2.md`](docs/epistemic_sycophancy_tdd_spec_v2.md).

## Datasets

The [`data/`](data/) directory holds the frozen TruthfulQA-derived experiment corpus: raw sources, deterministic question-level splits, belief-conditioned prompt variants, and validation reports.

Splits are assigned **at the original-question level before variant generation**, so belief paraphrases, answer-order variants, and MC formats cannot leak across development and holdout partitions.

### Layout

```text
data/
  data_raw/                 # upstream TruthfulQA sources
    TruthfulQA.csv
    mc_task.json
  data_processed/           # frozen experiment artifacts
    split_manifest.csv      # one row per question_id → split
    questions.parquet
    belief_candidates.parquet
    belief_triples.parquet
    semantic_filter.jsonl
    feature_selection/mc0.jsonl
    optimization/mc0.jsonl
    behavior_validation/{mc0,mc1,mc2}.jsonl
    holdout_test_behavior/{mc0,mc1,mc2}.jsonl
  reports/                  # split quality + validation summaries
  src/                      # deterministic build pipeline
  tests/                    # pipeline unit tests
  get_data.ipynb            # interactive rebuild / inspection notebook
```

### Split counts (seed = 42)

| Split | Questions | Formats emitted | Role |
| --- | ---: | --- | --- |
| `feature_selection` | 316 | MC0 | Jacobian / feature ranking only |
| `optimization` | 237 | MC0 | Coefficient optimization |
| `behavior_validation` | 118 | MC0, MC1, MC2 | Checkpoint / behavioral selection |
| `holdout_test_behavior` | 119 | MC0, MC1, MC2 | Final held-out evaluation |
| **Total** | **790** | | |

Feature selection and optimization intentionally emit **MC0 only**. MC1/MC2 rows are reserved for validation and holdout behavioral evaluation.

### Prompt variants

Each MC0 JSONL row is a fully rendered prompt with metadata such as `question_id`, `split`, `format`, `belief_condition` (`neutral` / `correct` / `incorrect`), answer order, option text, and question-level weights.

Belief candidates are matched into triples rather than a full Cartesian product of paraphrases. Current validated totals (`data/reports/validation_report.json`):

- 6 025 belief candidates
- 1 363 belief triples covering 750 questions
- 107 rejected beliefs (see `data/reports/rejected_beliefs.csv`)

### Rebuild / validate

From the `data/` directory (requires the pipeline dependencies used by that package):

```bash
cd data
python -m src \
  --csv data_raw/TruthfulQA.csv \
  --mc-json data_raw/mc_task.json \
  --output-dir data_processed \
  --reports-dir reports \
  --split-seed 42
```

Or step through [`data/get_data.ipynb`](data/get_data.ipynb).

Pipeline tests:

```bash
cd data && python -m pytest tests -q
```

Artifacts and source hashes are recorded in `data/reports/split_summary.json`. Do not recompute splits or regenerate variants casually: downstream experiments treat the current manifest as frozen.

## Development

Environment and tasks are managed with [Pixi](https://pixi.sh/) (`pyproject.toml` / `pixi.lock`):

```bash
pixi install
pixi run --environment test pytest -q
```

TDD progress is tracked in [`docs/tdd-progress.md`](docs/tdd-progress.md). Experimental policy decisions belong in [`docs/decisions.md`](docs/decisions.md).
