# TDD Progress

This file records observed red–green–refactor cycles.

| Spec ID | Test | Status | Red evidence | Green evidence | Production files | Notes |
|---|---|---|---|---|---|---|
| CFG-001 | `test_config__tau_nonpositive__raises_validation_error` | green | `pixi run --environment test pytest tests/unit/test_config_schema.py::test_config__tau_nonpositive__raises_validation_error -q` → `Failed: DID NOT RAISE InvalidExperimentConfig` (tau=0.0) | same command → `1 passed`; `pixi run --environment test pytest tests/unit -q` → `1 passed` | `src/epistemic_sycophancy/config/schema.py`, `src/epistemic_sycophancy/config/__init__.py` | Rejects `tau <= 0`; accepts finite positive `tau=1.0`. |
| CFG-002 | `test_config__negative_penalty_or_tolerance__raises_validation_error` | green | `pixi run --environment test pytest tests/unit/test_config_schema.py::test_config__negative_penalty_or_tolerance__raises_validation_error -q` → `TypeError: ... unexpected keyword argument 'lambda_n'` | same command → `1 passed`; module suite → `2 passed` | `src/epistemic_sycophancy/config/schema.py` | `lambda_n`, `lambda_c`, `lambda_beta`, `delta_n`, `delta_c` must be `>= 0`. |
| CFG-003 | `test_config__behavior_weights__are_nonnegative_and_normalized` | green | `pixi run --environment test pytest tests/unit/test_config_schema.py::test_config__behavior_weights__are_nonnegative_and_normalized -q` → `TypeError: ... unexpected keyword argument 'w_r'` | same command → `1 passed`; module suite → `3 passed` | `src/epistemic_sycophancy/config/schema.py` | Adopted recommended contract; see DEC-002. |
| CFG-004 | `test_config__suppression_only_bounds__cannot_include_positive_beta` | green | `pixi run --environment test pytest tests/unit/test_config_schema.py::test_config__suppression_only_bounds__cannot_include_positive_beta -q` → `TypeError: ... unexpected keyword argument 'beta_lower'` | same command → `1 passed`; module suite → `4 passed` | `src/epistemic_sycophancy/config/schema.py` | Requires `beta_lower <= beta_upper <= 0`; accepts recommended `[-2, 0]`. |
| CFG-005 | `test_config__feature_scales__must_be_finite_and_positive` | green | `pixi run --environment test pytest tests/unit/test_config_schema.py::test_config__feature_scales__must_be_finite_and_positive -q` → `TypeError: ... unexpected keyword argument 'feature_ids'` | same command → `1 passed`; module suite → `5 passed` | `src/epistemic_sycophancy/config/schema.py` | Rejects non-finite/non-positive scales, duplicate IDs, and length mismatches. |
| CFG-006 | `test_config__tie_and_invalid_row_policies__must_be_explicit` | green | `pixi run --environment test pytest tests/unit/test_config_schema.py::test_config__tie_and_invalid_row_policies__must_be_explicit -q` → `AssertionError: 'tie_policy' not in signature.parameters` | same command → `1 passed`; `pixi run --environment test pytest tests/unit -q` → `6 passed` | `src/epistemic_sycophancy/config/schema.py` | No defaults; `None` rejected. Specific policy values remain TBD (DEC-003). |
| DATA-002 | `test_dataset__question_id__appears_in_exactly_one_split` | green | `pixi run --environment test pytest tests/unit/test_dataset_validation.py::test_dataset__question_id__appears_in_exactly_one_split -q` → `Failed: DID NOT RAISE DataIntegrityError` (stub no-op) | same command → `1 passed`; `pixi run --environment test pytest tests/unit -q` → `7 passed` | `src/epistemic_sycophancy/data/validation.py`, `src/epistemic_sycophancy/data/__init__.py` | For every `question_id`, `n_unique(split) == 1`; same-split repeats allowed. |

## Status definitions

- `not_started`: no test written.
- `red`: test exists and has been observed failing for the intended reason.
- `green`: targeted and affected tests pass.
- `blocked`: a material decision or infrastructure issue prevents completion.
- `deferred`: intentionally postponed with a documented reason.
- `superseded`: replaced by another explicitly identified test or specification revision.
