# AGENTS.md — Test-Driven Implementation Contract

## 1. Purpose

This repository must be implemented with strict test-driven development.

The governing workflow is not:

> write the complete test suite, then write the complete implementation.

It is:

> select the smallest meaningful behavior → write a failing test → confirm the expected failure → implement the minimum code that passes → refactor safely → run the relevant regression suite → record the result → repeat.

The implementation concerns additive sparse-autoencoder interventions for reducing epistemic sycophancy while preserving correction selectivity and baseline truthfulness. Mathematical fidelity takes precedence over convenience, speed, or matching an existing partial implementation.

This file governs all coding agents working in the repository.

---

## 2. Sources of truth and precedence

Apply instructions in this order:

1. The user's explicit request for the current task.
2. This `AGENTS.md`.
3. The current TDD specification, expected at:
   - `docs/epistemic_sycophancy_tdd_spec_v2.md`, or
   - the path explicitly supplied by the user.
4. Frozen experiment configuration and decision records.
5. Existing tests.
6. Existing production code.

Existing code is not authoritative when it conflicts with the specification or a mathematically justified test.

Do not silently resolve a material ambiguity. If the specification intentionally leaves a policy open—such as tie handling, invalid-row handling, feature-pool quotas, or candidate continuation strings—make that policy explicit in configuration and test the configured behavior. Record the selected policy in `docs/decisions.md`.

---

## 3. Non-negotiable TDD cycle

Every implementation change must belong to a traceable cycle.

### 3.1 Select

Choose exactly one of the following:

- one specification test ID, such as `SCORE-002`;
- a very small cluster of inseparable test IDs;
- one regression bug with a minimal reproducer.

A cluster is acceptable only when the tests exercise the same production behavior and cannot reasonably pass independently.

Before editing production code:

1. Locate the relevant specification section.
2. Inspect the existing module, tests, fixtures, and configuration.
3. State the behavior being implemented.
4. Identify the mathematical invariant and expected failure mode.
5. Choose the narrowest test command that will demonstrate the cycle.

Do not implement unrelated cleanup in the same cycle.

### 3.2 Red

Write the smallest test that expresses the required behavior.

Then run it and confirm that it fails for the intended reason.

A valid red state must satisfy all of the following:

- the test is collected;
- the test reaches the behavior under development;
- the failure is an assertion failure or a clearly expected missing-interface failure;
- the failure message is consistent with the missing behavior;
- the failure is not caused by a syntax error, broken fixture, missing unrelated dependency, incorrect import path, or malformed test.

Record the command and concise failure reason.

Do not change production behavior before observing a valid red state.

Permitted pre-red changes are limited to:

- an empty module or importable public stub needed for test collection;
- a type or protocol declaration with no functional behavior;
- test-only fixtures;
- configuration plumbing that has no implementation semantics.

These exceptions must not make the new test pass.

### 3.3 Green

Implement the minimum production code required to pass the new test.

During green:

- do not add untested branches;
- do not generalize beyond the tested contract;
- do not optimize prematurely;
- do not weaken or delete the new assertion;
- do not alter golden expected values to match the implementation;
- do not copy production output into the test as the expected result;
- do not silently change the mathematical definition.

Run the targeted test until it passes.

Then run the immediately related module tests.

### 3.4 Refactor

Refactor only after the targeted behavior is green.

Allowed refactors include:

- removing duplication;
- improving names;
- extracting pure functions;
- narrowing interfaces;
- simplifying tensor shapes;
- improving error messages;
- replacing a slow correct implementation with an equivalent optimized implementation.

A refactor must not change observable behavior.

For mathematical or vectorized refactors, retain a slow scalar or dense reference implementation in tests when practical. Test the optimized implementation against that reference.

### 3.5 Verify

After refactoring, run:

1. the new targeted test;
2. all tests in the affected module;
3. all fast tests for directly dependent modules;
4. the full fast suite at the end of a coherent slice.

Real-model, GPU, or slow tests are run only when the change touches their contract or at a scheduled phase gate.

A cycle is incomplete while any newly introduced warning, flaky result, NaN, skipped required test, or nondeterministic mismatch remains unexplained.

### 3.6 Record

Update `docs/tdd-progress.md` after each completed cycle.

Use this table:

| Spec ID | Test | Status | Red evidence | Green evidence | Production files | Notes |
|---|---|---|---|---|---|---|
| SCORE-002 | `test_margin__swapping_candidate_positions_and_scores__preserves_semantic_margin` | green | `...` | `...` | `src/scoring/margins.py` | `...` |

Allowed statuses:

- `not_started`
- `red`
- `green`
- `blocked`
- `deferred`
- `superseded`

Do not mark a test `green` solely because an equivalent-looking test exists. Run it and record the result.

---

## 4. Prohibited false-TDD behavior

Never do any of the following:

1. Implement a complete module and add tests afterward.
2. Write dozens of placeholder tests without cycling them through red and green.
3. Make a failing test pass by loosening tolerances without a numerical justification.
4. Replace exact assertions with `is not None`, shape-only checks, or broad range checks when the specification gives exact values.
5. Use the current implementation to generate the expected values for its own test.
6. Delete, skip, mark `xfail`, or weaken a test because the implementation is difficult.
7. Mock the mathematical operation being tested.
8. Mock the SAE delta, margin, aggregation, or Jacobian in a test intended to validate that same operation.
9. Pool prompt rows when the specification requires averaging within question and then across questions.
10. Recompute baseline partitions after applying an intervention.
11. resample random answer order during optimization trials.
12. Access validation or holdout data from feature selection.
13. Access holdout data during optimizer or checkpoint selection.
14. replace the residual stream with the SAE reconstruction.
15. rank suppression-only candidates by absolute Jacobian magnitude while discarding the sign.
16. treat raw decoder projection as the normalized-coefficient Jacobian without the activity mask and feature scale.
17. average gradients before projection when prompt-specific masks are required for the exact Jacobian.
18. silently accept NaN, infinity, missing scores, malformed variants, or empty conditional subsets.
19. add a fallback whose semantics are not tested.
20. change a frozen experiment definition to accommodate existing code.

---

## 5. Mathematical invariants that tests must protect

These invariants are project-defining.

### 5.1 Semantic truthful margin

For every order regime:

\[
M = s_{\text{truthful}} - s_{\text{incorrect}}.
\]

Under correct-first:

\[
M=s_A-s_B.
\]

Under incorrect-first:

\[
M=s_B-s_A.
\]

Swapping candidate positions and their scores must preserve the semantic margin.

### 5.2 Two-candidate truthful probability

\[
p_{\mathrm{truth}}^{A/B}=\sigma(M).
\]

This is distinct from valid-answer mass under the full vocabulary.

### 5.3 Logistic margin loss

\[
\phi(M)=\operatorname{softplus}\left(-\frac{M}{\tau}\right),
\qquad \tau>0.
\]

The implementation must be numerically stable for extreme margins.

### 5.4 Loss before averaging

For belief variants:

\[
\operatorname{mean}_b \phi(M_b)
\]

is required.

Do not replace it with:

\[
\phi\left(\operatorname{mean}_b M_b\right).
\]

### 5.5 Question-macro aggregation

The required sequence is:

```text
score each concrete prompt
→ compute each prompt loss or indicator
→ average within original question
→ average across original questions
```

Each original question has equal conceptual weight within a component, regardless of its number of variants.

### 5.6 Frozen, order-specific baseline subsets

For each evaluation order, derive \(Q^+\), \(Q^-\), and any configured tie subset from the unmodified neutral baseline.

The partition:

- is order-specific;
- uses neutral baseline margins only;
- is frozen for the entire study;
- is never recomputed from intervened predictions;
- is selected by evaluation order during cross-order evaluation.

### 5.7 Additive SAE delta decoding

For selected features:

\[
\alpha_j=s_j\beta_j,
\qquad
z'_j=\operatorname{ReLU}(z_j+\alpha_j),
\]

\[
\Delta x=\operatorname{decode}(z')-\operatorname{decode}(z),
\qquad
x'=x+\Delta x.
\]

At \(\beta=0\), the hook must return the original residual stream, not the SAE reconstruction.

### 5.8 Exact local coefficient Jacobian

For prompt \(p\), layer \(\ell\), component \(u\), and feature \(j\):

\[
\left.
\frac{\partial \ell_u^{(p)}}{\partial\beta_{\ell,j}}
\right|_{\beta=0}
=
s_{\ell,j}
\mathbf 1[z_{\ell,j,t_p^*}^{(p)}>0]
\left\langle
g_{\ell,u}^{(p)},d_{\ell,j}
\right\rangle.
\]

The exact Jacobian includes:

- the decoder-direction projection;
- the prompt-specific ReLU activity mask;
- the feature scale;
- the configured token-scope contribution;
- question-macro weights.

For a multi-token intervention scope, sum the chain-rule contribution over all intervened positions.

### 5.9 Feature-selection components

Use separate component outputs for:

- resistance;
- recovery;
- neutral preservation surrogate;
- correct-belief preservation surrogate.

The optimizer's hinge preservation penalties remain unchanged. Feature selection uses non-flat preservation surrogates because the hinge penalties are locally flat at the null intervention.

### 5.10 Suppression-only sign

For a feasible negative coefficient change:

\[
\Delta L \approx J_j\Delta\beta_j.
\]

Therefore:

- \(J_j>0\) predicts that suppression decreases the loss;
- \(J_j<0\) predicts that suppression increases the loss.

Rank beneficial suppression candidates by descending signed Jacobian. Keep magnitude as a separate diagnostic.

### 5.11 Order-specific Jacobians and common pool

Compute feature rankings separately for CF, IF, and RO.

Construct the common candidate pool with a frozen deterministic union/quota rule. Do not average order-specific scores in a way that cancels strong candidates with opposing signs.

---

## 6. Test architecture

Use four levels.

### 6.1 Pure unit tests

No transformer and no real SAE.

Use these for:

- configuration validation;
- manifests and leakage checks;
- prompt structures;
- margins;
- scalar losses;
- question-macro aggregation;
- baseline partitions;
- behavioral metrics;
- MC1 and MC2 formulas;
- deterministic pool construction;
- bootstrap mechanics.

Prefer `float64` for scalar and toy numerical tests.

### 6.2 Property tests

Use Hypothesis or an equivalent framework for invariants such as:

- margin swap invariance;
- monotonicity of sigmoid and logistic loss;
- bounded metrics;
- row-order invariance;
- question-weight invariance;
- MC2 shift invariance;
- suppression not increasing nonnegative latents;
- serialization round-trip equivalence.

Property tests supplement, but do not replace, exact examples and golden fixtures.

### 6.3 Toy integration tests

Use a deterministic toy model and a linear toy SAE.

The toy system must be small enough for:

- hand-computed logits;
- hand-computed latent updates;
- direct autograd;
- central finite differences in smooth regions;
- feasible one-sided finite differences at the suppression boundary;
- dense versus feature-chunked projection comparisons;
- batched versus scalar comparisons;
- end-to-end objective calculations.

Do not use a real transformer to validate basic algebra.

### 6.4 Real-model checks and regression tests

Keep these separate and marked.

Use them for:

- tokenizer continuation regression;
- hook tensor shape and position;
- identity equivalence;
- deterministic scoring;
- beta-only gradient viability;
- one tiny objective evaluation;
- memory regression.

Real-model tests must use pinned revisions and a fixed prompt fixture.

---

## 7. Pytest conventions

### 7.1 Test names

Use:

```text
test_<module>__<behavior>__<expected_result>
```

Examples:

```python
test_margin__incorrect_first__truthful_score_is_still_subtracted_first
test_objective__variant_imbalance__preserves_equal_question_weight
test_feature_projection__active_scaled_derivative__matches_autograd
```

### 7.2 Markers

Register and use:

```python
@pytest.mark.unit
@pytest.mark.property
@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
```

Do not use a marker to hide a required fast test.

### 7.3 Test structure

Prefer Arrange–Act–Assert.

Each test should have one principal reason to fail. Multiple assertions are acceptable when they jointly define one behavior.

Use explicit local names that reflect the mathematics:

```python
truthful_score
incorrect_score
truthful_margin
question_loss
feature_scale
activity_mask
projected_gradient
coefficient_jacobian
```

Avoid one-letter names in production code except where a short local tensor expression is clearer and directly corresponds to a documented equation.

### 7.4 Fixtures

Fixtures must be:

- deterministic;
- minimal;
- semantically named;
- reusable only when they genuinely represent the same setup.

Do not create a global fixture with hundreds of fields.

Keep golden fixtures under `tests/fixtures/` and version them.

### 7.5 Numerical assertions

Use tolerances from the TDD specification.

Do not increase a tolerance merely because a test fails. First determine whether the difference comes from:

- dtype;
- operation order;
- device;
- non-deterministic kernel;
- incorrect masking;
- incorrect denominator;
- incorrect sequence position;
- an actual mathematical error.

When an optimized implementation changes floating-point reduction order, compare it with the reference implementation under a documented tolerance and retain exact structural assertions.

---

## 8. Production-code design rules

### 8.1 Prefer pure functions

Mathematical operations should be pure whenever possible:

```python
truthful_margin(...)
logistic_margin_loss(...)
question_macro_mean(...)
build_baseline_partition(...)
compute_behavioral_metrics(...)
project_residual_gradient(...)
apply_latent_delta(...)
```

Keep model execution, file I/O, logging, and configuration parsing outside core mathematical functions.

### 8.2 Narrow modules

Use the package boundaries defined in the TDD specification. Do not create a monolithic experiment module.

At minimum, separate:

- configuration;
- data validation and manifests;
- prompt ordering;
- continuation scoring;
- SAE intervention;
- feature selection;
- objective aggregation;
- behavioral metrics;
- statistics;
- optimization;
- artifact logging.

### 8.3 Typed public interfaces

Use type hints for public functions and dataclasses or validated models for structured records.

Validate shapes, IDs, and policy fields at boundaries. Fail early with domain-specific exceptions.

Examples:

```text
InvalidExperimentConfig
DataIntegrityError
DegenerateBaselineError
InvalidScoreError
ManifestMismatchError
HoldoutAccessError
```

### 8.4 Explicit tensor contracts

Document tensor shapes in docstrings.

Example:

```python
def project_residual_gradient(
    gradient: Tensor,      # [..., d_model]
    decoder: Tensor,       # [n_features, d_model]
) -> Tensor:               # [..., n_features]
    ...
```

Check layer, width, dtype, and device compatibility before expensive runs.

### 8.5 Reference before optimization

Implement a correct scalar or dense reference first.

Only then implement:

- vectorization;
- prompt batching;
- feature chunking;
- streamed accumulation;
- custom kernels;
- memory-saving hooks.

The optimized path must be tested against the reference path.

### 8.6 Determinism

Any function that depends on randomness must accept an explicit seed or manifest.

Never rely on process-global random state for:

- split assignment;
- RO assignment;
- random-feature controls;
- bootstrap replicates;
- optimizer initialization.

Record seeds and manifest hashes in artifacts.

### 8.7 No hidden policy

Do not encode experimental policy in scattered conditionals.

Policies such as tie handling, invalid rows, candidate scoring, coefficient bounds, and pool construction belong in validated configuration and artifact metadata.

---

## 9. Implementation order and phase gates

Follow the specification order. Do not jump to the optimizer before the mathematical substrate is green.

### Phase A — Configuration and data integrity

Implement cyclically:

1. configuration validation;
2. exact split counts;
3. one-question-one-split;
4. derived-variant inheritance;
5. content-hash leakage;
6. target joins;
7. reproducibility metadata.

Gate:

- no known split leakage;
- unresolved material policies are explicit;
- manifests are deterministic and hashed.

### Phase B — Prompts and ordering

Implement:

1. structured prompt representation;
2. CF mapping;
3. IF mapping;
4. deterministic RO;
5. RO reuse across conditions, variants, and trials;
6. continuation-string contract.

Gate:

- order manifests reproduce exactly;
- prompt conditions differ only in the intended belief block;
- tokenizer contract is versioned.

### Phase C — Scoring and scalar losses

Implement:

1. semantic margin;
2. option-swap invariance;
3. sigmoid equivalence;
4. single-token scoring;
5. multi-token scoring;
6. padding invariance;
7. valid-answer mass;
8. logistic loss and stability.

Gate:

- scalar and batched scoring agree;
- no non-finite score is silently accepted.

### Phase D — Aggregation, partitions, and metrics

Implement:

1. question-macro utilities;
2. order-specific baseline partitions;
3. frozen partition artifacts;
4. FTW;
5. CBR;
6. Selectivity;
7. PRA-mean;
8. PRA-all;
9. MC1 and MC2;
10. denominator reporting.

Gate:

- golden metric fixtures pass;
- cross-order denominator selection is correct;
- tie and degenerate-subset policies are tested.

### Phase E — SAE intervention

Implement:

1. normalized coefficients;
2. selected-latent update;
3. ReLU clamping;
4. linear decoder delta;
5. no-reconstruction replacement;
6. token-scope masking;
7. identity logits;
8. identity margins and decisions;
9. beta-only gradients.

Gate:

- identity tests pass at all required levels;
- optimization remains blocked if identity fails.

### Phase F — Gradient-based feature selection

Implement:

1. prompt-specific final-token indexing;
2. raw decoder projection;
3. activity mask;
4. feature scales;
5. exact coefficient Jacobian;
6. autograd equivalence;
7. finite-difference equivalence;
8. question-macro gradient weighting;
9. feature-chunked projection;
10. streaming accumulation;
11. component isolation;
12. separate CF/IF/RO rankings;
13. deterministic common-pool construction;
14. artifact fingerprints and leakage gates.

Gate:

- projected Jacobian matches direct autograd and finite differences on toy fixtures;
- real-model spot check passes;
- no downstream split is accessed.

### Phase G — Full objective

Implement each term separately:

1. resistance;
2. recovery;
3. behavioral combination;
4. neutral hinge;
5. correct-belief hinge;
6. coefficient regularization;
7. total objective;
8. required logging.

Gate:

- the golden objective fixture matches every component and total;
- batched and unbatched objective and gradient agree;
- the objective is deterministic.

### Phase H — Optimizers

Implement CMA-ES wrapper first or according to the task request, then projected Adam.

For each optimizer, cycle through:

- bounds;
- data coverage;
- deterministic objective use;
- state serialization;
- logging;
- budget accounting.

For Adam additionally test:

- beta is the only trainable parameter;
- projected clamping;
- correct microbatch gradient accumulation;
- zero-learning-rate behavior.

Gate:

- optimizer tests do not require holdout access;
- checkpoints round-trip exactly;
- one toy optimization improves the pinned toy objective.

### Phase I — Statistics, controls, and cross-order evaluation

Implement:

- question-cluster bootstrap;
- paired resampling;
- Selectivity recomputation;
- complete 3 × 3 order matrix;
- random-feature controls;
- shuffled-coefficient controls;
- phase gates.

### Phase J — Real-model and release checks

Run:

- tokenizer regression;
- hook contract;
- identity suite;
- deterministic objective repeat;
- gradient viability;
- memory regression;
- complete fast suite;
- required GPU suite.

Do not open the holdout until the frozen configuration artifact exists.

---

## 10. Golden values and independent expected results

When the specification supplies exact expected values, transcribe them directly and preserve sufficient precision.

For new golden cases:

1. derive the expected result by hand, symbolic algebra, or a separate simple reference implementation;
2. document the derivation in the test or fixture;
3. do not call the production function to construct its own expectation;
4. do not update the fixture automatically when production code changes.

A fixture update requires a documented specification change or a demonstrated error in the previous independent derivation.

---

## 11. Regression workflow

For every bug:

1. reproduce it with the smallest failing test;
2. confirm the test fails on the current code;
3. implement the minimum fix;
4. run the local regression tests;
5. run affected phase-gate tests;
6. record the bug and fix in `docs/tdd-progress.md`.

A bug fix without a reproducing test is incomplete unless the failure cannot be represented in the repository. In that exceptional case, document why.

---

## 12. Commands, environment and dependency management

Pixi is the sole environment, dependency, task, and lockfile manager for this repository.

### Required command interface

Agents must use:

```bash
pixi add ...
pixi remove ...
pixi install
pixi lock
pixi run ...
pixi shell ...
pixi task ...
```

Agents must not use the following for project dependency or environment management:

```bash
uv init
uv add
uv remove
uv sync
uv lock
pip install
python -m pip install
conda install
mamba install
```

Pixi may use uv internally for PyPI resolution and installation. This does not authorize agents to create a separate uv-managed environment or `uv.lock`.

### Authoritative artifacts

The authoritative environment artifacts are:

```text
pyproject.toml
pixi.lock
```

Agents must commit changes to both files when dependency changes modify the lockfile.

The following must not be introduced:

```text
uv.lock
requirements.txt
requirements-dev.txt
environment.yml
Pipfile
poetry.lock
.venv/
```

An exception requires an explicit user request and a documented decision in `docs/decisions.md`.

### Running commands

Agents should run commands through Pixi tasks when a corresponding task exists:

```bash
pixi run test
pixi run test-fast
pixi run lint
pixi run typecheck
```

For a targeted TDD test, use the explicit test environment:

```bash
pixi run --environment test pytest <test-node-id> -q
```

Agents must record the complete Pixi command in `docs/tdd-progress.md` as red or green evidence.

### Adding dependencies

Before adding a dependency, agents must:

1. Confirm that the dependency is necessary for the current TDD slice.
2. Prefer Conda packages from the configured Pixi channels for compiled or system-level dependencies.
3. Use PyPI dependencies only when appropriate or when the package is unavailable or unsuitable through the configured channels.
4. Add testing tools to the `test` feature.
5. Add linting, formatting, typing, and interactive development tools to the `dev` feature.
6. Avoid adding runtime dependencies to development-only features.

Examples:

```bash
pixi add numpy
pixi add --feature test pytest
pixi add --feature dev ruff mypy
pixi add --pypi some-python-package
```

Agents must not manually edit resolved versions in `pixi.lock`.

### Environment isolation

The standard environments are:

```text
default  Runtime and core project dependencies
test     Runtime dependencies plus testing tools
dev      Runtime, testing, linting, typing, and development tools
```

GPU or CUDA dependencies must be introduced through a separate feature and environment, such as `cuda`, rather than being forced into every environment.

### Reproducibility

CI and agent commands must use the committed lockfile. Dependency changes must not be mixed with unrelated mathematical or behavioral changes unless the dependency is required for that exact TDD cycle.

---

## 13. Agent work protocol

At the start of a coding session, report:

```text
Current slice:
Specification IDs:
Expected red behavior:
Targeted test command:
Production area that may change:
```

After observing red, report the actual failure briefly.

After green/refactor, report:

```text
Completed slice:
Red evidence:
Green evidence:
Files changed:
Mathematical invariant protected:
Remaining relevant tests:
```

Do not claim that a test passes unless it was run in the current environment.

Do not claim the full suite passes after running only a targeted test.

Do not claim mathematical equivalence without a test, derivation, or both.

Do not commit, push, open a pull request, modify remote resources, or start a long real-model experiment unless explicitly requested.

---

## 14. Definition of done for one TDD cycle

A cycle is done only when:

- the selected specification behavior has an executable test;
- the test was observed failing for the intended reason;
- the minimum implementation passes it;
- related tests pass;
- refactoring did not alter behavior;
- progress tracking is updated;
- no material policy was chosen silently;
- no downstream split or holdout leakage was introduced.

---

## 15. Definition of done for a feature or module

A module is done only when:

- all assigned specification IDs are green;
- exact and property tests both pass where required;
- optimized implementations match references;
- input validation and failure behavior are tested;
- public interfaces are typed and documented;
- deterministic artifacts include required metadata;
- the relevant phase gate passes;
- no tests are skipped merely because they are inconvenient.

---

## 16. When blocked

If a cycle is blocked by a material experimental decision:

1. do not guess silently;
2. implement the policy boundary or configuration field if that itself is unambiguous;
3. add a focused test showing that the policy must be explicit;
4. mark the affected behavior `blocked`;
5. record the exact decision required in `docs/decisions.md`;
6. continue with independent tests that do not require that choice.

If blocked by an infrastructure problem, distinguish it from a product failure. Record the command, error, and affected tests. Do not mark the behavior green.

---

## 17. Final principle

The objective is not merely to obtain a passing suite.

The objective is to build an executable mathematical specification in which:

- every important equation has a direct test;
- every experimental partition is immutable and auditable;
- every optimized implementation is checked against a simpler reference;
- every behavioral claim is computed at the original-question level;
- every intervention can be reproduced from versioned artifacts;
- every new behavior enters through a red–green–refactor cycle.
