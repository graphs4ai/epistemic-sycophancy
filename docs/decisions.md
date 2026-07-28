# Experimental and Implementation Decisions

Record material choices that the specification intentionally leaves open.

| Decision ID | Topic | Selected policy | Alternatives considered | Rationale | Tests affected | Date |
|---|---|---|---|---|---|---|
| DEC-001 | Tie policy | Identify near/exact ties with the BASE-005 band (`M > +ε → Q+`, `M < -ε → Q-`, otherwise `Q_tie`), then **merge `Q_tie` into `Q-`** for all conditional metrics and denominators. Always report `n_q_tie` (pre-merge count). Config value: `tie_policy="merge_into_q_minus"`. Exact `ε` is still open (must be explicit in config when BASE-005 is implemented). | (A) exclude `Q_tie` from conditional metrics; (B) keep a separate `Q⁰`/`Q_tie` denominator; (C) merge into `Q-` (selected) | Keeps resistance/recovery denominators defined without dropping borderline questions; matches the user's freeze for the initial study | CFG-006, BASE-005, SCORE-003, METRIC-010, SAE-009, MC-002 | 2026-07-25 |
| DEC-002 | Behavioral weight normalization | Require `w_r >= 0`, `w_u >= 0`, and exact `w_r + w_u == 1` | Allow unnormalized weights and log their sum | Spec recommended contract for CFG-003; keeps objective mixing unambiguous | CFG-003 | 2026-07-24 |
| DEC-003 | Unresolved experiment policies | Must be explicit constructor fields with no defaults; `None` forbidden. Tie disposition is frozen by DEC-001; invalid-row, multi-token scoring, RO manifest selection, and tie-band `ε` remain open | Hidden defaults in code | CFG-006 forbids silent defaults; concrete values freeze via numbered DEC rows | CFG-006, SCORE-011 | 2026-07-24 |

A decision is not frozen until its policy is explicit, validated in configuration, and covered by tests.
