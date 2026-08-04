#!/bin/bash

################################################################################
# Quick reminder:
# Pipeline order: identity → baseline → feature selection → optimize → freeze → full study → holdout.
#
#   • run-identity — Sanity-check the SAE hook stack at β = 0: hooked residuals must match the unhooked residual stream (identity gate). Blocks later stages if this fails.
#
#   • run-baseline — Score neutral prompts on the feature-selection split and freeze order-specific partitions \(Q^+\), \(Q^-\), ties. These stay fixed for the rest of the study.
#
#   • run-fs — Gradient-based feature selection: per-order Jacobians for resistance / recovery / preservation surrogates, then a deterministic common feature pool written under
#     feature_selection/.
#
#   • run-optimize — Fit SAE coefficients β on the optimization split (CMA-ES or projected Adam), using the selected pool and identity gate. Writes optimize/best_checkpoint.json.
#
#   • run-freeze — Seal the study into a FrozenExperimentConfig artifact (freeze/). Locks revisions, hashes, and selected features before validation/holdout.
#
#   • run-study — Post-freeze evaluation on behavior_validation with the best checkpoint (FTW, CBR, Selectivity, etc.). Requires sealed freeze; does not open holdout.
#
#   • run-holdout — Final unlock: load holdout only after a sealed freeze, mark holdout_started, and write holdout artifacts. One-shot terminal stage (DEC-071).
################################################################################

# Quit if any command fails
set -e

# Check if CFG is set and the config file exists
if [ -z "$CFG" ] || [ ! -f "$CFG" ]; then
    echo "CFG is not set or the config file does not exist"
    exit 1
fi

# Run the baseline partitioning
pixi run --environment test-cuda run-baseline -- --config "$CFG"

# Run the feature selection
pixi run --environment test-cuda run-fs -- --config "$CFG"

# Run the optimization
pixi run --environment test-cuda run-optimize -- --config "$CFG"

# Run the freeze
pixi run --environment test-cuda run-freeze -- --config "$CFG"

# Run the study
pixi run --environment test-cuda run-study -- --config "$CFG"

# Holdout evaluation is set aside for now
