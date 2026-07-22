#!/bin/bash
# Out-of-sample prediction pipeline shell script
# Validates environment variables and runs all three pipeline steps

set -e

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# ────────────────────────────────────────────────────────────────────────────
# Functions
# ────────────────────────────────────────────────────────────────────────────

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

log_section() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "$1"
    echo "════════════════════════════════════════════════════════════════"
}

# ────────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────────

log_section "Step 0: Validating Environment"

# Check required environment variables
required_vars=(
    "PANDA_DATA_PREDICT_START"
    "PANDA_DATA_PREDICT_END"
    "PANDA_DATA_USERNAME"
    "PANDA_DATA_PASSWORD"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    log_error "Missing required environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    exit 1
fi

log_info "Required environment variables are set"
log_info "Prediction period: $PANDA_DATA_PREDICT_START to $PANDA_DATA_PREDICT_END"

# ────────────────────────────────────────────────────────────────────────────
# Step 1: Build features
# ────────────────────────────────────────────────────────────────────────────

log_section "Step 1: Building Prediction Features"

if python -m scripts.predict --step features; then
    log_info "Feature building completed successfully"
    features_file="$(cd "$SCRIPT_DIR/.." && pwd)/dl-transformer-multiasset-production/data/predict_features.parquet"
    if [[ -f "$features_file" ]]; then
        log_info "Features written to: $features_file"
    else
        log_error "Features file not found at: $features_file"
        exit 1
    fi
else
    log_error "Feature building failed"
    exit 1
fi

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Run prediction
# ────────────────────────────────────────────────────────────────────────────

log_section "Step 2: Running Model Inference"

if python -m scripts.predict --step predict; then
    log_info "Model inference completed successfully"
    predict_file="$(cd "$SCRIPT_DIR/.." && pwd)/dl-transformer-multiasset-production/data/predict.parquet"
    if [[ -f "$predict_file" ]]; then
        log_info "Predictions written to: $predict_file"
    else
        log_error "Predictions file not found at: $predict_file"
        exit 1
    fi
else
    log_error "Model inference failed"
    exit 1
fi

# ────────────────────────────────────────────────────────────────────────────
# Step 3: Evaluate predictions
# ────────────────────────────────────────────────────────────────────────────

log_section "Step 3: Evaluating Predictions"

if python -m scripts.predict --step evaluate; then
    log_info "Prediction evaluation completed successfully"
    report_file="$(cd "$SCRIPT_DIR/.." && pwd)/dl-transformer-multiasset-production/data/predict_report.json"
    if [[ -f "$report_file" ]]; then
        log_info "Report written to: $report_file"
    else
        log_info "Report not generated (may not have forward returns yet)"
    fi
else
    log_error "Prediction evaluation failed"
    exit 1
fi

# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────

log_section "Pipeline Complete"

log_info "All steps completed successfully!"
log_info ""
log_info "Output files:"
log_info "  - Features: $(cd "$SCRIPT_DIR/.." && pwd)/dl-transformer-multiasset-production/data/predict_features.parquet"
log_info "  - Predictions: $(cd "$SCRIPT_DIR/.." && pwd)/dl-transformer-multiasset-production/data/predict.parquet"
log_info "  - Report: $(cd "$SCRIPT_DIR/.." && pwd)/dl-transformer-multiasset-production/data/predict_report.json (if available)"
