#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 CONFIG COUNCIL_DIR LAMBDA_U LAMBDA_D TRAINING_STATS_JSON [OUTPUT_DIR]" >&2
  exit 2
fi

CONFIG=$1
COUNCIL_DIR=$2
LAMBDA_U=$3
LAMBDA_D=$4
TRAINING_STATS=$5
OUTPUT_DIR=${6:-artifacts}

python scripts/viz_extract_weights.py \
  --config "$CONFIG" \
  --council-checkpoint-dir "$COUNCIL_DIR" \
  --lambda-u "$LAMBDA_U" \
  --lambda-d "$LAMBDA_D" \
  --standardization-path "$TRAINING_STATS" \
  --output-dir "$OUTPUT_DIR" \
  --resume

python scripts/viz_repetition.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --resume

python scripts/viz_expert_branching.py \
  --config "$CONFIG" \
  --council-checkpoint-dir "$COUNCIL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --resume

python scripts/viz_answer_gain.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --resume

python scripts/viz_removal.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --resume

python scripts/viz_make_plots.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR"
