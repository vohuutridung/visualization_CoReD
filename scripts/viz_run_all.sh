#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CONFIG [OUTPUT_DIR]" >&2
  echo "Set server paths in CONFIG or export CORED_COUNCIL_DIR/CORED_EXPERT_PATHS," >&2
  echo "CORED_BACKBONE, CORED_EMBEDDING_MODEL, and CORED_STANDARDIZATION_PATH." >&2
  exit 2
fi

CONFIG=$1
OUTPUT_ARGS=()
if [[ $# -eq 2 ]]; then
  OUTPUT_ARGS=(--output-dir "$2")
fi

python scripts/viz_extract_weights.py \
  --config "$CONFIG" \
  "${OUTPUT_ARGS[@]}" \
  --resume

python scripts/viz_repetition.py \
  --config "$CONFIG" \
  "${OUTPUT_ARGS[@]}" \
  --resume

python scripts/viz_expert_branching.py \
  --config "$CONFIG" \
  "${OUTPUT_ARGS[@]}" \
  --resume

python scripts/viz_answer_gain.py \
  --config "$CONFIG" \
  "${OUTPUT_ARGS[@]}" \
  --resume

python scripts/viz_removal.py \
  --config "$CONFIG" \
  "${OUTPUT_ARGS[@]}" \
  --resume

python scripts/viz_make_plots.py \
  --config "$CONFIG" \
  "${OUTPUT_ARGS[@]}"
