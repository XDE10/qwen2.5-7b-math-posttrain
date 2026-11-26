#!/usr/bin/env bash
# Run single-GPU vLLM evaluation generation and offline Math-Verify scoring.
# Usage:
#   bash scripts/run_eval.sh <model_spec_or_path> <tag> [suite]
# Examples:
#   bash scripts/run_eval.sh base base full
#   bash scripts/run_eval.sh /path/to/checkpoint my_model competition
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_eval.sh <model_spec_or_path> <tag> [suite=full]" >&2
  exit 2
fi

MODEL="$1"
TAG="$2"
SUITE="${3:-full}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${P1_PYTHON:-python3}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1

echo "==> [1/2] Generating responses: model=$MODEL tag=$TAG suite=$SUITE"
"$PY" -m math_posttrain.generate --model "$MODEL" --tag "$TAG" --suite "$SUITE"

echo "==> [2/2] Scoring responses with Math-Verify..."
"$PY" -m math_posttrain.score --tags "$TAG" --per-problem
