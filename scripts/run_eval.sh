#!/usr/bin/env bash
# One-click evaluation: auto-fetch pinned benchmarks -> vLLM 32K generation -> Math-Verify scoring.
# Usage:
#   bash scripts/run_eval.sh [model=qfft_r1] [tag=qfft_r1] [suite=full] [extra generate args...]
# Examples:
#   bash scripts/run_eval.sh qfft_r1 qfft_r1 full
#   bash scripts/run_eval.sh XDE10/Qwen2.5-7B-Math-QFFT-R1 qfft_r1 competition
#   bash scripts/run_eval.sh base smoke_demo smoke --engine fake
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL="${1:-qfft_r1}"
TAG="${2:-$MODEL}"
SUITE="${3:-full}"
shift $(( $# >= 3 ? 3 : $# ))

PY="${PYTHON:-python3}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1

BENCH_DIR="${MATH_POSTTRAIN_DATA_DIR:-$REPO/data}/benchmarks"
if [[ ! -f "$BENCH_DIR/math500.jsonl" ]]; then
  echo "==> Benchmarks missing under $BENCH_DIR; downloading pinned benchmarks..."
  "$PY" -m math_posttrain.prepare_data --benchmarks
fi

echo "==> [1/2] Generating responses: model=$MODEL tag=$TAG suite=$SUITE"
"$PY" -m math_posttrain.generate --model "$MODEL" --tag "$TAG" --suite "$SUITE" "$@"

echo "==> [2/2] Scoring responses with Math-Verify..."
"$PY" -m math_posttrain.score --tags "$TAG"
