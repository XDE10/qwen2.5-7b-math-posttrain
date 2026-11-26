#!/usr/bin/env bash
# Prepare pinned benchmarks and (optionally) build local SFT/QFFT training data.
# Usage:
#   bash scripts/prepare_data.sh                # download benchmarks + sources + inspect status
#   bash scripts/prepare_data.sh --benchmarks-only
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${P1_PYTHON:-python3}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1

echo "==> Step 1/2: Fetching pinned benchmarks from Hugging Face..."
if [[ "${1:-}" == "--benchmarks-only" ]]; then
  "$PY" -m math_posttrain.fetch_benchmarks
  echo "Done. Benchmarks saved under data/benchmarks_v93/."
  exit 0
fi

"$PY" -m math_posttrain.fetch_benchmarks --sources

echo "==> Step 2/2: Checking dataset build plan and status..."
"$PY" -m math_posttrain.data_build plan
"$PY" -m math_posttrain.data_build status || true
echo "To build final datasets once teacher traces are available, run:"
echo "  math-posttrain build-data build"
