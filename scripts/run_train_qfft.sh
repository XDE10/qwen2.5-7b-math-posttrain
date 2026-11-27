#!/usr/bin/env bash
# One-click single-GPU full-parameter QFFT / SFT training (auto-prepares decontaminated s1K-1.1 data).
# Usage:
#   bash scripts/run_train_qfft.sh [config=train/qfft_r1.json] [extra train_sft args...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONFIG="${1:-train/qfft_r1.json}"
shift $(( $# >= 1 ? 1 : 0 ))

PY="${PYTHON:-python3}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1

DATA_FILE="${MATH_POSTTRAIN_DATA_DIR:-$REPO/data}/train/r1_qfft.jsonl"
if [[ ! -f "$DATA_FILE" ]]; then
  echo "==> Preparing decontaminated R1 QFFT dataset from public Hugging Face sources..."
  "$PY" -m math_posttrain.prepare_data --sft
fi

echo "==> Launching single-GPU training: config=$CONFIG"
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  -m math_posttrain.train_sft --config "$CONFIG" "$@"
