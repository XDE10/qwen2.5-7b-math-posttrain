#!/usr/bin/env bash
# One-click single-GPU LoRA-GRPO / MMR-DR-GRPO training (auto-prepares decontaminated OpenR1 prompt pool).
# Usage:
#   bash scripts/run_train_grpo.sh [config=grpo/mmr_dr50.json] [extra grpo args...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONFIG="${1:-grpo/mmr_dr50.json}"
shift $(( $# >= 1 ? 1 : 0 ))

PY="${PYTHON:-python3}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1

POOL_FILE="${MATH_POSTTRAIN_DATA_DIR:-$REPO/data}/train/rl_pool.jsonl"
if [[ ! -f "$POOL_FILE" ]]; then
  echo "==> Preparing decontaminated OpenR1 RL prompt pool from Hugging Face..."
  "$PY" -m math_posttrain.prepare_data --rl
fi

echo "==> Launching single-GPU LoRA-GRPO training: config=$CONFIG"
"$PY" -m math_posttrain.grpo --config "$CONFIG" "$@"
