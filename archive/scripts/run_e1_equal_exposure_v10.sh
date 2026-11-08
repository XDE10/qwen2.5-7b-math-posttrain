#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_ROOT="${1:-/content/drive/MyDrive/runs/project1/E1_equal_exposure_v10}"
MODEL_PATH="${E0_MODEL_PATH:-/content/models/Qwen2.5-7B-Instruct--a09a35458c702b33eeacc393d103063234e8bc28}"
MODEL_IDENTITY="${E0_MODEL_IDENTITY:-/content/project1-env-evidence/e0-identity/e0_model_identity.json}"
mkdir -p "$OUTPUT_ROOT"
if [ ! -f "$OUTPUT_ROOT/environment_initial.txt" ]; then
  python3 -m pip freeze > "$OUTPUT_ROOT/environment_initial.txt"
fi
for arm in raw compact; do
  echo "Starting formal E1 ${arm}: one epoch, 1500 paired questions, expected 94 updates."
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -u -m scripts.train_project1_sft_v10 \
    --config "configs/project1_${arm}_sft_equal_exposure_v10.json" \
    --output-root "$OUTPUT_ROOT" \
    --e0-model-path "$MODEL_PATH" \
    --e0-model-identity "$MODEL_IDENTITY" \
    --swanlab-mode offline \
    2>&1 | tee -a "$OUTPUT_ROOT/${arm}.log"
done
echo "Both E1 arms finished. Keep the model outputs for dev evaluation and the fixed 637-question probe."
