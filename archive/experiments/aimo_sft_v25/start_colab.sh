#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 experiments/aimo_sft_v25/install_missing.py
python3 -u experiments/aimo_sft_v25/run.py train
