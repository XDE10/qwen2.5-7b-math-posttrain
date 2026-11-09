#!/usr/bin/env bash
# The v93 experiment matrix as resumable phases, in priority order. Each phase is one or more launch.sh
# calls; every stage skips work that is already complete on Drive, so re-running a phase is always safe.
#
#   experiments/qfft_v93/colab/pipeline.sh <phase> [launch.sh options, e.g. --gpu G4 --wait-min 120]
#
#   0  CPU VM: fetch the pinned benchmarks and models straight to Drive (no local data needed)
#   1  GPU smoke at worst-case lengths: 2 full-FT steps of 32K rows, 24 generations, 2 GRPO steps of
#      8192-token completions (s/step, peak GPU and host memory)
#   2  anchors: E0, QFFT authors' checkpoint, public SFT twin, Qwen2.5-Math reference
#   d  CPU VM: upload the locally built data/qfft1000_v93 (+ benchmarks) once data_build has finished
#   3  DeepSeek pair: QFFT and SFT on the DS-V4.1-Flash traces (each trained, then evaluated on its VM)
#   4  RL: LoRA-GRPO from the QFFT/DS checkpoint, merge, evaluate
#   5  R1 pair: QFFT and SFT on the s1K-1.1 R1 traces (controlled teacher comparison)
#   6  optional: GRPO from the SFT/DS checkpoint with the same budget (init comparison)
#   7  CPU VM: re-score everything, results.md and figures
#
# GPU preference when --gpu is not given: training/RL phases try H100 then G4 (faster; the smoke also
# checks the tighter 80 GB), evaluation-only phase 2 tries G4 then H100 (more KV cache). A --gpu option
# overrides it; CPU phases always use a CPU VM.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L="$HERE/launch.sh"
PHASE="${1:-}"; shift || true
OPTS=("$@")
TRAIN=(--gpu "H100,G4" "${OPTS[@]}")   # a later --gpu in OPTS wins inside launch.sh
EVAL=(--gpu "G4,H100" "${OPTS[@]}")
CPU=("${OPTS[@]}" --gpu cpu)
TAGS=base,qwen_math,qfft_official,sft_repro,sft_ds_v93,qfft_ds_v93,grpo_qfft_ds_v93_merged,sft_r1_v93,qfft_r1_v93

case "$PHASE" in
  0) bash "$L" "${CPU[@]}" benchmarks models:base,qwen_math,qfft_official,sft_repro ;;
  d) bash "$L" "${CPU[@]}" --with-data sync-data ;;
  1) bash "$L" "${TRAIN[@]}" smoke-train smoke-eval smoke-grpo ;;
  2) bash "$L" "${EVAL[@]}" eval:base:full eval:qfft_official:full eval:sft_repro:full eval:qwen_math:full ;;
  3) bash "$L" "${TRAIN[@]}" train:qfft_ds eval:run:qfft_ds_v93:full
     bash "$L" "${TRAIN[@]}" train:sft_ds  eval:run:sft_ds_v93:full ;;
  4) bash "$L" "${TRAIN[@]}" grpo:grpo_qfft_ds eval:run:grpo_qfft_ds_v93/merged:full ;;
  5) bash "$L" "${TRAIN[@]}" train:qfft_r1 eval:run:qfft_r1_v93:full
     bash "$L" "${TRAIN[@]}" train:sft_r1  eval:run:sft_r1_v93:full ;;
  6) bash "$L" "${TRAIN[@]}" grpo:grpo_sft_ds eval:run:grpo_sft_ds_v93/merged:full ;;
  7) bash "$L" "${CPU[@]}" score status "analyze:$TAGS:grpo_qfft_ds_v93" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
