#!/usr/bin/env bash
# One command = one fresh Colab VM that runs one or more v93 stages, then releases itself.
#
#   experiments/qfft_v93/colab/launch.sh [--gpu G4|H100|A100|L4|cpu] [--keep] [--session NAME] [--with-data] STAGE...
#
# Examples
#   launch.sh --gpu cpu --with-data sync-data models:base,qwen_math,qfft_official,sft_repro
#   launch.sh --gpu G4 smoke-train smoke-eval smoke-grpo
#   launch.sh --gpu G4 train:qfft_ds eval:run:qfft_ds_v93:full
#   launch.sh --gpu H100 grpo:grpo_qfft_ds eval:run:grpo_qfft_ds_v93/merged:full
#   launch.sh --gpu cpu score analyze:base,sft_ds_v93,qfft_ds_v93:grpo_qfft_ds_v93
#
# Requirements on this machine: Linux, macOS or WSL2 (the Colab CLI does not support native Windows),
# `uv tool install google-colab-cli`, git, python3, tar. All state lives on Google Drive, so a stage can
# be re-launched on a new VM at any time and resumes from its last checkpoint / saved generation.
set -euo pipefail

GPU="G4"; KEEP=0; SESSION=""; WITH_DATA=0
STAGES=()
usage() { sed -n '2,16p' "$0"; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --session) SESSION="$2"; shift 2 ;;
    --with-data) WITH_DATA=1; shift ;;
    -h|--help) usage ;;
    *) STAGES+=("$1"); shift ;;
  esac
done
[[ ${#STAGES[@]} -gt 0 ]] || usage
command -v colab >/dev/null || { echo "colab CLI not found: uv tool install google-colab-cli (Linux/macOS/WSL2)"; exit 1; }

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
WORK="$(mktemp -d)"
BUNDLE="$WORK/p1_bundle.tgz"
# NUL-separated so non-ASCII file names (tools/qfft/*.cmd) survive; untracked-but-not-ignored files included.
git ls-files -z -co --exclude-standard -- experiments src scripts configs tools pyproject.toml > "$WORK/files.txt"
tar --null -czf "$BUNDLE" -T "$WORK/files.txt"
CODE_SHA="$(sha256sum "$BUNDLE" | cut -c1-12)"
GIT_HEAD="$(git rev-parse --short=12 HEAD)$(git diff --quiet HEAD -- experiments src scripts || echo '+dirty')"
if [[ $WITH_DATA == 1 ]]; then
  for d in data/qfft1000_v93 data/benchmarks_v93; do
    [[ -d $d ]] || { echo "Missing $d: run data_build / fetch_benchmarks locally first"; exit 1; }
  done
  tar -czf "$WORK/p1_data.tgz" --exclude='data/benchmarks_v93/_sources' data/qfft1000_v93 data/benchmarks_v93
fi

SESSION="${SESSION:-p1-$(date +%m%d-%H%M%S)}"
LOGDIR="$REPO/reports/colab_logs"; mkdir -p "$LOGDIR"; LOG="$LOGDIR/$SESSION.log"
echo "Session $SESSION | GPU $GPU | code $CODE_SHA (git $GIT_HEAD) | stages: ${STAGES[*]}"
colab usage || true

NEW=(new -s "$SESSION" --high-mem)
[[ "${GPU,,}" != "cpu" ]] && NEW+=(--gpu "$GPU")
colab "${NEW[@]}"
finish() {
  if [[ $KEEP == 0 ]]; then
    colab stop -s "$SESSION" >/dev/null 2>&1 || true
    echo "VM $SESSION released."
  else
    echo "VM kept: colab console -s $SESSION   |   colab stop -s $SESSION"
  fi
  rm -rf "$WORK"
}
trap finish EXIT

colab drivemount -s "$SESSION"
colab upload -s "$SESSION" "$BUNDLE" /content/p1_bundle.tgz
if [[ $WITH_DATA == 1 ]]; then colab upload -s "$SESSION" "$WORK/p1_data.tgz" /content/p1_data.tgz; fi

ENTRY="$WORK/entry.py"
RELEASE=$(( KEEP == 0 ? 1 : 0 ))
python3 - "$ENTRY" "$CODE_SHA" "$GIT_HEAD" "$RELEASE" "${STAGES[@]}" <<'PY'
import json, pathlib, sys
entry, sha, head, release, *stages = sys.argv[1:]
args = {'code_sha': sha, 'git_head': head, 'auto_release': release == '1', 'stages': stages,
        'bundle': '/content/p1_bundle.tgz'}
remote = pathlib.Path('experiments/qfft_v93/colab/remote.py').read_text(encoding='utf-8')
# Python literal (True/False), not JSON (true/false): the kernel executes this file as Python.
pathlib.Path(entry).write_text('P1_ARGS = ' + repr(args) + '\n' + remote, encoding='utf-8')
print('entry args:', json.dumps(args))
PY

# --timeout is an idle limit in current CLIs and a total limit in old ones; 24 h is safe for both.
set +e
colab exec -s "$SESSION" --timeout 86400 -f "$ENTRY" 2>&1 | tee "$LOG"
set -e
if grep -q '^P1_RESULT=OK' "$LOG"; then
  echo "SUCCESS: ${STAGES[*]}  (log: $LOG)"
else
  echo "FAILED or interrupted: see $LOG. Re-running the same command resumes from Drive."
  exit 1
fi
