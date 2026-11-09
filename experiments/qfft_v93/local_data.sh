#!/usr/bin/env bash
# v93 local data build on Linux / WSL2 (CPU): the five steps of experiments/qfft_v93/README.md section 1,
# in order, stopping at the first error. Every step resumes or rebuilds deterministically, so running this
# again is always safe. Step 4 calls the DeepSeek API only for questions that still lack a usable trace.
#
#   bash experiments/qfft_v93/local_data.sh            # from zsh too: the script itself runs in bash
#
#   P1_PYTHON      interpreter with pandas, pyarrow and tokenizers (default: python3)
#   P1_FILL_ARGS   extra options for step 4, e.g.  P1_FILL_ARGS='--max-attempts 12' bash .../local_data.sh
#   Keys           OPENCODE_GO_API_KEY / DEEPSEEK_API_KEY, or the files listed in teacher.py (Windows paths
#                  are also tried under /mnt/c and /mnt/d); without a Go key step 4 starts on the official API.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
cd "$REPO"
PY="${P1_PYTHON:-python3}"
export PYTHONUTF8=1
echo "Repository: $REPO"
echo "Python:     $PY ($("$PY" --version 2>&1))"
if ! "$PY" -c 'import pandas, pyarrow, tokenizers' 2>/dev/null; then
  echo "$PY lacks pandas, pyarrow or tokenizers. Install them (e.g. '$PY -m pip install pandas pyarrow tokenizers')"
  echo "or point P1_PYTHON at an environment that has them."
  exit 1
fi

step() {
  local title="$1"; shift
  echo; echo "===== $title"
  "$PY" "$@"
}

step "1/5 benchmarks and sources at pinned revisions" -m experiments.qfft_v93.fetch_benchmarks --sources
step "2/5 1000-question plan (v93 replacement, decontamination)" -m experiments.qfft_v93.data_build plan
step "3/5 questions still without a usable DeepSeek trace" -m experiments.qfft_v93.data_build status

echo; echo "===== 4/5 DeepSeek fill until 1000/1000 (API, resumable)"
read -r -a FILL_ARGS <<< "${P1_FILL_ARGS:-}"
set +e
"$PY" -m experiments.qfft_v93.teacher fill --fallback-official "${FILL_ARGS[@]}"
CODE=$?
set -e
if [[ $CODE == 3 ]]; then
  echo
  echo "Some questions used their whole attempt budget without a usable trace (listed above). Raise it and rerun:"
  echo "    P1_FILL_ARGS='--max-attempts 12' bash experiments/qfft_v93/local_data.sh"
  exit 3
fi
if [[ $CODE != 0 ]]; then
  echo; echo "FAILED (exit $CODE) - see the messages above. Fix the cause and rerun; finished work is kept."
  exit "$CODE"
fi

step "5/5 build r1/, ds/, ds_verified/ and the teacher report" -m experiments.qfft_v93.data_build build --ds-verified
echo
echo "Done. Read data/qfft1000_v93/teacher_report.md, then upload the data with:"
echo "    bash experiments/qfft_v93/colab/pipeline.sh d"
