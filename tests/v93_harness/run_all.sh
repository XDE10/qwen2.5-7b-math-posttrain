#!/usr/bin/env bash
# Mock test harness for the v93 tooling: Colab launcher and pipeline, VM stages, rclone Drive transfers and
# checkpoint mirrors, GitHub reports, the DeepSeek teacher client, data_build and the local helper scripts.
# Nothing real is contacted: colab, rclone, mountpoint and apt-get are stand-ins (mockbin/), GitHub is a
# local bare repository and the DeepSeek API is a local HTTP server.
#
#   bash tests/v93_harness/run_all.sh                       # every suite (scenarios alone take ~25 minutes)
#   bash tests/v93_harness/run_all.sh drive teacher data    # some suites, see SUITES below
#   P1_SCENARIOS="A R2 RP SD" bash tests/v93_harness/run_all.sh scenarios
#
# Everything runs on a temporary snapshot of this working tree (tracked and untracked files that git does
# not ignore), so the checkout's data/ and reports/ are never touched. Needs bash, git, tar, python3 >= 3.10.
# The teacher, pinned and data suites also need an interpreter with `tokenizers` and `math-verify`
# (P1_PYTHON, default python3) and the Qwen2.5-7B-Instruct tokenizer.json: found under data/ once
# local_data.sh has run, or set P1_HARNESS_TOKENIZER. Without them those suites are skipped, not failed.
set -uo pipefail
H_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$H_SRC" rev-parse --show-toplevel)" || exit 1
SUITES=(scenarios drive smoke teacher pinned data setup local)
[[ $# -gt 0 ]] && SUITES=("$@")
PY="${P1_PYTHON:-python3}"
TOKENIZER_SHA=c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539

WORK="$(mktemp -d "${TMPDIR:-/tmp}/p1_harness.XXXXXX")" || { echo "FAIL cannot create a temporary directory"; exit 1; }
trap 'rm -rf "$WORK"' EXIT
SNAP="$WORK/repo"
mkdir -p "$SNAP"
git -C "$REPO" ls-files -z -co --exclude-standard | tar -C "$REPO" --null --ignore-failed-read -T - -cf - 2>/dev/null \
  | tar -C "$SNAP" -xf - || { echo "FAIL could not copy the working tree to $SNAP"; exit 1; }
git -C "$SNAP" init -q && git -C "$SNAP" add -A &&
  git -C "$SNAP" -c user.name=harness -c user.email=harness@localhost -c commit.gpgsign=false \
    -c core.hooksPath=/dev/null commit -q -m "harness snapshot of $(git -C "$REPO" rev-parse --short HEAD)" ||
  { echo "FAIL could not create the git snapshot in $SNAP"; exit 1; }
[[ -f $SNAP/tests/v93_harness/run_all.sh ]] || { echo "FAIL the snapshot is incomplete"; exit 1; }
touch "$SNAP/.p1_harness_snapshot"   # scenarios.sh refuses to run anywhere else (it deletes data/ folders)

TOK="${P1_HARNESS_TOKENIZER:-}"
if [[ -z $TOK ]]; then
  for candidate in "$REPO/data/tokenizers/Qwen2.5-7B-Instruct/tokenizer.json" \
                   "$REPO/data/benchmarks_v93/_sources/Qwen2.5-7B-Instruct-tokenizer.json"; do
    if [[ -f $candidate && $(sha256sum "$candidate" | cut -c1-64) == "$TOKENIZER_SHA" ]]; then TOK="$candidate"; break; fi
  done
fi
export P1_HARNESS_REPO="$SNAP" P1_HARNESS_DIR="$SNAP/tests/v93_harness" P1_HARNESS_WORK="$WORK" \
       P1_HARNESS_TOKENIZER="$TOK" P1_HARNESS_PYTHON="$PY"
H="$P1_HARNESS_DIR"
needs_ml() {
  if [[ -z $TOK ]]; then echo "SKIP $1: no Qwen2.5-7B-Instruct tokenizer.json (set P1_HARNESS_TOKENIZER)"; return 1; fi
  if ! "$PY" -c 'import tokenizers, math_verify' 2>/dev/null; then
    echo "SKIP $1: $PY lacks tokenizers or math-verify (set P1_PYTHON)"; return 1
  fi
}

declare -A PASSED FAILED
for suite in "${SUITES[@]}"; do
  echo; echo "########## $suite"
  log="$WORK/$suite.out"
  case "$suite" in
    scenarios) bash "$H/scenarios.sh" ${P1_SCENARIOS:-A B C D DR E F G H I J K Ldata M N P R1 R2 R3 RP SD} 2>&1 | tee "$log" ;;
    drive) python3 "$H/drive_units.py" 2>&1 | tee "$log" ;;
    smoke) python3 "$H/smoke_report_check.py" 2>&1 | tee "$log" ;;
    teacher) needs_ml teacher && "$PY" "$H/teacher_scenarios.py" 2>&1 | tee "$log" ;;
    pinned) needs_ml pinned && "$PY" "$H/pinned_inputs_check.py" 2>&1 | tee "$log" ;;
    data) needs_ml data && "$PY" "$H/data_build_check.py" 2>&1 | tee "$log" ;;
    setup) bash "$H/setup_check.sh" 2>&1 | tee "$log" ;;
    local) bash "$H/check_local_check.sh" 2>&1 | tee "$log" ;;
    *) echo "unknown suite $suite (known: scenarios drive smoke teacher pinned data setup local)"; exit 2 ;;
  esac
  PASSED[$suite]=$(grep -c '^PASS ' "$log" 2>/dev/null)
  FAILED[$suite]=$(grep -c '^FAIL ' "$log" 2>/dev/null)
  # a suite that crashed before printing any result counts as a failure
  if [[ -f $log && ${PASSED[$suite]:-0} == 0 && ${FAILED[$suite]:-0} == 0 ]]; then FAILED[$suite]=1; fi
done

echo; echo "########## summary"
total_fail=0
for suite in "${SUITES[@]}"; do
  if [[ ! -f $WORK/$suite.out ]]; then printf '  %-10s skipped\n' "$suite"; continue; fi
  printf '  %-10s %3s passed, %s failed\n' "$suite" "${PASSED[$suite]:-0}" "${FAILED[$suite]:-0}"
  total_fail=$((total_fail + ${FAILED[$suite]:-0}))
done
[[ $total_fail == 0 ]]
