#!/usr/bin/env bash
# One command = one fresh Colab VM that runs one or more v93 stages, then releases itself.
#
#   launch.sh [--gpu LIST] [--wait-min N] [--keep] [--force] [--session NAME] [--with-data] STAGE...
#   launch.sh --attach SESSION          # follow a job whose terminal was closed or lost its connection
#
#   --gpu LIST    T4, L4, G4, H100, A100 or cpu; a comma list is tried in order, e.g. --gpu H100,G4
#   --wait-min N  keep retrying the list for up to N minutes when no listed GPU can be allocated
#   --keep        do not release the VM at the end (inspect with: colab console -s SESSION)
#   --force       start a stage even if Drive says another VM is still running it
#
# Examples
#   launch.sh --gpu cpu ping:30                      # one-minute end-to-end check of CLI, Drive and logs
#   launch.sh --gpu cpu benchmarks models:base,qwen_math,qfft_official,sft_repro
#   launch.sh --gpu H100,G4 smoke-train smoke-eval smoke-grpo
#   launch.sh --gpu H100,G4 --wait-min 120 train:qfft_ds eval:run:qfft_ds_v93:full
#   launch.sh --gpu cpu score analyze:base,sft_ds_v93,qfft_ds_v93:grpo_qfft_ds_v93
#
# Needs Linux or WSL2 with GNU coreutils (the Colab CLI does not support native Windows), git, python3, tar
# and `uv tool install google-colab-cli` (the CLI needs Python >= 3.12; uv fetches it). The first CLI command
# prints a Google sign-in URL: open it, approve, paste the code back. The first Drive mount prints a second
# link (Drive access for Colab VMs): approve it and press Enter; later VMs mount Drive without prompting.
#
# The job runs inside the VM's kernel. Closing this terminal or Ctrl+C only detaches (exit code 75): the
# stages keep running and the VM releases itself when they finish (abort with: colab stop -s SESSION). A
# dropped connection switches to polling the VM's log. All state lives on Google Drive, so re-running a
# command later resumes from the last checkpoint; a stage still running on another VM is refused.
set -euo pipefail

GPU_LIST="G4"; KEEP=0; FORCE=0; SESSION=""; WITH_DATA=0; WAIT_MIN=0; ATTACH=""
STAGES=()
usage() { sed -n '2,27p' "$0"; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_LIST="$2"; shift 2 ;;
    --wait-min) WAIT_MIN="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --force) FORCE=1; shift ;;
    --session) SESSION="$2"; shift 2 ;;
    --with-data) WITH_DATA=1; shift ;;
    --attach) ATTACH="$2"; shift 2 ;;
    -h|--help) usage ;;
    -*) echo "Unknown option: $1"; usage ;;
    *) STAGES+=("$1"); shift ;;
  esac
done
[[ -n $ATTACH || ${#STAGES[@]} -gt 0 ]] || usage
command -v colab >/dev/null || { echo "colab CLI not found: uv tool install google-colab-cli (Linux/WSL2)"; exit 1; }
[[ $WAIT_MIN =~ ^[0-9]+$ ]] || { echo "--wait-min needs whole minutes"; exit 1; }

IFS=',' read -r -a GPUS <<< "${GPU_LIST^^}"
for g in "${GPUS[@]}"; do
  case "$g" in
    T4|L4|G4|H100|A100|CPU) ;;
    # The CLI silently turns unknown names into an A100 request, so reject them here.
    *) echo "Unsupported --gpu value '$g' (use T4, L4, G4, H100, A100 or cpu)"; exit 1 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"   # works from any directory (e.g. a Windows wrapper)
cd "$REPO"
WORK="$(mktemp -d)"
LOGDIR="$REPO/reports/colab_logs"; mkdir -p "$LOGDIR"
CHUNK=$((8 * 1024 * 1024))   # each `colab upload` is one base64 HTTP request; keep requests small
POLL_S="${P1_POLL_S:-60}"    # seconds between log polls while following a detached job
STOP_ON_EXIT=0               # 1 only while this script owns the VM: before the job starts or after it ended
DETACHED=0
RESULT=""
LOG=""

finish() {
  local code=$?
  if [[ $STOP_ON_EXIT == 1 && $KEEP == 0 && -n $SESSION ]]; then
    colab stop -s "$SESSION" >/dev/null 2>&1 || true
    echo "VM $SESSION released."
  fi
  rm -rf "$WORK"
  exit "$code"
}
trap finish EXIT
trap 'DETACHED=1' INT TERM HUP

hhmm() { python3 -c 'import sys, time; print(time.strftime("%H:%M", time.localtime(int(sys.argv[1]))))' "$1"; }

# Follow /content/p1/session_tail.json (rewritten by remote.py every ~15 s) until P1_RESULT appears, the VM
# disappears or the user detaches. The Jupyter contents API answers while the kernel is busy.
follow() {
  local seen=0 misses=0 out line
  echo "Following $SESSION through its session log (Ctrl+C detaches; the job keeps running)."
  while [[ $DETACHED == 0 ]]; do
    if colab download -s "$SESSION" /content/p1/session_tail.json "$WORK/tail.json" >/dev/null 2>&1; then
      misses=0
      out="$(python3 - "$WORK/tail.json" "$seen" <<'PY'
import json, sys
tail = json.load(open(sys.argv[1], encoding='utf-8'))
seen, first = int(sys.argv[2]), tail['first_index']
if seen + 1 < first:
    print(f'... {first - seen - 1} earlier lines not shown (full log: /content/p1/session.log on the VM)')
for i, text in enumerate(tail['lines'], start=first):
    if i > seen:
        print(text)
print(f"@@P1SEEN={tail['lines_total']}")
if tail.get('result'):
    print('@@P1RESULT=' + tail['result'])
PY
)" || out=""
      while IFS= read -r line; do
        case "$line" in
          @@P1SEEN=*) seen="${line#@@P1SEEN=}" ;;
          @@P1RESULT=*) RESULT="${line#@@P1RESULT=}" ;;
          '') ;;
          *) printf '%s\n' "$line" | tee -a "$LOG" ;;
        esac
      done <<< "$out"
      [[ -n $RESULT ]] && return 0
    else
      misses=$((misses + 1))
      if (( misses >= 5 )); then
        echo "Session $SESSION no longer answers (released, reclaimed or never created)."
        return 2
      fi
    fi
    sleep "$POLL_S" || true
  done
  return 130
}

report() {  # interpret RESULT; once the job has ended the VM is released on exit (unless --keep)
  if [[ -n $RESULT ]]; then
    STOP_ON_EXIT=1
    if [[ $RESULT == P1_RESULT=OK* ]]; then
      echo "SUCCESS: ${STAGES[*]:-attached job}  (log: $LOG)"
      return 0
    fi
    echo "FAILED: ${RESULT#P1_RESULT=FAILED }"
    echo "Log: $LOG and Drive runs/project1/v93/logs/. Re-running the same command resumes from Drive."
    return 1
  fi
  if [[ $DETACHED == 1 ]]; then
    echo
    echo "Detached from $SESSION. The stages keep running on the VM, which releases itself when they end."
    echo "  follow again:  experiments/qfft_v93/colab/launch.sh --attach $SESSION"
    echo "  abort:         colab stop -s $SESSION"
    echo "  progress:      experiments/qfft_v93/colab/launch.sh --gpu cpu status   (logs: Drive runs/project1/v93/logs/)"
    return 75   # not success: pipeline.sh must not start its next launch while this job is still running
  fi
  echo "No result received from $SESSION. Check: launch.sh --gpu cpu status (and Drive runs/project1/v93/logs/)."
  return 1
}

if [[ -n $ATTACH ]]; then
  SESSION="$ATTACH"; LOG="$LOGDIR/$SESSION.log"
  follow || true
  report
  exit
fi

# ------------------------------------------------------------------ bundles
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

upload() {  # upload LOCAL REMOTE, in parts when larger than CHUNK (reassembled and hash-checked on the VM)
  local src="$1" dst="$2" size n=0 part
  size=$(wc -c < "$src")
  if (( size <= CHUNK )); then
    colab upload -s "$SESSION" "$src" "$dst"
    return
  fi
  mkdir -p "$WORK/parts"; rm -f "$WORK"/parts/*
  split -b "$CHUNK" -d -a 3 "$src" "$WORK/parts/p"
  for part in "$WORK"/parts/p*; do
    colab upload -s "$SESSION" "$part" "$dst.part${part##*/p}"
    n=$((n + 1))
  done
  printf '{"parts": %d, "bytes": %d, "sha256": "%s"}\n' "$n" "$size" "$(sha256sum "$src" | cut -c1-64)" \
    > "$WORK/parts.json"
  colab upload -s "$SESSION" "$WORK/parts.json" "$dst.parts.json"
  echo "Uploaded $(basename "$src") in $n parts ($((size / 1048576)) MiB)."
}

# ------------------------------------------------------------------ allocate
SESSION="${SESSION:-p1-$(date +%m%d-%H%M%S)}"
LOG="$LOGDIR/$SESSION.log"
echo "Session $SESSION | GPU ${GPUS[*]} | code $CODE_SHA (git $GIT_HEAD) | stages: ${STAGES[*]}"
colab usage || true

DEADLINE=$(( $(date +%s) + WAIT_MIN * 60 ))
GPU_USED=""
while [[ -z $GPU_USED ]]; do
  for g in "${GPUS[@]}"; do
    args=(new -s "$SESSION" --high-mem)
    [[ $g != CPU ]] && args+=(--gpu "$g")
    if colab "${args[@]}"; then GPU_USED="$g"; break; fi
    echo "Could not allocate $g."
  done
  [[ -n $GPU_USED ]] && break
  if (( $(date +%s) >= DEADLINE )) || [[ $DETACHED == 1 ]]; then
    echo "No VM from [${GPUS[*]}] could be allocated; nothing was started."
    exit 1
  fi
  echo "Retrying in 5 minutes (until $(hhmm "$DEADLINE"))."
  sleep 300 || true
done
STOP_ON_EXIT=1
echo "Allocated $GPU_USED as $SESSION."

colab drivemount -s "$SESSION"
CHECK="$(echo 'import os; print("P1_DRIVE_" + ("OK" if os.path.isdir("/content/drive/MyDrive") else "MISSING"))' \
  | colab exec -s "$SESSION" --timeout 120 2>&1 || true)"
if ! grep -q 'P1_DRIVE_OK' <<< "$CHECK"; then
  echo "Google Drive is not mounted on $SESSION. Run this command in a terminal, open the Drive link the CLI"
  echo "prints, approve access and press Enter (needed once per Google account)."
  exit 1
fi
upload "$BUNDLE" /content/p1_bundle.tgz
if [[ $WITH_DATA == 1 ]]; then upload "$WORK/p1_data.tgz" /content/p1_data.tgz; fi

ENTRY="$WORK/entry.py"
RELEASE=$(( KEEP == 0 ? 1 : 0 ))
python3 - "$ENTRY" "$CODE_SHA" "$GIT_HEAD" "$RELEASE" "$SESSION" "$GPU_USED" "$FORCE" "${STAGES[@]}" <<'PY'
import json, pathlib, sys
entry, sha, head, release, session, gpu, force, *stages = sys.argv[1:]
args = {'code_sha': sha, 'git_head': head, 'auto_release': release == '1', 'stages': stages,
        'bundle': '/content/p1_bundle.tgz', 'session': session, 'gpu': gpu, 'force': force == '1'}
remote = pathlib.Path('experiments/qfft_v93/colab/remote.py').read_text(encoding='utf-8')
# Python literal (True/False), not JSON (true/false): the kernel executes this file as Python.
pathlib.Path(entry).write_text('P1_ARGS = ' + repr(args) + '\n' + remote, encoding='utf-8')
print('entry args:', json.dumps(args))
PY

# ------------------------------------------------------------------ run
# From here on the job owns the VM; this script releases it only after seeing the job's P1_RESULT line.
STOP_ON_EXIT=0
: > "$LOG"
# --timeout bounds this client only (24 h, the Colab VM limit); it never interrupts the kernel.
colab exec -s "$SESSION" --timeout 86400 -f "$ENTRY" > >(tee -a "$LOG") 2>&1 &
EXEC_PID=$!
while kill -0 "$EXEC_PID" 2>/dev/null; do
  RESULT="$(grep -a '^P1_RESULT=' "$LOG" | tail -n 1 || true)"
  [[ -n $RESULT || $DETACHED == 1 ]] && break
  sleep 5 || true
done
[[ -n $RESULT ]] || RESULT="$(grep -a '^P1_RESULT=' "$LOG" | tail -n 1 || true)"
kill "$EXEC_PID" 2>/dev/null || true   # after P1_RESULT the VM only waits out remote.py's release grace
wait "$EXEC_PID" 2>/dev/null || true
if [[ -z $RESULT && $DETACHED == 0 ]]; then
  echo "Lost the live connection to $SESSION before the job finished; switching to log polling."
  follow || true
fi
report
