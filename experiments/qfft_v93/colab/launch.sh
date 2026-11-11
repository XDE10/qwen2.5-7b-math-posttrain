#!/usr/bin/env bash
# One command = one fresh Colab VM that runs one or more v93 stages, then releases itself.
#
#   launch.sh [--gpu LIST] [--wait-min N] [--keep] [--force] [--session NAME] [--drive rclone|colab]
#             [--no-report] [--with-data [--data-root DIR]] STAGE...
#   launch.sh --attach SESSION          # follow a job whose terminal was closed or lost its connection
#
#   --gpu LIST    T4, L4, G4, H100, A100 or cpu; a comma list is tried in order, e.g. --gpu G4,H100.
#                 Each GPU is requested with --high-mem first, then in the standard shape (training
#                 stages refuse a VM with less than 128 GiB of host RAM themselves)
#   --wait-min N  keep retrying the list for up to N minutes when no listed GPU can be allocated
#   --keep        do not release the VM at the end (inspect with: colab console -s SESSION)
#   --force       start a stage even if Drive says another VM is still running it
#   --data-root DIR  with --with-data: read data/qfft1000_v93 and data/benchmarks_v93 under DIR instead of
#                 this checkout (default: $P1_DATA_ROOT, else this repository)
#   --drive MODE  rclone (default, $P1_DRIVE_MODE): upload $P1_RCLONE_CONF (default ~/.config/rclone/rclone.conf)
#                 and mount remote $P1_RCLONE_REMOTE (default mydrive:) at /content/mydrive, no prompts;
#                 colab: Colab's own Drive mount at /content/drive/MyDrive (asks for approval once)
#   --no-report   do not push reports to GitHub. By default, when $P1_GH_CREDENTIALS (default
#                 ~/.config/p1/gh_credentials, one line https://USER:TOKEN@github.com) exists, the VM pushes
#                 small text reports to branch $P1_REPORT_BRANCH (default colab-reports) every
#                 $P1_REPORT_MINUTES (default 30) minutes and at the end; the token stays out of logs and files
#
# Examples
#   launch.sh --gpu cpu ping:30                      # one-minute end-to-end check of CLI, Drive and logs
#   launch.sh --gpu cpu benchmarks models:base,qwen_math,qfft_official,sft_repro
#   launch.sh --gpu G4,H100 smoke-train smoke-eval smoke-grpo
#   launch.sh --gpu G4,H100 --wait-min 120 train:qfft_ds eval:run:qfft_ds_v93:full
#   launch.sh --gpu cpu --with-data --data-root "/mnt/d/Users/Mr.Z/Desktop/项目一_7B数学后训练" sync-data
#   launch.sh --gpu cpu score analyze:base,sft_ds_v93,qfft_ds_v93:grpo_qfft_ds_v93
#
# Needs Linux or WSL2 with GNU coreutils (the Colab CLI does not support native Windows), git, python3, tar
# and `uv tool install google-colab-cli` (the CLI needs Python >= 3.12; uv fetches it). The first CLI command
# prints a Google sign-in URL: open it, approve, paste the code back. This is a bash script; from zsh run it
# as `bash .../launch.sh ...` or `./.../launch.sh ...`. Shells inside the VM (colab console) are bash.
#
# The job runs inside the VM's kernel. Closing this terminal or Ctrl+C only detaches (exit code 75): the
# stages keep running and the VM releases itself when they finish (abort with: colab stop -s SESSION). A
# dropped connection switches to polling the VM's log. All state lives on Google Drive, so re-running a
# command later resumes from the last checkpoint; a stage still running on another VM is refused.
set -euo pipefail

GPU_LIST="G4"; KEEP=0; FORCE=0; SESSION=""; WITH_DATA=0; WAIT_MIN=0; ATTACH=""; DATA_ROOT="${P1_DATA_ROOT:-}"
DRIVE_MODE="${P1_DRIVE_MODE:-rclone}"; REPORT=1
RCLONE_CONF="${P1_RCLONE_CONF:-$HOME/.config/rclone/rclone.conf}"; RCLONE_REMOTE="${P1_RCLONE_REMOTE:-mydrive:}"
GH_CREDENTIALS="${P1_GH_CREDENTIALS:-$HOME/.config/p1/gh_credentials}"
STAGES=()
usage() { sed -n '2,32p' "$0"; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_LIST="$2"; shift 2 ;;
    --wait-min) WAIT_MIN="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --force) FORCE=1; shift ;;
    --session) SESSION="$2"; shift 2 ;;
    --with-data) WITH_DATA=1; shift ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --drive) DRIVE_MODE="$2"; shift 2 ;;
    --no-report) REPORT=0; shift ;;
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

# ------------------------------------------------------------------ Drive and report settings (before any VM)
case "$DRIVE_MODE" in
  rclone)
    DRIVE_ROOT="/content/mydrive"
    [[ $RCLONE_REMOTE == *:* ]] || { echo "P1_RCLONE_REMOTE must look like 'mydrive:' (got '$RCLONE_REMOTE')"; exit 1; }
    if [[ -z $ATTACH ]]; then
      if [[ ! -f $RCLONE_CONF ]]; then
        echo "rclone config not found: $RCLONE_CONF"
        echo "Set P1_RCLONE_CONF to the rclone.conf that defines [${RCLONE_REMOTE%%:*}], or use --drive colab."
        exit 1
      fi
      grep -q "^\[${RCLONE_REMOTE%%:*}\]" "$RCLONE_CONF" || {
        echo "$RCLONE_CONF has no [${RCLONE_REMOTE%%:*}] section (remote $RCLONE_REMOTE)"; exit 1; }
    fi
    ;;
  colab) DRIVE_ROOT="/content/drive/MyDrive"; RCLONE_REMOTE="" ;;
  *) echo "--drive must be rclone or colab (got '$DRIVE_MODE')"; exit 1 ;;
esac
REPORT_REPO=""
if [[ $REPORT == 1 && -z $ATTACH ]]; then
  REPORT_REPO="${P1_REPORT_REPO:-$(python3 "$HERE/report_push.py" "$(git remote get-url origin 2>/dev/null || true)")}"
  if [[ -z $REPORT_REPO ]]; then
    echo "Reports: origin is not a GitHub remote; results stay on Drive (use --no-report to hide this)."
  elif [[ $REPORT_REPO == https://* && ! -f $GH_CREDENTIALS ]]; then
    echo "Reports: $GH_CREDENTIALS not found, so nothing is pushed to GitHub (see experiments/qfft_v93/README.md)."
    REPORT_REPO=""
  else
    echo "Reports: pushed to $REPORT_REPO branch ${P1_REPORT_BRANCH:-colab-reports}"
  fi
fi

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
  DATA_ROOT="${DATA_ROOT:-$REPO}"
  for f in data/qfft1000_v93/manifest.json data/benchmarks_v93/manifest.json; do
    if [[ ! -f "$DATA_ROOT/$f" ]]; then
      echo "Missing $DATA_ROOT/$f: finish data_build build / fetch_benchmarks first."
      [[ $DATA_ROOT == "$REPO" ]] && echo "If the data was built in another checkout (e.g. the Windows working copy), pass" \
        "--data-root \"/mnt/d/Users/Mr.Z/Desktop/项目一_7B数学后训练\" or set P1_DATA_ROOT."
      exit 1
    fi
  done
  echo "Data from $DATA_ROOT"
  tar -C "$DATA_ROOT" -czf "$WORK/p1_data.tgz" --exclude='data/benchmarks_v93/_sources' \
    data/qfft1000_v93 data/benchmarks_v93
fi

colab_upload_retry() {
  local src="$1" dst="$2" retries=5 delay=3 attempt=1
  while true; do
    if colab upload -s "$SESSION" "$src" "$dst"; then
      return 0
    fi
    if (( attempt >= retries )); then
      echo "ERROR: Failed to upload $src to $dst after $retries attempts." >&2
      return 1
    fi
    echo "WARNING: Upload of $(basename "$src") failed (attempt $attempt/$retries); retrying in ${delay}s..." >&2
    sleep "$delay"
    delay=$(( delay * 2 ))
    attempt=$(( attempt + 1 ))
  done
}

upload() {  # upload LOCAL REMOTE, in parts when larger than CHUNK (reassembled and hash-checked on the VM)
  local src="$1" dst="$2" size n=0 part
  size=$(wc -c < "$src")
  if (( size <= CHUNK )); then
    colab_upload_retry "$src" "$dst"
    return
  fi
  mkdir -p "$WORK/parts"; rm -f "$WORK"/parts/*
  split -b "$CHUNK" -d -a 3 "$src" "$WORK/parts/p"
  for part in "$WORK"/parts/p*; do
    colab_upload_retry "$part" "$dst.part${part##*/p}"
    n=$((n + 1))
  done
  printf '{"parts": %d, "bytes": %d, "sha256": "%s"}\n' "$n" "$size" "$(sha256sum "$src" | cut -c1-64)" \
    > "$WORK/parts.json"
  colab_upload_retry "$WORK/parts.json" "$dst.parts.json"
  echo "Uploaded $(basename "$src") in $n parts ($((size / 1048576)) MiB)."
}

# ------------------------------------------------------------------ allocate
SESSION="${SESSION:-p1-$(date +%m%d-%H%M%S)}"
LOG="$LOGDIR/$SESSION.log"
echo "Session $SESSION | GPU ${GPUS[*]} | code $CODE_SHA (git $GIT_HEAD) | stages: ${STAGES[*]}"
colab usage || true

DEADLINE=$(( $(date +%s) + WAIT_MIN * 60 ))
GPU_USED=""; SHAPE=""
while [[ -z $GPU_USED ]]; do
  for g in "${GPUS[@]}"; do
    args=(new -s "$SESSION")
    [[ $g != CPU ]] && args+=(--gpu "$g")
    if colab "${args[@]}" --high-mem; then GPU_USED="$g"; SHAPE="high-mem"; break; fi
    # The CLI sends shape=hm for G4/H100 too; if that shape is refused, the standard one may still exist.
    if colab "${args[@]}"; then GPU_USED="$g"; SHAPE="standard"; break; fi
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
echo "Allocated $GPU_USED ($SHAPE shape) as $SESSION."

if [[ $DRIVE_MODE == colab ]]; then
  colab drivemount -s "$SESSION"
  CHECK="$(echo 'import os; print("P1_DRIVE_" + ("OK" if os.path.isdir("/content/drive/MyDrive") else "MISSING"))' \
    | colab exec -s "$SESSION" --timeout 120 2>&1 || true)"
  if ! grep -q 'P1_DRIVE_OK' <<< "$CHECK"; then
    echo "Google Drive is not mounted on $SESSION. Run this command in a terminal, open the Drive link the CLI"
    echo "prints, approve access and press Enter (needed once per Google account)."
    exit 1
  fi
else
  # The same steps as mounting by hand in `colab console` (rclone.conf -> ~/.config/rclone, apt-get rclone
  # fuse3 only if missing, rclone mount --daemon), run without prompts and with time limits.
  colab_upload_retry "$RCLONE_CONF" /content/.colab-drive-rclone-upload.conf
  python3 - "$WORK/mount.py" "$DRIVE_ROOT" "$RCLONE_REMOTE" <<'PY'
import pathlib, sys
out, mnt, remote = sys.argv[1:]
code = f'''import os, shutil, subprocess, time
MNT, REMOTE = {mnt!r}, {remote!r}
UPLOAD = '/content/.colab-drive-rclone-upload.conf'
CONF = os.path.expanduser('~/.config/rclone/rclone.conf')
FLAGS = ['--vfs-cache-mode', 'writes', '--vfs-cache-max-size', '10G', '--poll-interval', '15s', '--daemon']

def run(cmd, timeout):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, DEBIAN_FRONTEND='noninteractive'))
        return r.returncode, (r.stdout + r.stderr).strip()[-1500:]
    except (OSError, subprocess.TimeoutExpired) as error:
        return 124, f'{{type(error).__name__}}: {{error}}'

def mounted():
    return subprocess.run(['mountpoint', '-q', MNT]).returncode == 0

os.makedirs(os.path.dirname(CONF), exist_ok=True)
if os.path.exists(UPLOAD):
    shutil.move(UPLOAD, CONF)
if os.path.exists(CONF):
    os.chmod(CONF, 0o600)
    print('[mount] rclone.conf ready', flush=True)
else:
    print('no rclone config at', CONF, flush=True)
if not mounted():
    if shutil.which('rclone') is None or not (shutil.which('fusermount3') or shutil.which('fusermount')):
        print('[mount] Installing rclone and fuse3 via apt-get...', flush=True)
        code, out = run(['apt-get', 'install', '-y', '-qq', 'rclone', 'fuse3'], 300)
        if code != 0:  # no or stale package lists: update once, then retry
            print('[mount] apt-get install failed; running apt-get update...', flush=True)
            run(['apt-get', 'update', '-qq'], 180)
            code, out = run(['apt-get', 'install', '-y', '-qq', 'rclone', 'fuse3'], 300)
        print('install rclone fuse3: exit', code, out[-300:] if code else '', flush=True)
    os.makedirs(MNT, exist_ok=True)
    print('[mount] Mounting', REMOTE, 'at', MNT, '...', flush=True)
    code, out = run(['rclone', 'mount', REMOTE, MNT, *FLAGS], 60)
    print('rclone mount: exit', code, out[-500:], flush=True)
    for _ in range(30):
        if mounted():
            break
        time.sleep(1)
ok = mounted()
if ok:
    top = sorted(os.listdir(MNT))
    print('mounted', REMOTE, 'at', MNT, '| top level:', top[:20], flush=True)
    if not any(name in top for name in ('models', 'runs', 'projects')):
        print('P1_MOUNT_WARNING: no models/, runs/ or projects/ at the top of', REMOTE, '- is it your My Drive root?')
print('P1_MOUNT_' + ('OK' if ok else 'FAILED'), flush=True)
'''
pathlib.Path(out).write_text(code, encoding='utf-8')
PY
  MOUNT_LOG="$WORK/mount.log"
  echo "Mounting Google Drive ($RCLONE_REMOTE at $DRIVE_ROOT)..."
  colab exec -s "$SESSION" --timeout 300 -f "$WORK/mount.py" 2>&1 | tee "$MOUNT_LOG" | grep -a -v '^\[colab\]' || true
  if ! grep -q 'P1_MOUNT_OK' "$MOUNT_LOG"; then
    echo "rclone could not mount $RCLONE_REMOTE at $DRIVE_ROOT on $SESSION (see log above); the VM is released."
    echo "To debug by hand: rerun with --keep, then 'colab console -s SESSION' (a bash shell) and run the"
    echo "mount steps from experiments/qfft_v93/README.md; or use --drive colab."
    exit 1
  fi
fi
upload "$BUNDLE" /content/p1_bundle.tgz
if [[ $WITH_DATA == 1 ]]; then upload "$WORK/p1_data.tgz" /content/p1_data.tgz; fi
if [[ -n $REPORT_REPO && -f $GH_CREDENTIALS ]]; then
  colab_upload_retry "$GH_CREDENTIALS" /content/gh_credentials >/dev/null   # remote.py sets mode 600
fi

ENTRY="$WORK/entry.py"
RELEASE=$(( KEEP == 0 ? 1 : 0 ))
python3 - "$ENTRY" "$CODE_SHA" "$GIT_HEAD" "$RELEASE" "$SESSION" "$GPU_USED" "$FORCE" "$DRIVE_ROOT" \
  "$RCLONE_REMOTE" "$REPORT_REPO" "${P1_REPORT_BRANCH:-colab-reports}" "${P1_REPORT_MINUTES:-30}" "${STAGES[@]}" <<'PY'
import json, pathlib, sys
(entry, sha, head, release, session, gpu, force, drive_root, rclone_remote, report_repo, report_branch,
 report_minutes, *stages) = sys.argv[1:]
args = {'code_sha': sha, 'git_head': head, 'auto_release': release == '1', 'stages': stages,
        'bundle': '/content/p1_bundle.tgz', 'session': session, 'gpu': gpu, 'force': force == '1',
        'drive_root': drive_root, 'rclone_remote': rclone_remote,
        'report': ({'repo': report_repo, 'branch': report_branch, 'credentials': '/content/gh_credentials',
                    'interval_minutes': float(report_minutes)} if report_repo else None)}
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
