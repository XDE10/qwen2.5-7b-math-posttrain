#!/usr/bin/env bash
# Mock end-to-end checks for launch.sh / pipeline.sh / remote.py / stages.py against the fake Colab CLI,
# rclone and Drive in mockbin/. Run through run_all.sh, which prepares a snapshot of the working tree:
#   bash tests/v93_harness/run_all.sh scenarios
#   P1_SCENARIOS="A R2" bash tests/v93_harness/run_all.sh scenarios     (each scenario prints PASS/FAIL lines)
set -u
REPO="${P1_HARNESS_REPO:?run this through tests/v93_harness/run_all.sh}"
T="${P1_HARNESS_DIR:?run this through tests/v93_harness/run_all.sh}"
# Scenario G creates and deletes data/qfft1000_v93 and data/benchmarks_v93: never in a real checkout.
[[ -f $REPO/.p1_harness_snapshot ]] || { echo "FAIL $REPO is not a harness snapshot; use run_all.sh"; exit 1; }
export PATH="$T/mockbin:$PATH" MOCK_PYPATH="$T/pypath" P1_POLL_S=1
L="$REPO/experiments/qfft_v93/colab/launch.sh"
fresh() { export MOCK_ROOT; MOCK_ROOT="$(mktemp -d)"; unset MOCK_FAIL_GPUS MOCK_FAIL_HIGHMEM MOCK_DISCONNECT_AFTER MOCK_NO_DRIVE MOCK_GRACE P1_DATA_ROOT \
    MOCK_RCLONE_MOUNT_FAIL P1_RCLONE_REMOTE P1_REPORT_REPO P1_REPORT_MINUTES P1_REPORT_BRANCH \
    MOCK_POLL_STALLS P1_POLL_TIMEOUT_S MOCK_STATUS_UNAVAILABLE MOCK_BUSY_SESSION; \
  export P1_DRIVE_MODE=colab P1_RCLONE_CONF=$T/fake_rclone.conf P1_GH_CREDENTIALS=/nonexistent/gh_credentials; }
st() { python3 -c "import json,sys; s=json.load(open('$MOCK_ROOT/vms/$1/state.json')); print(s.get('alive'), s.get('stopped_by'), s.get('gpu'))"; }
marker() { cat "$MOCK_ROOT/drive/runs/project1/v93/stages/$1.json" 2>/dev/null | python3 -c "import json,sys; s=json.load(sys.stdin); print(s['state'], s.get('host'), (s.get('error') or '')[:150])"; }
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1   ($2)"; fi; }

A() {  # happy path on a CPU VM
  fresh; out="$("$L" --gpu cpu --session tA ping:3 2>&1)"; rc=$?
  echo "$out" | tail -n 4
  check "A exit 0" "[[ $rc == 0 ]]"
  check "A success line" "grep -q '^SUCCESS: ping:3' <<< \"\$out\""
  check "A VM released" "[[ \"\$(st tA)\" == 'False colab stop CPU' ]]"
  check "A marker complete" "[[ \"\$(marker ping__3)\" == complete* ]]"
  check "A session log on Drive" "[[ -s $MOCK_ROOT/drive/runs/project1/v93/logs/session_tA.log ]]"
  check "A high-mem requested" "grep -q 'new -s tA --high-mem' $MOCK_ROOT/calls.log"
}

B() {  # GPU fallback list
  fresh; export MOCK_FAIL_GPUS=H100; out="$("$L" --gpu h100,g4 --session tB ping:1 2>&1)"; rc=$?
  check "B exit 0" "[[ $rc == 0 ]]"
  check "B fell back to G4" "[[ \"\$(st tB)\" == False*G4 ]]"
  check "B tried H100 first" "grep -q 'new -s tB --gpu H100 --high-mem' $MOCK_ROOT/calls.log"
}

C() {  # unknown GPU name is rejected before any VM is requested
  fresh; out="$("$L" --gpu h200 --session tC ping:1 2>&1)"; rc=$?
  check "C exit 1" "[[ $rc == 1 ]]"
  check "C message" "grep -q \"Unsupported --gpu value 'H200'\" <<< \"\$out\""
  check "C no allocation" "! grep -q '^new' $MOCK_ROOT/calls.log 2>/dev/null"
}

D() {  # live connection drops: job keeps running, launch.sh follows the tail file and still gets the result
  fresh; export MOCK_DISCONNECT_AFTER=3 MOCK_GRACE=30 MOCK_POLL_STALLS=1 P1_POLL_TIMEOUT_S=1
  out="$("$L" --gpu cpu --session tD ping:8 2>&1)"; rc=$?
  echo "$out" | grep -E 'Lost|Following|SUCCESS|FAILED|released' | head
  check "D exit 0" "[[ $rc == 0 ]]"
  check "D switched to polling" "grep -q 'switching to log polling' <<< \"\$out\""
  check "D stalled poll times out and reconnects" "grep -q 'unsuccessful polls' <<< \"\$out\" && grep -q 'Log polling reconnected to tD' <<< \"\$out\""
  check "D success" "grep -q '^SUCCESS' <<< \"\$out\""
  check "D VM released by launch.sh" "[[ \"\$(st tD)\" == 'False colab stop CPU' ]]"
}

DR() {  # a transient or busy same-session lookup must not allocate or stop a runtime
  fresh
  colab new -s tDR >/dev/null
  export MOCK_STATUS_UNAVAILABLE=1
  out="$("$L" --gpu cpu --session tDR ping:1 2>&1)"; rc=$?
  check "DR failed status does not allocate another VM" "[[ $rc == 1 ]] && grep -q 'Cannot verify existing session' <<< \"\$out\" && [[ \$(grep -c '^new' $MOCK_ROOT/calls.log) == 1 ]]"
  unset MOCK_STATUS_UNAVAILABLE
  export MOCK_BUSY_SESSION=tDR
  out="$("$L" --gpu cpu --session tDR ping:1 2>&1)"; rc=$?
  check "DR busy session tells the user to attach" "[[ $rc == 1 ]] && grep -q 'Session tDR is busy. Use --attach' <<< \"\$out\""
  unset MOCK_BUSY_SESSION
  out="$("$L" --attach tDR 2>&1)"; rc=$?
  check "DR missed polls keep session state unknown and the VM alive" "[[ $rc == 1 ]] && grep -q 'VM and job status are unknown' <<< \"\$out\" && [[ \"\$(st tDR)\" == True* ]] && ! grep -q '^stop' $MOCK_ROOT/calls.log"
  out="$("$L" --gpu cpu --session tDR ping:1 2>&1)"; rc=$?
  check "DR idle session is reused without a second allocation" "[[ $rc == 0 ]] && grep -q 'Allocated CPU (existing shape)' <<< \"\$out\" && [[ \$(grep -c '^new' $MOCK_ROOT/calls.log) == 1 ]] && ! grep -q '^stop' $MOCK_ROOT/calls.log"
}

E() {  # a second VM refuses a stage that is still running elsewhere
  fresh; export MOCK_GRACE=1
  "$L" --gpu cpu --session tE1 ping:12 > "$MOCK_ROOT/e1.out" 2>&1 &
  p1=$!; sleep 5
  out="$("$L" --gpu cpu --session tE2 ping:12 2>&1)"; rc=$?
  echo "$out" | grep -E 'FAILED' | head -3
  check "E second launch fails" "[[ $rc == 1 ]]"
  check "E lease message" "grep -q 'already running on tE1@' <<< \"\$out\""
  check "E second VM released" "[[ \"\$(st tE2)\" == False* ]]"
  wait $p1; rc1=$?
  check "E first launch succeeds" "[[ $rc1 == 0 ]]"
  check "E marker owned by tE1" "[[ \"\$(marker ping__12)\" == 'complete tE1@'* ]]"
}

F() {  # detach (SIGTERM/HUP like closing the terminal), then --attach picks the job up again
  fresh; export MOCK_GRACE=20
  "$L" --gpu cpu --session tF ping:10 > "$MOCK_ROOT/f.out" 2>&1 &
  lp=$!; sleep 4; kill -TERM $lp; wait $lp; rc=$?
  tail -n 5 "$MOCK_ROOT/f.out"
  check "F detach exit 75" "[[ $rc == 75 ]]"
  check "F detach note" "grep -q 'Detached from tF' $MOCK_ROOT/f.out"
  check "F VM still alive after detach" "[[ \"\$(st tF)\" == True* ]]"
  out="$("$L" --attach tF 2>&1)"; rc=$?
  echo "$out" | tail -n 3
  check "F attach exit 0" "[[ $rc == 0 ]]"
  check "F attach saw success" "grep -q '^SUCCESS' <<< \"\$out\""
  check "F attach observes completion without issuing stop" "! grep -q '^stop' $MOCK_ROOT/calls.log"
}

G() {  # data bundle larger than one upload request: split, reassembled and hash-checked on the VM
  fresh
  rm -rf "$REPO/data/qfft1000_v93" "$REPO/data/benchmarks_v93"
  mkdir -p "$REPO/data/qfft1000_v93/ds" "$REPO/data/benchmarks_v93"
  python3 - "$REPO/data" <<'PY'
import base64, hashlib, json, os, sys
root = sys.argv[1]
path = f'{root}/qfft1000_v93/ds/train.jsonl'
with open(path, 'w') as f:
    for i in range(1500):
        f.write(json.dumps({'i': i, 'text': base64.b64encode(os.urandom(9000)).decode()}) + '\n')
sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
json.dump({'variants': {'ds': {'sha256': sha, 'rows': 1500}}}, open(f'{root}/qfft1000_v93/manifest.json', 'w'))
json.dump({'note': 'test'}, open(f'{root}/benchmarks_v93/manifest.json', 'w'))
PY
  out="$("$L" --gpu cpu --with-data --session tG sync-data 2>&1)"; rc=$?
  echo "$out" | grep -E 'Uploaded|Reassembled|Synced|SUCCESS|FAILED' | head
  check "G exit 0" "[[ $rc == 0 ]]"
  check "G uploaded in parts" "grep -q 'Uploaded p1_data.tgz in [2-9] parts' <<< \"\$out\""
  check "G reassembled" "grep -q 'Reassembled p1_data.tgz' <<< \"\$out\""
  check "G synced to Drive" "cmp -s $REPO/data/qfft1000_v93/ds/train.jsonl $MOCK_ROOT/drive/projects/project1/v93/data/qfft1000_v93/ds/train.jsonl"
  rm -rf "$REPO/data/qfft1000_v93" "$REPO/data/benchmarks_v93"
}

H() {  # Drive mount that silently fails is caught before anything runs
  fresh; export MOCK_NO_DRIVE=1; out="$("$L" --gpu cpu --session tH ping:1 2>&1)"; rc=$?
  check "H exit 1" "[[ $rc == 1 ]]"
  check "H message" "grep -q 'Google Drive is not mounted on tH' <<< \"\$out\""
  check "H VM released" "[[ \"\$(st tH)\" == 'False colab stop CPU' ]]"
  check "H nothing executed" "! grep -q -- '-f ' $MOCK_ROOT/calls.log"
}

I() {  # a failing stage reports FAILED, marks the stage failed and releases the VM
  fresh; out="$("$L" --gpu cpu --session tI bogus 2>&1)"; rc=$?
  echo "$out" | grep -E '^FAILED' | head -2
  check "I exit 1" "[[ $rc == 1 ]]"
  check "I failed line" "grep -q '^FAILED: bogus: ValueError' <<< \"\$out\""
  check "I marker failed" "[[ \"\$(marker bogus)\" == failed* ]]"
  check "I VM released" "[[ \"\$(st tI)\" == False* ]]"
}

P() {  # pipeline.sh: per-phase GPU preference, user --gpu overrides, CPU phases stay on CPU, stop on detach
  fresh; export MOCK_FAIL_GPUS=H100
  P="$REPO/experiments/qfft_v93/colab/pipeline.sh"
  "$P" 1 --session tP1 --wait-min 0 > "$MOCK_ROOT/p1.out" 2>&1 || true
  check "P phase 1 tries G4 first (default G4,H100)" "grep -q 'new -s tP1 --gpu G4 --high-mem' $MOCK_ROOT/calls.log && ! grep -q 'new -s tP1 --gpu H100' $MOCK_ROOT/calls.log"
  "$P" 2 --session tP2 --gpu A100 > "$MOCK_ROOT/p2.out" 2>&1 || true
  check "P user --gpu overrides" "grep -q 'new -s tP2 --gpu A100 --high-mem' $MOCK_ROOT/calls.log && ! grep -q 'new -s tP2 --gpu G4' $MOCK_ROOT/calls.log"
  "$P" 7 --session tP7 --gpu H100 > "$MOCK_ROOT/p7.out" 2>&1 || true
  check "P CPU phase ignores --gpu" "grep -q 'new -s tP7 --high-mem$' $MOCK_ROOT/calls.log"
}

J() {  # stage guards: training data missing, too little host RAM for a full-FT smoke
  fresh; out="$("$L" --gpu cpu --session tJ1 train:qfft_ds 2>&1)"; rc=$?
  echo "$out" | grep -E '^FAILED' | head -2
  check "J train without data fails fast" "[[ $rc == 1 ]] && grep -q 'qfft1000_v93/ds/train.jsonl is missing: build the data locally, then run pipeline.sh d' <<< \"\$out\""
  out="$("$L" --gpu cpu --session tJ2 smoke-train 2>&1)"; rc=$?
  echo "$out" | grep -E '^FAILED|GPU:' | head -3
  check "J smoke-train refuses a small-RAM VM" "[[ $rc == 1 ]] && grep -q 'full-parameter training needs about 122 GB' <<< \"\$out\""
  check "J GPU banner without nvidia-smi" "grep -q 'nvidia-smi not found' <<< \"\$out\""
}

K() {  # high-RAM shape refused for G4 -> standard G4 shape before moving on to the next GPU
  fresh; export MOCK_FAIL_HIGHMEM=G4; out="$("$L" --gpu G4,H100 --session tK ping:1 2>&1)"; rc=$?
  check "K exit 0" "[[ $rc == 0 ]]"
  check "K tried high-mem, then standard G4" "grep -q 'new -s tK --gpu G4 --high-mem' $MOCK_ROOT/calls.log && grep -qx 'new -s tK --gpu G4' $MOCK_ROOT/calls.log"
  check "K did not fall through to H100" "! grep -q 'gpu H100' $MOCK_ROOT/calls.log"
  check "K reports the shape" "grep -q 'Allocated G4 (standard shape) as tK' <<< \"\$out\""
}

Ldata() {  # --with-data from another checkout (--data-root), and the hint when the data is missing here
  fresh; D="$(mktemp -d)"; mkdir -p "$D/data/qfft1000_v93/ds" "$D/data/benchmarks_v93"
  echo '{"x": 1}' > "$D/data/qfft1000_v93/ds/train.jsonl"
  python3 -c "import hashlib,json,sys; d=sys.argv[1]; sha=hashlib.sha256(open(d+'/data/qfft1000_v93/ds/train.jsonl','rb').read()).hexdigest(); json.dump({'variants': {'ds': {'sha256': sha, 'rows': 1}}}, open(d+'/data/qfft1000_v93/manifest.json','w')); json.dump({}, open(d+'/data/benchmarks_v93/manifest.json','w'))" "$D"
  out="$("$L" --gpu cpu --session tL1 --with-data sync-data 2>&1)"; rc=$?
  check "L missing data here exits 1 before allocation" "[[ $rc == 1 ]] && ! grep -q '^new' $MOCK_ROOT/calls.log"
  check "L hint mentions --data-root" "grep -q -- '--data-root' <<< \"\$out\""
  out="$("$L" --gpu cpu --session tL2 --with-data --data-root "$D" sync-data 2>&1)"; rc=$?
  check "L --data-root succeeds" "[[ $rc == 0 ]] && grep -q 'Data from $D' <<< \"\$out\""
  check "L data synced to Drive" "cmp -s $D/data/qfft1000_v93/ds/train.jsonl $MOCK_ROOT/drive/projects/project1/v93/data/qfft1000_v93/ds/train.jsonl"
  export P1_DATA_ROOT="$D"; out="$("$L" --gpu cpu --session tL3 --with-data sync-data 2>&1)"; rc=$?
  check "L P1_DATA_ROOT works too" "[[ $rc == 0 ]]"
  rm -rf "$D"
}

M() {  # a failing smoke stage still prints the summary block, before P1_RESULT
  fresh; out="$("$L" --gpu cpu --session tM smoke-train 2>&1)"; rc=$?
  check "M exit 1" "[[ $rc == 1 ]]"
  check "M summary block printed" "grep -q 'P1 SMOKE SUMMARY' <<< \"\$out\" && grep -q 'END P1 SMOKE SUMMARY' <<< \"\$out\""
  check "M summary shows the failed train stage and the reason" "grep -A3 '^train (full FT' <<< \"\$out\" | grep -q 'failed' && grep -q 'Host RAM is' <<< \"\$out\""
  b=$(grep -n 'P1 SMOKE SUMMARY (copy' <<< "$out" | head -1 | cut -d: -f1); r=$(grep -n '^P1_RESULT=' <<< "$out" | head -1 | cut -d: -f1)
  check "M summary comes before P1_RESULT" "[[ -n $b && -n $r && $b -lt $r ]]"
  check "M summary saved on Drive" "[[ -s $MOCK_ROOT/drive/runs/project1/v93/smoke/latest.txt ]]"
  out="$("$L" --gpu cpu --session tM2 smoke-report 2>&1)"; rc=$?
  check "M smoke-report re-prints it" "[[ $rc == 0 ]] && grep -q 'P1 SMOKE SUMMARY' <<< \"\$out\""
}

N() {  # a failed smoke stage does not stop the later smoke stages of the same session
  fresh; out="$("$L" --gpu cpu --session tN smoke-train smoke-report 2>&1)"; rc=$?
  check "N exit 1 (smoke-train failed)" "[[ $rc == 1 ]]"
  check "N continued to the next smoke stage" "grep -q 'smoke-train failed (RuntimeError); continuing' <<< \"\$out\" && grep -q '===== STAGE smoke-report' <<< \"\$out\""
  check "N summary printed once" "[[ \$(grep -c 'P1 SMOKE SUMMARY (copy' <<< \"\$out\") == 1 ]]"
  check "N result lists the failed smoke stage" "grep -q '^P1_RESULT=FAILED smoke-train: RuntimeError: Host RAM' <<< \"\$out\""
}

R1() {  # rclone mode without a config: stop before any VM is requested
  fresh; export P1_DRIVE_MODE=rclone P1_RCLONE_CONF=/nonexistent/rclone.conf
  out="$("$L" --session tR1 ping:1 2>&1)"; rc=$?
  check "R1 exit 1" "[[ $rc == 1 ]]"
  check "R1 message names the file" "grep -q 'rclone config not found: /nonexistent/rclone.conf' <<< \"\$out\""
  check "R1 no allocation" "! grep -q '^new' $MOCK_ROOT/calls.log 2>/dev/null"
  export P1_RCLONE_CONF=$T/wrong_rclone.conf; out="$("$L" --session tR1b ping:1 2>&1)"; rc=$?
  check "R1 config without [mydrive] is refused" "[[ $rc == 1 ]] && grep -q 'has no \[mydrive\] section' <<< \"\$out\""
}

R2() {  # rclone mode (the default): config uploaded, mounted at /content/mydrive, stages use it
  fresh; unset P1_DRIVE_MODE
  out="$("$L" --gpu cpu --session tR2 ping:2 2>&1)"; rc=$?
  echo "$out" | grep -E 'mounted|P1_MOUNT|SUCCESS|FAILED' | head -4
  check "R2 exit 0" "[[ $rc == 0 ]]"
  check "R2 config uploaded to the path used by hand" "grep -q 'upload -s tR2 $T/fake_rclone.conf /content/.colab-drive-rclone-upload.conf' $MOCK_ROOT/calls.log"
  check "R2 no colab drivemount" "! grep -q '^drivemount' $MOCK_ROOT/calls.log"
  check "R2 rclone mount flags" "grep -q 'mount mydrive: .*/content/mydrive --vfs-cache-mode writes --vfs-cache-max-size 10G --poll-interval 15s --daemon' $MOCK_ROOT/rclone_calls.log"
  check "R2 config moved into ~/.config/rclone with mode 600" "[[ \$(stat -c %a $MOCK_ROOT/vms/tR2/.config/rclone/rclone.conf) == 600 ]] && [[ ! -e $MOCK_ROOT/vms/tR2/content/.colab-drive-rclone-upload.conf ]]"
  check "R2 stage wrote through the rclone mount" "ls $MOCK_ROOT/drive/projects/project1/v93/ping/ | grep -q tR2"
  check "R2 marker on Drive" "[[ \$(python3 -c \"import json; print(json.load(open('$MOCK_ROOT/drive/runs/project1/v93/stages/ping__2.json'))['state'])\") == complete ]]"
}

R3() {  # mount failure: clear message, VM released, nothing executed
  fresh; export P1_DRIVE_MODE=rclone MOCK_RCLONE_MOUNT_FAIL=1
  out="$("$L" --gpu cpu --session tR3 ping:1 2>&1)"; rc=$?
  check "R3 exit 1" "[[ $rc == 1 ]]"
  check "R3 message" "grep -q 'rclone could not mount mydrive: at /content/mydrive on tR3' <<< \"\$out\" && grep -q -- '--keep' <<< \"\$out\""
  check "R3 VM released" "[[ \"\$(st tR3)\" == False* ]]"
  check "R3 entry never ran" "! grep -q 'entry.py' $MOCK_ROOT/calls.log"
}

RP() {  # reports pushed to a git remote (a local bare repo stands in for GitHub)
  fresh; export P1_DRIVE_MODE=rclone; ORIGIN="$(mktemp -d)/origin.git"; git init -q --bare "$ORIGIN"
  CRED="$(mktemp -d)/gh_credentials"; printf 'https://XDE10:ghp_FAKEtoken0123456789abcdefghij@github.com\n' > "$CRED"
  export P1_REPORT_REPO="file://$ORIGIN" P1_GH_CREDENTIALS="$CRED" P1_REPORT_MINUTES=0.05
  out="$("$L" --gpu cpu --session tRP ping:8 2>&1)"; rc=$?
  echo "$out" | grep -E '^\[report\]|Reports:' | head -4
  check "RP exit 0" "[[ $rc == 0 ]]"
  B=colab-reports
  check "RP branch exists" "git --git-dir=$ORIGIN rev-parse -q --verify $B >/dev/null"
  check "RP periodic and final pushes" "(( \$(git --git-dir=$ORIGIN rev-list --count $B) >= 2 ))"
  check "RP session README says OK" "git --git-dir=$ORIGIN show $B:v93/sessions/tRP/README.md | grep -q 'result: OK'"
  check "RP latest status and log tail" "git --git-dir=$ORIGIN show $B:v93/latest/status.md | grep -q 'ping__8\|ping:8' && git --git-dir=$ORIGIN show $B:v93/sessions/tRP/session_log_tail.txt | grep -q 'ping:'"
  check "RP stage marker included" "git --git-dir=$ORIGIN show $B:v93/latest/stages.json | grep -q 'ping__8'"
  check "RP token never pushed" "! git --git-dir=$ORIGIN grep -q ghp_FAKEtoken $B"
  check "RP token not in the local log" "! grep -q ghp_FAKEtoken $REPO/reports/colab_logs/tRP.log"
  check "RP credentials uploaded, then removed from the VM" "grep -q 'upload -s tRP $CRED /content/gh_credentials' $MOCK_ROOT/calls.log && [[ ! -e $MOCK_ROOT/vms/tRP/content/gh_credentials ]]"
  out="$("$L" --gpu cpu --session tRP2 bogus 2>&1)"; rc=$?
  check "RP failed session is reported too" "[[ $rc == 1 ]] && git --git-dir=$ORIGIN show $B:v93/sessions/tRP2/README.md | grep -q 'result: FAILED bogus'"
  check "RP both sessions kept, latest points at the last" "git --git-dir=$ORIGIN show $B:v93/sessions/tRP/README.md >/dev/null && git --git-dir=$ORIGIN show $B:v93/latest/session.txt | grep -q tRP2"
  export P1_REPORT_REPO="https://github.com/XDE10/example.git" P1_GH_CREDENTIALS=/nonexistent/cred
  out="$("$L" --gpu cpu --session tRP3 ping:1 2>&1)"; rc=$?
  check "RP missing credentials: job runs, nothing pushed, notice shown" "[[ $rc == 0 ]] && grep -q 'not found, so nothing is pushed' <<< \"\$out\" && ! grep -q 'upload -s tRP3 .*gh_credentials' $MOCK_ROOT/calls.log"
  out="$("$L" --gpu cpu --session tRP4 --no-report ping:1 2>&1)"; rc=$?
  check "RP --no-report is silent" "[[ $rc == 0 ]] && ! grep -q '^Reports:' <<< \"\$out\""
}

SD() {  # sync-data refuses a preview build (DS gaps) before touching Drive; a paired exclusion is accepted
  fresh; D="$(mktemp -d)"; mkdir -p "$D/data/qfft1000_v93/ds" "$D/data/benchmarks_v93"
  old="$MOCK_ROOT/drive/projects/project1/v93/data/qfft1000_v93/ds/train.jsonl"
  mkdir -p "$(dirname "$old")"; echo '{"previous": "upload"}' > "$old"
  echo '{"x": 1}' > "$D/data/qfft1000_v93/ds/train.jsonl"
  echo '{}' > "$D/data/benchmarks_v93/manifest.json"
  manifest() {  # $1: gaps|paired
    python3 - "$D" "$1" <<'PY'
import hashlib, json, sys
root, kind = sys.argv[1], sys.argv[2]
sha = hashlib.sha256(open(f'{root}/data/qfft1000_v93/ds/train.jsonl', 'rb').read()).hexdigest()
m = {'variants': {'ds': {'sha256': sha, 'rows': 1}}, 'gaps': [], 'paired_exclusions': []}
if kind == 'gaps':
    m['gaps'] = ['s1k11/row/0021', 's1k11/row/0024']
else:
    m['paired_exclusions'] = [{'sample_id': 's1k11/row/0021', 'index': 21, 'ds_attempts': []}]
json.dump(m, open(f'{root}/data/qfft1000_v93/manifest.json', 'w'))
PY
  }
  manifest gaps
  out="$("$L" --gpu cpu --session tSD1 --with-data --data-root "$D" sync-data 2>&1)"; rc=$?
  echo "$out" | grep -E 'preview build|FAILED' | head -2
  check "SD preview build refused" "[[ $rc == 1 ]] && grep -q 'is a preview build (data_build build --allow-gaps): 2 questions' <<< \"\$out\""
  check "SD Drive copy untouched" "grep -q previous $old"
  manifest paired
  out="$("$L" --gpu cpu --session tSD2 --with-data --data-root "$D" sync-data 2>&1)"; rc=$?
  check "SD paired exclusion accepted and listed" "[[ $rc == 0 ]] && grep -q 'dropped from both teachers (paired exclusion): s1k11/row/0021' <<< \"\$out\""
  check "SD new data on Drive" "cmp -s $D/data/qfft1000_v93/ds/train.jsonl $old"
  rm -rf "$D"
}

for s in "$@"; do echo "===== scenario $s"; "$s"; done
