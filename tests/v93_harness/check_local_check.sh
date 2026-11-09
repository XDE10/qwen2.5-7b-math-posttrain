#!/usr/bin/env bash
# Checks for experiments/qfft_v93/check_local.sh with a throwaway HOME, stub colab/rclone (localbin/), a stub
# Python (stdlib only plus fake package folders) and a local bare repository standing in for GitHub.
# shellcheck disable=SC2034  # out and code are read inside the eval-ed check expressions
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${P1_HARNESS_REPO:-$(git -C "$H" rev-parse --show-toplevel)}"
[[ -f $REPO/.p1_harness_snapshot ]] || { echo "FAIL $REPO is not a harness snapshot; use run_all.sh"; exit 1; }
C="$REPO/experiments/qfft_v93/check_local.sh"
T="$(mktemp -d "${TMPDIR:-/tmp}/p1_local.XXXXXX")"
pass=0; fail=0
check() { if eval "$2"; then echo "PASS $1"; pass=$((pass + 1)); else echo "FAIL $1"; fail=$((fail + 1)); fi; }

# GitHub stand-in with main and colab-reports, wired as origin of the snapshot
ORIGIN="$T/origin.git"; git init -q --bare "$ORIGIN"
git -C "$REPO" remote remove origin 2>/dev/null; git -C "$REPO" remote add origin "file://$ORIGIN"
git -C "$REPO" push -q origin HEAD:refs/heads/main
R="$T/reports"; git init -q "$R"; mkdir -p "$R/v93/latest"; printf 'p1-1109-180121\nOK\n' > "$R/v93/latest/session.txt"
git -C "$R" add -A; git -C "$R" -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit -q -m r
git -C "$R" push -q "file://$ORIGIN" HEAD:refs/heads/colab-reports

# fake Python: stdlib only (-S), plus fake pandas/pyarrow/tokenizers/math_verify packages on PYTHONPATH
mkdir -p "$T/pkgs" "$T/bin"
for m in pandas pyarrow tokenizers math_verify; do mkdir -p "$T/pkgs/$m"; echo "__version__ = '9.9-$m'" > "$T/pkgs/$m/__init__.py"; done
printf '#!/usr/bin/env bash\nexec python3 -S "$@"\n' > "$T/bin/py_ok"; chmod +x "$T/bin/py_ok"
mkdir -p "$REPO/data/qfft1000_v93"
manifest() { python3 -c 'import json, sys; json.dump({"created": "2026-11-09T12:00:00", "gaps": sys.argv[2:],
  "paired_exclusions": [], "variants": {"r1": {"rows": 1000}, "ds": {"rows": 1000 - len(sys.argv[2:])}}},
  open(sys.argv[1], "w"))' "$REPO/data/qfft1000_v93/manifest.json" "$@"; }

home() {  # $1: full | empty
  rm -rf "${T:?}/home"; mkdir -p "$T/home"
  [[ $1 == empty ]] && return
  mkdir -p "$T/home/.config/p1" "$T/home/.config/rclone" "$T/home/.config/colab-cli"
  printf 'https://XDE10:ghp_SECRETsecretSECRETsecret0123@github.com\n' > "$T/home/.config/p1/gh_credentials"
  chmod 600 "$T/home/.config/p1/gh_credentials"
  printf '[mydrive]\ntype = drive\ntoken = {"access_token":"SECRETDRIVE"}\n' > "$T/home/.config/rclone/rclone.conf"
  chmod 600 "$T/home/.config/rclone/rclone.conf"
  echo '{"token": "x"}' > "$T/home/.config/colab-cli/token.json"
  printf '{"DEEPSEEK_API_KEY": "sk-SECRETdeepseek"}\n' > "$T/home/.config/p1/deepseek.json"
}
run() { env -i HOME="$T/home" PATH="$H/localbin:/usr/local/bin:/usr/bin:/bin" TERM=dumb PYTHONPATH="$T/pkgs" \
        P1_PYTHON="$T/bin/py_ok" "$@" bash "$C" --no-data 2>&1; }

home full; manifest
out="$(run)"; code=$?
check "A ready setup exits 0" '[[ $code == 0 && $out == *"Result: "*"no failures."* ]]'
check "A token file accepted, user shown, token hidden" '[[ $out == *"GitHub token file"*"(mode 600, user XDE10)"* && $out != *SECRET* ]]'
check "A rclone config, SwanLab key, the four models and the GRPO pool" '[[ $out == *"rclone config"*"[mydrive] (mode 600)"* && $out == *"SwanLab key file"* && $(grep -c "ok    model " <<< "$out") == 4 && $out == *"GRPO pool"* ]]'
check "A Colab CLI version and sign-in" '[[ $out == *"ok    colab 0.7.5"* && $out == *"signed in (token cache"* ]]'
check "A GitHub reachable, up to date, last report read" '[[ $out == *"GitHub reachable"* && $out == *"up to date with origin/main"* && $out == *"last session p1-1109-180121: OK"* ]]'
check "A data interpreter modules" '[[ $(grep -c "ok    \(pandas\|pyarrow\|tokenizers\|math_verify\) 9.9-" <<< "$out") == 4 ]]'
check "A final manifest summarised" '[[ $out == *"ok    manifest 2026-11-09T12:00:00: r1 1000, ds 1000"* ]]'
check "A DeepSeek key source named, key hidden" '[[ $out == *"DeepSeek key source for teacher fill: $T/home/.config/p1/deepseek.json"* ]]'

manifest s1k11/row/0021 s1k11/row/0024
out="$(run)"
check "B preview manifest is flagged" '[[ $out == *"WARN  manifest is a preview build (2 DS gaps"* ]]'
manifest

home full; chmod 644 "$T/home/.config/p1/gh_credentials"; touch "$T/home/.proxy_env.sh"
out="$(run)"; code=$?
check "C loose token file mode fails with the chmod fix" '[[ $code == 1 && $out == *"has mode 644: chmod 600"* ]]'
check "C unloaded ~/.proxy_env.sh is pointed out" '[[ $out == *"source ~/.proxy_env.sh"* ]]'
out="$(run HTTPS_PROXY=http://user:pw@127.0.0.1:7890)"
check "C proxy shown without its password" '[[ $out == *"proxy set: http://127.0.0.1:7890"* && $out != *"user:pw"* ]]'

home empty
out="$(run P1_PYTHON=/nonexistent/python)"; code=$?
check "D empty HOME: missing rclone config and interpreter fail, token file only warns" '[[ $code == 1 && $out == *"FAIL  no rclone config at"* && $out == *"FAIL  /nonexistent/python not found"* && $out == *"WARN  no GitHub token file"* ]]'

home full
out="$(run MOCK_RCLONE_FAIL=1)"; code=$?
check "E Drive listing failure suggests reconnecting rclone" '[[ $code == 1 && $out == *"rclone config reconnect mydrive:"* ]]'
out="$(env -i HOME="$T/home" PATH="$H/localbin:/usr/local/bin:/usr/bin:/bin" PYTHONPATH="$T/pkgs" P1_PYTHON="$T/bin/py_ok" bash "$C" --no-data --no-network 2>&1)"
check "E --no-network skips GitHub and Drive" '[[ $out != *"GitHub reachable"* && $(grep -c "skip  --no-network" <<< "$out") == 2 ]]'

git -C "$REPO" remote remove origin
rm -rf "$T" "$REPO/data/qfft1000_v93"
echo "check_local_check: $pass passed, $fail failed"
[[ $fail == 0 ]]
