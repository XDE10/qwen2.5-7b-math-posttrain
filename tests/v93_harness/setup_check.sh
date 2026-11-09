#!/usr/bin/env bash
# Checks for experiments/qfft_v93/colab/setup_wsl.sh with stub colab/uv (setupbin*/) and a throwaway HOME.
# shellcheck disable=SC2034  # out, code and F are read inside the eval-ed check expressions
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${P1_HARNESS_REPO:-$(git -C "$H" rev-parse --show-toplevel)}"
S="$REPO/experiments/qfft_v93/colab/setup_wsl.sh"
T="$(mktemp -d "${TMPDIR:-/tmp}/p1_setup.XXXXXX")"
pass=0; fail=0
check() { if eval "$2"; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1"; fail=$((fail+1)); fi; }
fresh() { rm -rf "$T"; mkdir -p "$T/home"; }
conf() { mkdir -p "$T/home/.config/rclone"; printf '[mydrive]\ntype = drive\n' > "$T/home/.config/rclone/rclone.conf"; }
GH=https://github.com/XDE10/qwen2.5-7b-math-posttrain.git
BASEPATH="$H/setupbin:/usr/local/bin:/usr/bin:/bin"
run() { env -i HOME="$T/home" PATH="${RUNPATH:-$BASEPATH}" TERM=dumb P1_HARNESS_SETUPBIN="$H/setupbin" "$@" bash "$S"; }
tty_run() { (sleep 3; printf '%s\n' "$1") | env -i HOME="$T/home" PATH="$BASEPATH" TERM=dumb P1_REPORT_REPO=$GH \
             script -qec "bash $S" /dev/null; }

fresh; out="$(run P1_REPORT_REPO=$GH </dev/null 2>&1)"; code=$?
check "A missing rclone conf fails" '[[ $code == 1 && $out == *"FAIL  no rclone config with a [mydrive] section"* ]]'
check "A missing token only warns (non-interactive)" '[[ $out == *"WARN  $T/home/.config/p1/gh_credentials not found"* ]]'
check "A no Next line on failure" '[[ $out != *"Next:"* && $out == *"Fix the FAIL lines"* ]]'
check "A existing CLI kept" '[[ $out == *"ok    colab CLI 0.7.5 ($H/setupbin/colab)"* ]]'

fresh; conf; mkdir -p "$T/home/.config/p1"; printf 'https://XDE10:ghp_abcdefghijklmnopqrstuvwxyz0123@github.com\n' > "$T/home/.config/p1/gh_credentials"
chmod 644 "$T/home/.config/p1/gh_credentials"
out="$(run P1_REPORT_REPO=$GH </dev/null 2>&1)"; code=$?
check "B ready setup exits 0 with Next" '[[ $code == 0 && $out == *"Next: bash experiments/qfft_v93/colab/launch.sh --gpu cpu ping:30"* ]]'
check "B rclone conf ok" '[[ $out == *"ok    rclone config $T/home/.config/rclone/rclone.conf defines [mydrive]"* ]]'
check "B token file tightened to 600" '[[ $(stat -c %a "$T/home/.config/p1/gh_credentials") == 600 && $out == *"set $T/home/.config/p1/gh_credentials to mode 600"* ]]'
check "B report target shown" '[[ $out == *"reports go to XDE10/qwen2.5-7b-math-posttrain, branch colab-reports"* ]]'
check "B token never printed" '[[ $out != *ghp_abcdef* ]]'

fresh; conf; out="$(tty_run 'ghp_ZYXWVUTSRQPONMLKJIHGFEDCBA9876' 2>&1)"; code=$?
F="$T/home/.config/p1/gh_credentials"
check "C prompt writes the token file" '[[ $code == 0 && -f $F && $(cat "$F") == "https://XDE10:ghp_ZYXWVUTSRQPONMLKJIHGFEDCBA9876@github.com" ]]'
check "C file and dir modes" '[[ $(stat -c %a "$F") == 600 && $(stat -c %a "$(dirname "$F")") == 700 ]]'
check "C token not echoed" '[[ $out != *ghp_ZYX* ]]'

fresh; conf; out="$(tty_run 'https://XDE10:ghp_x@github.com' 2>&1)"; code=$?
check "D malformed token rejected, nothing written" '[[ $code == 1 && ! -e $T/home/.config/p1/gh_credentials && $out == *"does not look like a GitHub token"* ]]'

fresh; conf; out="$(tty_run '' 2>&1)"; code=$?
check "E Enter skips with a warning" '[[ $code == 0 && ! -e $T/home/.config/p1/gh_credentials && $out == *"WARN  no token"* ]]'

fresh; conf; out="$(run P1_REPORT_REPO=$GH STUB_COLAB_VERSION=0.6.0 </dev/null 2>&1)"; code=$?
check "F other CLI version warns but continues" '[[ $code == 0 && $out == *"WARN  colab CLI 0.6.0"* && $out == *"uv tool install --force"* ]]'

fresh; conf; out="$(RUNPATH="$H/setupbin_uv:/usr/local/bin:/usr/bin:/bin" run P1_REPORT_REPO=$GH </dev/null 2>&1)"; code=$?
check "G installs the CLI with uv when missing" '[[ $code == 0 && $(cat "$T/home/uv_called" 2>/dev/null) == "tool install -q google-colab-cli>=0.7.5,<0.8" && $out == *"ok    colab CLI 0.7.5 (google-colab-cli"* ]]'

fresh; conf; printf '[url "https://git.example.invalid/"]\n\tinsteadOf = https://github.com/\n' > "$T/home/.gitconfig"
out="$(run </dev/null 2>&1)"; code=$?
check "H non-GitHub origin only warns" '[[ $code == 0 && $out == *"WARN  origin is not a GitHub remote"* ]]'

fresh; out="$(run P1_DRIVE_MODE=colab P1_REPORT_REPO=$GH </dev/null 2>&1)"; code=$?
check "I --drive colab mode needs no rclone conf" '[[ $code == 0 && $out == *"Drive mode colab"* ]]'

rm -rf "$T"
echo "setup_check: $pass passed, $fail failed"
[[ $fail == 0 ]]
