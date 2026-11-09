#!/usr/bin/env bash
# One-time setup of WSL2 (or Linux) for the v93 Colab pipeline. Safe to re-run: what is already in place
# is only checked.
#
#   bash experiments/qfft_v93/colab/setup_wsl.sh        (Windows: experiments\qfft_v93\windows\v93_colab.cmd setup)
#
# 1. git/python3/tar/curl, the repository opens, its shell scripts have LF line endings.
# 2. Colab CLI: a `colab` already on PATH is kept (installed with uv, pipx or pip); otherwise installs uv and
#    google-colab-cli in the version range launch.sh was checked against.
# 3. The files launch.sh uploads to every VM: the rclone config with the Drive remote (P1_RCLONE_CONF, default
#    ~/.config/rclone/rclone.conf, section [mydrive]) and the GitHub token for the reports (P1_GH_CREDENTIALS,
#    default ~/.config/p1/gh_credentials). A missing token file is created after asking for the token once
#    (hidden input, mode 600).
# 4. Signs in to Colab: the first time, the CLI prints a URL; approve in a browser and paste the code back.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_SPEC='google-colab-cli>=0.7.5,<0.8'
ok() { echo "  ok    $*"; }
warn() { echo "  WARN  $*"; }
bad() { echo "  FAIL  $*"; }
cli_version() { colab version 2>/dev/null | sed -n 's/^Version: *//p' | tail -n 1; }
problems=0

echo "v93 Colab setup"
missing=0
for tool in git python3 tar curl; do
  if command -v "$tool" >/dev/null; then ok "$tool"; else bad "$tool missing (sudo apt-get install -y $tool)"; missing=1; fi
done
[[ $missing == 0 ]] || exit 1

if ! REPO="$(git -C "$HERE" rev-parse --show-toplevel 2>&1)"; then
  bad "git cannot open the repository: $REPO"
  echo "        If it says 'dubious ownership', run once:"
  echo "        git config --global --add safe.directory '$(cd "$HERE/../../.." && pwd)'"
  exit 1
fi
ok "repository $REPO"
if grep -l $'\r' "$HERE"/*.sh >/dev/null 2>&1; then
  bad "shell scripts have Windows (CRLF) line endings; after pulling the commit with .gitattributes run:"
  echo "        git -C \"$REPO\" checkout -- experiments/qfft_v93"
  exit 1
fi
ok "shell scripts use LF line endings"

export PATH="$HOME/.local/bin:$PATH"
if command -v colab >/dev/null; then
  have="$(cli_version)"
  if [[ $have =~ ^0\.7\.([5-9]|[1-9][0-9]+)$ ]]; then
    ok "colab CLI $have ($(command -v colab))"
  else
    warn "colab CLI ${have:-of unknown version} at $(command -v colab); launch.sh was checked against 0.7.5"
    echo "        To switch to the checked range: uv tool install --force '$CLI_SPEC'"
  fi
else
  if ! command -v uv >/dev/null; then
    echo "  ..    installing uv (official installer from https://docs.astral.sh/uv/)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  ok "$(uv --version)"
  uv tool install -q "$CLI_SPEC"
  uv tool update-shell >/dev/null 2>&1 || true   # puts ~/.local/bin on PATH for new shells
  ok "colab CLI $(cli_version) ($CLI_SPEC; needs Python >= 3.12, uv provides it)"
fi

echo
echo "Files launch.sh uploads to each VM"
CONF="${P1_RCLONE_CONF:-$HOME/.config/rclone/rclone.conf}"
REMOTE="${P1_RCLONE_REMOTE:-mydrive:}"; REMOTE="${REMOTE%%:*}"
if [[ ${P1_DRIVE_MODE:-rclone} != rclone ]]; then
  ok "Drive mode ${P1_DRIVE_MODE}: Colab's own mount, no rclone config needed"
elif [[ -f $CONF ]] && grep -q "^\[$REMOTE\]" "$CONF"; then
  ok "rclone config $CONF defines [$REMOTE]"
else
  bad "no rclone config with a [$REMOTE] section at $CONF"
  echo "        Copy the rclone.conf you used in 'colab console' there, or add this line to ~/.zshrc:"
  echo "        export P1_RCLONE_CONF=/path/to/rclone.conf"
  problems=1
fi

CRED="${P1_GH_CREDENTIALS:-$HOME/.config/p1/gh_credentials}"
REPORT_REPO="${P1_REPORT_REPO:-$(python3 "$HERE/report_push.py" "$(git -C "$REPO" remote get-url origin 2>/dev/null || true)")}"
SLUG="${REPORT_REPO#https://github.com/}"; SLUG="${SLUG%.git}"
if [[ -z $REPORT_REPO ]]; then
  warn "origin is not a GitHub remote, so reports stay on Drive (P1_REPORT_REPO=https://github.com/OWNER/REPO.git changes that)"
elif [[ $REPORT_REPO != https://* ]]; then
  ok "reports go to $REPORT_REPO (no token needed)"
elif [[ -s $CRED ]]; then
  if [[ $(stat -c %a "$CRED") != 600 ]]; then chmod 600 "$CRED"; ok "set $CRED to mode 600"; fi
  ok "GitHub token file $CRED: reports go to $SLUG, branch ${P1_REPORT_BRANCH:-colab-reports}"
elif [[ -t 0 ]]; then
  echo "  ..    GitHub token for the reports (fine-grained: only $SLUG, Contents: Read and write)"
  TOKEN=""
  read -rsp "        Paste it (input hidden; Enter skips): " TOKEN || true
  echo
  if [[ -z $TOKEN ]]; then
    warn "no token: everything still runs, results stay on Drive"
  elif [[ ! $TOKEN =~ ^[A-Za-z0-9_]{20,}$ ]]; then
    bad "that does not look like a GitHub token (ghp_... or github_pat_...); nothing was written"
    problems=1
  else
    (umask 077; mkdir -p "$(dirname "$CRED")"; printf 'https://%s:%s@github.com\n' "${SLUG%%/*}" "$TOKEN" > "$CRED")
    ok "wrote $CRED (mode 600): reports go to $SLUG, branch ${P1_REPORT_BRANCH:-colab-reports}"
  fi
  unset TOKEN
else
  warn "$CRED not found, so reports stay on Drive; run this script in a terminal to create it"
fi

echo
echo "Signing in to Colab. The first time, open the printed URL, approve, and paste the code shown."
colab sessions
echo
if [[ $problems != 0 ]]; then
  echo "Fix the FAIL lines above, then run this script again."
  exit 1
fi
echo "Next: bash experiments/qfft_v93/colab/launch.sh --gpu cpu ping:30"
echo "      (a few minutes on a CPU VM: rclone mount, upload, run, logs, report push, release)"
