#!/usr/bin/env bash
# Read-only check of the local WSL2 setup for v93, in the order the work needs it: repository, network,
# Colab CLI, the credential files launch.sh uploads to every VM, Google Drive (when rclone is installed
# locally), the interpreter for the local data build, the data itself, and the latest Colab report.
# It installs and sends nothing. The only writes: `git fetch` updates origin/main and origin/colab-reports,
# and `data_build status` refreshes data/qfft1000_v93/ds_status.json.
#
#   bash experiments/qfft_v93/check_local.sh               # from zsh too: the script itself runs in bash
#   bash experiments/qfft_v93/check_local.sh --no-data     # skip `data_build status` (it re-reads every attempt)
#   bash experiments/qfft_v93/check_local.sh --no-network  # skip GitHub and Drive
#
# Exit code 1 when a FAIL line is printed. Secrets are never printed: only whether a file or variable exists.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)" || { echo "git cannot open the repository at $HERE"; exit 1; }
cd "$REPO" || exit 1
DATA=1; NET=1
for arg in "$@"; do
  case "$arg" in
    --no-data) DATA=0 ;;
    --no-network) NET=0 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg"; exit 2 ;;
  esac
done
fails=0; warns=0
ok() { echo "  ok    $*"; }
warn() { echo "  WARN  $*"; warns=$((warns + 1)); }
bad() { echo "  FAIL  $*"; fails=$((fails + 1)); }
mode() { stat -c %a "$1" 2>/dev/null; }
no_userinfo() { sed -E 's#://[^/@]*@#://#'; }   # drop user:password@ from URLs before printing them
limit() { timeout "$1" "${@:2}"; }

echo "Repository"
ok "$REPO (branch $(git branch --show-current 2>/dev/null || echo '?'), HEAD $(git rev-parse --short=12 HEAD))"
dirty="$(git status --porcelain | wc -l | tr -d ' ')"
if [[ $dirty == 0 ]]; then ok "working tree clean"; else warn "$dirty changed or untracked files (git status)"; fi
origin="$(git remote get-url origin 2>/dev/null | no_userinfo)"
[[ -n $origin ]] && ok "origin $origin" || bad "no origin remote"

echo
echo "Network"
proxy="${HTTPS_PROXY:-${https_proxy:-${ALL_PROXY:-${all_proxy:-}}}}"
if [[ -n $proxy ]]; then
  ok "proxy set: $(no_userinfo <<< "$proxy")"
elif [[ -f $HOME/.proxy_env.sh ]]; then
  # shellcheck disable=SC2088  # the tilde is shown to the user, who types it in zsh
  warn "~/.proxy_env.sh exists but no proxy variable is set in this shell: run 'source ~/.proxy_env.sh' first"
else
  ok "no proxy configured (none needed if GitHub and Google are reachable directly)"
fi
if [[ $NET == 1 && -n $origin ]]; then
  if limit 60 git ls-remote --heads origin main >/dev/null 2>&1; then
    ok "GitHub reachable (git ls-remote origin)"
    limit 120 git fetch -q origin main colab-reports 2>/dev/null || limit 120 git fetch -q origin main 2>/dev/null || true
    if git rev-parse -q --verify origin/main >/dev/null; then
      read -r behind ahead <<< "$(git rev-list --left-right --count origin/main...HEAD 2>/dev/null || echo '? ?')"
      if [[ $behind == 0 ]]; then ok "up to date with origin/main (ahead $ahead)"
      else warn "$behind commits behind origin/main: git pull"; fi
    fi
  else
    bad "GitHub not reachable from this shell (proxy loaded? 'source ~/.proxy_env.sh')"
  fi
fi

echo
echo "Colab CLI"
export PATH="$HOME/.local/bin:$PATH"
if command -v colab >/dev/null; then
  version="$(colab version 2>/dev/null | sed -n 's/^Version: *//p' | tail -n 1)"
  if [[ $version =~ ^0\.7\.([5-9]|[1-9][0-9]+)$ ]]; then ok "colab $version ($(command -v colab))"
  else warn "colab ${version:-of unknown version}: launch.sh was checked against >=0.7.5,<0.8"; fi
  if [[ -s $HOME/.config/colab-cli/token.json ]]; then ok "signed in (token cache ~/.config/colab-cli/token.json)"
  else warn "not signed in yet: run 'colab sessions' once and approve the printed link"; fi
else
  bad "colab CLI not found: bash experiments/qfft_v93/colab/setup_wsl.sh"
fi

echo
echo "Files launch.sh uploads to every VM"
CRED="${P1_GH_CREDENTIALS:-$HOME/.config/p1/gh_credentials}"
if [[ ! -f $CRED ]]; then
  warn "no GitHub token file $CRED: jobs run, but nothing is pushed to colab-reports (setup_wsl.sh creates it)"
elif ! grep -Eq '^https://[^:/@[:space:]]+:[^@[:space:]]+@github\.com/?$' "$CRED" || [[ $(grep -c . "$CRED") != 1 ]]; then
  bad "$CRED is not one line https://USER:TOKEN@github.com"
elif [[ $(mode "$CRED") != 600 ]]; then
  bad "$CRED has mode $(mode "$CRED"): chmod 600 '$CRED'"
else
  ok "GitHub token file $CRED (mode 600, user $(sed -E 's#^https://([^:]+):.*#\1#' "$CRED"))"
fi
CONF="${P1_RCLONE_CONF:-$HOME/.config/rclone/rclone.conf}"
REMOTE="${P1_RCLONE_REMOTE:-mydrive:}"; NAME="${REMOTE%%:*}"
if [[ ${P1_DRIVE_MODE:-rclone} != rclone ]]; then
  ok "P1_DRIVE_MODE=${P1_DRIVE_MODE}: Colab's own Drive mount, no rclone config needed"
elif [[ ! -f $CONF ]]; then
  bad "no rclone config at $CONF (set P1_RCLONE_CONF in ~/.zshrc if it lives elsewhere)"
elif ! grep -q "^\[$NAME\]" "$CONF"; then
  bad "$CONF has no [$NAME] section"
elif [[ $(mode "$CONF") != 600 ]]; then
  warn "$CONF has mode $(mode "$CONF"); it holds the Drive token: chmod 600 '$CONF'"
else
  ok "rclone config $CONF with [$NAME] (mode 600)"
fi
key=""
[[ -n ${DEEPSEEK_API_KEY:-} ]] && key="DEEPSEEK_API_KEY"
[[ -n ${OPENCODE_GO_API_KEY:-} ]] && key="${key:+$key, }OPENCODE_GO_API_KEY"
for f in "$HOME/.config/p1/deepseek.json" /mnt/c/Users/Mr.Z/.codex/private/project1/deepseek.json \
         /mnt/d/OpenCode/runtime/data/opencode/opencode.db "$HOME/.local/share/opencode/opencode.db"; do
  [[ -f $f ]] && key="${key:+$key, }$f"
done
if [[ -n $key ]]; then ok "DeepSeek key source for teacher fill: $key"
else warn "no DeepSeek key found (only teacher fill needs one; see experiments/qfft_v93/README.md)"; fi

echo
echo "Google Drive ($REMOTE, read-only listing with the local rclone)"
if [[ $NET == 0 ]]; then
  echo "  skip  --no-network"
elif [[ ${P1_DRIVE_MODE:-rclone} != rclone || ! -f $CONF ]]; then
  echo "  skip  no rclone config"
elif ! command -v rclone >/dev/null; then
  warn "rclone is not installed in WSL, so Drive was not listed (optional: sudo apt-get install -y rclone)"
else
  creds="$(limit 90 rclone --config "$CONF" lsf "${REMOTE}.project1_credentials/" 2>/dev/null)"; rc=$?
  if [[ $rc != 0 ]]; then
    bad "rclone could not list ${REMOTE}.project1_credentials/ (exit $rc): token expired? rclone config reconnect $REMOTE"
  else
    if grep -qx 'swanlab.json' <<< "$creds"; then ok "SwanLab key file .project1_credentials/swanlab.json"
    else bad "no .project1_credentials/swanlab.json on Drive: training refuses to run without online SwanLab"; fi
    models="$(limit 90 rclone --config "$CONF" lsf --dirs-only "${REMOTE}models/" 2>/dev/null)"
    for name in base qwen_math qfft_official sft_repro; do
      dir="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["drive_dir"])' \
             experiments/qfft_v93/configs/pipeline.json "$name")"
      if grep -qx "$dir/" <<< "$models"; then ok "model $name: models/$dir"
      else bad "model $name missing on Drive (models/$dir): bash experiments/qfft_v93/colab/pipeline.sh 0"; fi
    done
    pool="runs/project1/rl_approved_v8_n4/project1_rl_rollout_approved_v1.jsonl"
    if limit 90 rclone --config "$CONF" lsf "${REMOTE}$pool" 2>/dev/null | grep -q .; then ok "GRPO pool $pool"
    else warn "GRPO pool $pool not found (needed from phase 4)"; fi
  fi
fi

echo
PY="${P1_PYTHON:-python3}"
echo "Python for the local data build (P1_PYTHON=$PY)"
if ! command -v "$PY" >/dev/null && [[ ! -x $PY ]]; then
  bad "$PY not found: set P1_PYTHON to an interpreter with pandas, pyarrow, tokenizers and math-verify"
else
  ok "$("$PY" --version 2>&1)"
  for module in pandas pyarrow tokenizers math_verify; do
    found="$("$PY" -c "import importlib, sys; sys.path.append('.pilot_deps'); m = importlib.import_module('$module'); print(getattr(m, '__version__', 'present'))" 2>/dev/null)"
    if [[ -n $found ]]; then ok "$module $found"
    else bad "$module missing for $PY ($PY -m pip install pandas pyarrow tokenizers 'math-verify[antlr4_13_2]==0.9.0')"; fi
  done
fi

echo
echo "Data (data/qfft1000_v93)"
if [[ $DATA == 1 ]] && command -v "$PY" >/dev/null; then
  status_out="$(PYTHONUTF8=1 "$PY" -m experiments.qfft_v93.data_build status 2>&1)"; rc=$?
  line="$(grep -a 'DS usable' <<< "$status_out" | tail -n 1 | sed -E 's/^\[[^]]*\] //')"
  if [[ $rc != 0 || -z $line ]]; then
    bad "data_build status failed: $(tail -n 2 <<< "$status_out" | tr '\n' ' ' | cut -c1-300)"
  elif grep -q 'gaps: 0$' <<< "$line"; then ok "$line"
  else warn "$line (see docs/status/current_status.md for the data status)"; fi
elif [[ $DATA == 0 ]]; then
  echo "  skip  --no-data"
fi
manifest="data/qfft1000_v93/manifest.json"
if [[ -f $manifest ]]; then
  summary="$(python3 - "$manifest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding='utf-8'))
rows = ', '.join(f"{name} {v['rows']}" for name, v in m['variants'].items())
print(f"{len(m.get('gaps') or [])}|{len(m.get('paired_exclusions') or [])}|{rows}|{m.get('created', '?')}")
PY
)"
  IFS='|' read -r ngaps nexcluded rows created <<< "$summary"
  if [[ ${ngaps:-0} != 0 ]]; then
    warn "manifest is a preview build ($ngaps DS gaps, $rows; $created): pipeline.sh d refuses it"
  else
    extra=""; [[ ${nexcluded:-0} != 0 ]] && extra=", $nexcluded questions dropped from both teachers"
    ok "manifest $created: $rows$extra"
  fi
else
  echo "  info  no manifest yet (data_build build has not run)"
fi

echo
echo "Latest Colab report (origin/colab-reports)"
if [[ $NET == 1 ]] && git rev-parse -q --verify origin/colab-reports >/dev/null; then
  read -r -d '' session result < <(git show origin/colab-reports:v93/latest/session.txt 2>/dev/null; printf '\0')
  ok "last session ${session:-?}: ${result:-?} ($(git log -1 --format=%cd --date=iso origin/colab-reports))"
  echo "        git show origin/colab-reports:v93/latest/status.md"
elif [[ $NET == 1 ]]; then
  echo "  info  no colab-reports branch yet (it appears after the first VM run with a token file)"
else
  echo "  skip  --no-network"
fi

echo
if [[ $fails == 0 ]]; then echo "Result: $warns warnings, no failures."; else echo "Result: $fails failures, $warns warnings."; fi
[[ $fails == 0 ]]
