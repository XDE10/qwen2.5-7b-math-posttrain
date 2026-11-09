#!/usr/bin/env bash
# One-time setup of WSL2 (or Linux) for the v93 Colab pipeline. Safe to re-run.
#
#   bash experiments/qfft_v93/colab/setup_wsl.sh        (Windows: experiments\qfft_v93\windows\v93_colab.cmd setup)
#
# Checks git/python3/tar/curl, that the repository opens and its shell scripts have LF line endings,
# installs uv and google-colab-cli (the version range launch.sh was checked against), then signs in to
# Colab once: the CLI prints a URL, you approve in a browser and paste the code back.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_SPEC='google-colab-cli>=0.7.5,<0.8'
ok() { echo "  ok    $*"; }
bad() { echo "  FAIL  $*"; }

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
if ! command -v uv >/dev/null; then
  echo "  ..    installing uv (official installer from https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
ok "$(uv --version)"
uv tool install -q --upgrade "$CLI_SPEC"
uv tool update-shell >/dev/null 2>&1 || true   # puts ~/.local/bin on PATH for new shells
ok "colab CLI $(colab version 2>/dev/null | tail -n 1) ($CLI_SPEC; needs Python >= 3.12, uv provides it)"

echo
echo "Signing in to Colab. The first time, open the printed URL, approve, and paste the code shown."
colab sessions
echo
echo "Next: experiments/qfft_v93/colab/launch.sh --gpu cpu ping:30"
echo "      (the first Drive mount prints one more link: approve it and press Enter)"
