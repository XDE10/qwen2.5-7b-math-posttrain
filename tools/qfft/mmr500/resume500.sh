#!/usr/bin/env bash
set -eo pipefail
source "$HOME/.proxy_env.sh"
set -u
cd /home/xde10/.codex/worktrees/mmr-dr500/qwen2.5-7b-math-posttrain
folder=/mnt/d/CodexArtifacts/project1-mmr-step500-20261118
session="${P1_500_SESSION:?Set the unique authorized continuation session}"
case "$session" in p1-mmr500-c-*) ;; *) echo 'Unexpected account or session scope'; exit 2;; esac
export P1_RCLONE_CONF="$HOME/.config/p1/rclone_project1.conf"
export P1_RCLONE_REMOTE=p1drive:
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
cat > "$work/colab" <<'CLI'
#!/usr/bin/env bash
set -eu
case "$1" in
 exec) exec /home/xde10/.local/bin/colab-c "$@" ;;
 new|upload) exec timeout -k 5s 90s /home/xde10/.local/bin/colab-c "$@" ;;
 *) exec timeout -k 2s 15s /home/xde10/.local/bin/colab-c "$@" ;;
esac
CLI
chmod 700 "$work/colab"
export PATH="$work:$PATH"
git ls-files -z -co --exclude-standard -- experiments src scripts configs tools pyproject.toml > "$work/files"
tar --null -czf "$work/current.tgz" -T "$work/files"
python3 - "$folder/READY.json" "$work/current.tgz" <<'PY'
import hashlib,json,pathlib,sys
ready=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()==ready['code_bundle_sha256'], 'Frozen continuation source changed'
p=pathlib.Path('data/rl_openr1_46k_decontaminated/mmr_dr500_frozen_train.jsonl')
assert hashlib.sha256(p.read_bytes()).hexdigest()==ready['training_data_sha256'], 'Frozen 4000-question file changed'
PY
if [[ ${1:-} == --check ]]; then
  echo 'FROZEN_CONTINUATION_OK data=4000 validation=25 account=C; no GPU requested'
  exit 0
fi
flags=()
if [[ ${P1_500_RECOVERY_CONFIRMED:-0} == 1 ]]; then flags+=(--force); fi
bash experiments/qfft_v93/colab/launch.sh --drive rclone --gpu G4 --wait-min 0 --session "$session" \
 "${flags[@]}" grpo:grpo_qfft_r1_mmr_dr500:resume
