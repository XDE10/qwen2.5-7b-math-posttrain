#!/usr/bin/env bash
set -eo pipefail
source "$HOME/.proxy_env.sh"
set -u
cd /home/xde10/.codex/worktrees/mmr-dr500/qwen2.5-7b-math-posttrain
root=/mnt/d/CodexArtifacts/project1-mmr-step500-20261118/validation25
step="${1:?validation step}"; session="${P1_VAL_SESSION:?unique validation session}"
case "$session" in p1-mmrval-a-*) ;; *) exit 2;; esac
export P1_RCLONE_CONF="$HOME/.config/p1/rclone_project1.conf" P1_RCLONE_REMOTE=p1drive:
work=$(mktemp -d);trap 'rm -rf "$work"' EXIT
cat > "$work/colab" <<'CLI'
#!/bin/bash
set -eu
case "$1" in
 exec) exec /home/xde10/.local/bin/colab-a "$@" ;;
 new|upload) exec timeout -k 5s 90s /home/xde10/.local/bin/colab-a "$@" ;;
 *) exec timeout -k 2s 20s /home/xde10/.local/bin/colab-a "$@" ;;
esac
CLI
chmod 700 "$work/colab";export PATH="$work:$PATH"
bundle_sha=$(git ls-files -z -co --exclude-standard -- experiments src scripts configs tools pyproject.toml | tar --null -czf - -T - | sha256sum | cut -c1-12)
expected_sha=$(python3 -c 'import json;print(json.load(open("/mnt/d/CodexArtifacts/project1-mmr-step500-20261118/READY.json"))["code_bundle"])')
[[ $bundle_sha == "$expected_sha" ]] || { echo 'The frozen validation launcher source changed'; exit 1; }
bash experiments/qfft_v93/colab/launch.sh --no-report --drive rclone --gpu G4 --keep --session "$session" shell
timeout 30s colab upload -s "$session" /mnt/d/CodexArtifacts/project1-mmr-step150-preserve-20261117/metadata/c7bcd9df7b7e.tgz /content/p1_validation_source.tgz
timeout 30s colab ssh -s "$session" -- python3 -c 'import hashlib,pathlib,tarfile; p=pathlib.Path("/content/p1_validation_source.tgz"); assert hashlib.sha256(p.read_bytes()).hexdigest()=="c7bcd9df7b7e4d713ff354c054926fb65468ed2444c62747f6c4f0b0fe2025e6"; dest=pathlib.Path("/content/p1/code/c7bcd9df7b7e"); dest.mkdir(parents=True,exist_ok=True); tarfile.open(p).extractall(dest,filter="data")'
for spec in validation.py:p1_validation.py mmr_validation25.json:mmr_validation25.json validation_worker.py:p1_validation_worker.py; do
 timeout 30s colab upload -s "$session" "$root/${spec%%:*}" "/content/${spec#*:}"
done
timeout 30s colab upload -s "$session" "$HOME/.config/p1/rclone_validation.conf" /content/validation_rclone.conf
timeout 30s colab upload -s "$session" /home/xde10/projects/qwen2.5-7b-math-posttrain/data/r1_grpo_dev/r1_grpo_dev.jsonl /content/r1_grpo_dev.jsonl
timeout 40s colab ssh -s "$session" -- bash -c 'chmod 600 /content/validation_rclone.conf; mkdir -p /content/validationdrive; mountpoint -q /content/validationdrive || rclone mount mydrive: /content/validationdrive --config /content/validation_rclone.conf --daemon --vfs-cache-mode writes --vfs-cache-max-size 1G --poll-interval 15s'
# Only this newly allocated evaluation runtime is addressed. No kernel queue or training process is touched.
timeout 25s colab ssh -s "$session" -- bash -c "nohup python3 -u /content/p1_validation_worker.py $step >/content/p1/validation.log 2>&1 </dev/null & echo validation_pid=\$!"
