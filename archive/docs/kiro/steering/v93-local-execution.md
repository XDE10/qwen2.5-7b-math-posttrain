---
inclusion: always
---

# 项目一 v93：本机执行约束（WSL2 + Kiro IDE）

根目录 AGENTS.md 是长期约定，Kiro 会自动加载，这里不重复。本文件只放在本机执行命令时要遵守的规则（2026-11-09 实测总结）。当前状态和下一步见 `docs/handover/kiro_ide_onboarding.md`。

- 机器是 Windows 11 + WSL2 Ubuntu，仓库在 `/home/xde10/projects/qwen2.5-7b-math-posttrain`。交互 shell 是 zsh：给用户的命令必须能直接粘贴进 zsh；仓库脚本都是 bash，一律 `bash 脚本` 运行。Colab VM 里（`colab console`）是 bash。
- 新终端先 `source ~/.proxy_env.sh`，否则 GitHub、Colab CLI、rclone 可能连不上。`bash experiments/qfft_v93/check_local.sh` 会指出代理没加载、凭据缺失或权限不是 600。
- 多步、带嵌套引号、heredoc 或循环的操作写成 `.sh` 脚本再 `bash` 执行，不拼长单行命令，也不经 PowerShell 或 cmd 转发。
- 远程命令要实时可见、有时间上限：`colab exec` 的输出用 `tee` 显示，挂载这类短步骤的超时设几分钟；长任务交给 launch.sh 的脱离与 `--attach`，不要把它包在会静默捕获输出的 `$(...)` 里。
- GPU 任务和付费 API 请求（`teacher fill`）由用户决定并启动。Claude 只准备命令、读结果，不自行启动。
- 结果从 GitHub 读，不让用户复制日志：`git fetch origin colab-reports`，再 `git show origin/colab-reports:v93/latest/status.md`（还有 `v93/latest_smoke_summary.txt`、`v93/sessions/<session>/`）。VM 只推这个分支，永远不推 main。Agent 在本机拉取远端分支或执行外部命令时必须加 `timeout 10s`，严禁无超时阻塞让用户干等。
- 给用户命令前必须核准参数名（如 `launch.sh` 是 `--session` 而非 `-s`、`colab-cli 0.8.0` 列出会话是 `colab sessions` 而非 `colab list`），不碰用户正在运行的其他并发实例（如 `opd`、`p3`）。
- 底层已固化两项保护：(1) `remote.py` 每次启动写入 `active_run_id`，`finally` 的 120s 延迟 `runtime.unassign()` 执行前校验 `active_run_id`，防止前一次失败任务的倒计时销毁后续在同一 VM 上新启动的任务；(2) `launch.sh` 默认轮询间隔 `POLL_S=15` 秒，且传入 `--session` 已存在时直接复用不重复 `colab new`。
- Google Drive 每个账号每 24 小时约 750 GiB 上传上限。一个全参训练上传约 241 GiB，24 小时内最多安排 2 个全参训练。
- 代码改动按阶段建分支（`feat/…`、`fix/…`）、中文 conventional commit 小步提交、PR 合入 main，用户另有指示时从其指示。改了 `docs/metadata/migration_manifest.json` 里登记过的文件要重算其大小与 SHA-256。
- 改调度、教师或 Drive 相关代码后跑 `bash tests/v93_harness/run_all.sh`（只跑一部分时写套件名，例如 `drive teacher data`）。
