# experiments/qfft_v93：1000 题双教师 × SFT/QFFT 全参 × LoRA-GRPO 管线

v93 取代 `aimo_sft_v26`（LoRA、8K 窗口、静态 YaRN、并发 4 的评测）。一条命令在 Colab VM 上跑一个或多个阶段。评测可从 Drive 已保存的生成继续，训练续跑需要完整训练状态。

方案和面试材料见 [docs/blueprints/三周冲刺方案_v93.md](../../docs/blueprints/三周冲刺方案_v93.md) 和 [docs/interview/面试追问与参考答案.md](../../docs/interview/面试追问与参考答案.md)。

## 实验矩阵

| 名称 | 教师轨迹 | 格式 | 初始化 | 说明 |
|---|---|---|---|---|
| `qfft_ds_v93` | DeepSeek-V4.1-Flash（自建） | QFFT | Qwen2.5-7B-Instruct | 主线 |
| `sft_ds_v93` | 同上，response 逐字相同 | SFT | 同上 | 与 QFFT 只差"有没有题目" |
| `qfft_r1_v93` / `sft_r1_v93` | s1K-1.1 的 R1 轨迹 | QFFT / SFT | 同上 | 受控的教师对比 |
| `grpo_qfft_ds_v93` | 920 + 637 题 RL 池 | LoRA-GRPO | `qfft_ds_v93` 合并权重 | 主线 RL |
| `grpo_sft_ds_v93` | 同上 | LoRA-GRPO | `sft_ds_v93` | 可选：同预算比较 RL 起点 |
| 锚点 | — | — | E0、QFFT 作者权重、公开 SFT 孪生、Qwen2.5-Math-7B-Instruct | 只评测 |

固定配方（[configs/train/_defaults.json](configs/train/_defaults.json)）：全参、LR 1e-5、cosine、warmup 10%、6 epoch、wd 1e-4、AdamW(0.9, 0.999)，与 QFFT 官方 yaml 相同；只改两处：有效 batch 16（官方 8，取 s1.1-7B 的值，CPU-Adam 步数减半）和 32K 窗口不截断（官方 16K 截断）。系统实现为 DeepSpeed ZeRO-2 + CPU AdamW offload、梯度检查点、Liger。

### R1 原版配方对照

11 月 12 日用户同意准备这个对照。它用 Drive 上已上传的 v93 去污 R1 集、固定底座和现有训练器，采用[原版配方](https://github.com/LWL-cpu/Question-Free-Fine-Tuning/blob/main/LLaMA-Factory/examples/train_qfft/train_s1_qfft.yaml)的 response 前 16,384 token、有效 batch 8 和 6 epoch。使用独立的 `qfft_r1_16k_b8_v93` run，训练数据文件不需要重建。题目替换和训练框架沿用 v93，所以这是原版配方对照。第 750 步后导出最终权重并跑 v93 全套评测。

本地用固定 tokenizer 核对了 1,000 条轨迹和数据 SHA-256。86 条触发截断，共移除 129,034 token，其中 57 条截断后没有最终回答段。当前配置每 125 步本地保存一次完整检查点，中间 Drive 镜像排除优化器状态。只有轻量镜像时，训练入口会提示缺少完整状态并退出。

由用户在 WSL2 仓库根目录启动：

```zsh
source ~/.proxy_env.sh
bash experiments/qfft_v93/colab/launch.sh --gpu G4,H100 \
  train:qfft_r1_16k_b8 eval:run:qfft_r1_16k_b8_v93:full
```

报告用同一版本的评分器核对 `base`、`qfft_official` 和新模型，同时给出 AIME24+25+AMC23 合并成绩及相对基座的按题配对 95% CI、MATH-500、GSM8K、平均输出长度与截断率。最终成绩以 `TRAIN_COMPLETE.json` 和五个评测集全部完成为准，和 `qfft_official` 的差距用于判断复现程度。

## 一、本地（WSL2，CPU）：把 1000 题数据做完

项目在 WSL2 Ubuntu 的 `/home/xde10/projects/qwen2.5-7b-math-posttrain`，交互 shell 是 zsh；仓库脚本都是 bash，用 `bash 脚本` 运行即可。`reports/`、`artifacts/`、`data/` 这些本地数据 git 不管，要在这份目录里。

一键（zsh 里直接粘贴）：

```zsh
bash experiments/qfft_v93/local_data.sh
P1_FILL_ARGS='--max-attempts 12' bash experiments/qfft_v93/local_data.sh   # 第 4 步加大额度时
```

它按顺序跑下面五步，出错即停，重跑会续做。解释器用 `P1_PYTHON`（默认 `python3`，需要 pandas、pyarrow、tokenizers；Math-Verify 缺失时用 `.pilot_deps`）。DeepSeek Key 按顺序找：环境变量 `OPENCODE_GO_API_KEY` / `DEEPSEEK_API_KEY`，`--go-db` / `--official-key-file`，再找 Windows 上原来的位置（也会在 `/mnt/c`、`/mnt/d` 下找）和 `~/.config/p1/deepseek.json`；没有 Go Key 而有官方 Key 时直接用官方 API。不想让 Key 进 shell 历史时，在 zsh 里用 `read -rs "DEEPSEEK_API_KEY?DeepSeek key: "; echo; export DEEPSEEK_API_KEY`。Windows 上的旧副本可以继续用 `experiments\qfft_v93\windows\v93_local_data.cmd`。

手动（仓库根目录）：

```bash
python -m experiments.qfft_v93.fetch_benchmarks --sources   # 固定 revision 下载评测集，核对 MATH-500 封存摘要
python -m experiments.qfft_v93.data_build plan              # 1000 题计划：替换与 MATH-500 同题的 s1k11/row/0918，去污
python -m experiments.qfft_v93.data_build status            # 哪些题还没有可用的 DeepSeek 轨迹
python -m experiments.qfft_v93.teacher fill --fallback-official   # 调 API 补齐，可反复运行
python -m experiments.qfft_v93.data_build build --ds-verified     # 生成 r1/、ds/、ds_verified/ 与教师报告
```

- 准入规则（两位教师、1000 题一律相同）：`finish_reason == "stop"`、思考与最终回答都非空、SFT 编码后 ≤ 32768 token。按时间顺序取第一次可用的尝试（v66 → v86 → v91 → v93），不按答案对错挑选，不截断。
- `teacher fill` 只为还没有可用轨迹的题请求 API，每题最多 8 次（`--max-attempts`），effort 依次为 `high,high,max,...`，提示词默认用 v91 的防重复版本。每次尝试单独落盘到 `artifacts/qfft_teacher_v93/attempts/<index>/aNN/`，中断后重跑会接着做。
  - 开始前先读 Key（Go 的 opencode.db 或 `OPENCODE_GO_API_KEY`；官方 Key 文件或 `DEEPSEEK_API_KEY`），找不到就直接退出，一个请求都不发、一次额度都不占。
  - 额度只算"拿到或可能拿到了输出"的尝试（正常返回、流中途断开、进程被打断）；服务端在生成前就拒绝的请求（配额、限流、HTTP 5xx）占一个编号，不占额度。
  - 鉴权或请求格式错误（HTTP 400/401/403/404/413/422）立即停止整个运行；连续 5 次或单次运行累计 20 次失败也会停（服务端故障时不会把每道题的额度都耗光）。退出码 2，修好原因后重跑即可。
  - 还有题用完额度仍没有可用轨迹时退出码为 3，并列出这些题；加大 `--max-attempts` 重跑。
- s1K-1.1 parquet 与 tokenizer 先用本地旧副本；本地副本缺失或哈希不对时，自动改用第 1 步按固定 revision 下载并核对过的 `data/benchmarks_v93/_sources/` 副本（同一 revision、同样的字节）。
- `data_build status` 的预期量级（按 history 里 v66/v86/v91 的记录推算）：v66 最多约 911 题可用（747 + 164 道完整轨迹，按 Qwen tokenizer 超过 32,768 的会被拒），v86 最多 25 题（补全的 27 题里有 2 题超过 32K），v91 最多 62 题；缺口至少 1 题（第 918 题已换成新题，旧轨迹判为 `different_question`）。如果大量出现 `different_question` 或 `missing_reasoning`，说明文件对不上，先不要跑第 4 步。
- 用完额度、轨迹一直跑出窗口的题（2026-11-09 实测还剩 23 题）：先用同一模型、同一提示词的最低思考档追加尝试，`teacher fill --provider official --efforts low --max-attempts N`（DeepSeek 文档：effort 有 low、high、max 三档，medium 映射为 high）；之后仍没有可用轨迹的题用 `data_build build --ds-verified --paired-exclusion` 从两位教师的训练集里一起去掉（记在 manifest 的 `paired_exclusions`，教师报告列出），保证四个训练臂题目相同。不用 R1 轨迹补 DS 臂，也不训练只缺一边的数据。理由和完整命令见 [交接文档第 2 节](../../archive/docs/handover/kiro_ide_onboarding.md)。`--allow-gaps` 只做预览：两位教师题目不同，阶段 `d` 会拒收。2026-11-10 实际处理：4 题补齐，剩下 19 题按用户决定用 OpenR1 原题替换、没有成对剔除（`V93_REPLACE` 固定了这 19 个 raw id；选题看过 Flash 输出，是对上述规则的偏离），经过见交接文档 2.1。
- 教师报告的"DeepSeek 轨迹来源"一节按轮次（v66/v86/v91/v93）、effort、提示词和接口统计每一行轨迹的来历。
- 可选的 `teacher verify` 会为"可用但答案可核对且错误"的题继续采样，`build --ds-verified` 再用第一条答对的可用尝试替换；主矩阵仍用 `ds`。
- `data/qfft1000_v93/teacher_report.md` 给出两位教师在同一批可核对题上的正确率、对错矩阵、长度和反思词频率。参照值：R1 在 s1K-1.1 的 742 道可核对题上答对 62.7%。

## 二、Colab：一键运行

需要 Linux 或 WSL2（Colab CLI 不支持原生 Windows）。以下内容按 google-colab-cli 0.7.5 的源码与 `--help` 核对过（2026-11-08）。

项目和 Colab CLI 都在 WSL2，下面的命令都在仓库根目录、在 zsh 里运行。上传数据（阶段 `d`）默认打包本仓库的 `data/`；数据在别处时用 `--data-root DIR`（或 `export P1_DATA_ROOT=DIR`，写进 `~/.zshrc`）。

**Drive（rclone，默认）**：每台新 VM 上自动做一遍和在 `colab console` 里手动挂载相同的步骤，不需要交互：

1. 把本机的 rclone 配置（`P1_RCLONE_CONF`，默认 `~/.config/rclone/rclone.conf`，必须有 `[mydrive]` 段；远端名用 `P1_RCLONE_REMOTE` 改）上传到 VM 的 `/content/.colab-drive-rclone-upload.conf`，再移到 `~/.config/rclone/rclone.conf` 并设为 600。
2. VM 上没有 rclone 或 fuse3 时才 `apt-get install -y rclone fuse3`（先不更新源；装不上再 `apt-get update` 一次，都有时间上限）。
3. `rclone mount mydrive: /content/mydrive --vfs-cache-mode writes --vfs-cache-max-size 10G --poll-interval 15s --daemon`，用 `mountpoint` 确认。`--poll-interval 15s` 是新增的：命令行上传的文件 15 秒内就能在挂载点看到。
4. 失败时打印输出并释放 VM；要手动排查就加 `--keep`，再 `colab console -s SESSION`（VM 里是 bash，输入密码用 `read -rsp`）。

大文件不经过挂载点：挂载的写缓存要先把整个文件存到 VM 本地盘，等文件关闭后才上传，而一个全参检查点里有一个约 91 GB 的优化器文件，和 122 GB 的本地检查点加起来放不进约 235 GB 的盘。所以检查点、导出权重、模型、环境快照都用 `rclone copy` 直接流式上传（`drive_io.py`），被新镜像取代的检查点用 `rclone purge --drive-use-trash=false` 永久删除（每份 122 GB，进回收站会占 30 天空间）。`--drive colab` 改用 Colab 自带的挂载（`/content/drive/MyDrive`，第一次要在浏览器里同意）。

**Drive 每日上传上限**：Google Drive 每个账号每 24 小时约 750 GiB（[rclone 文档](https://rclone.org/drive/) 的 `--drive-stop-on-upload-limit`）。默认镜像排除优化器状态。撞上限时镜像报错并记录警告，训练继续。最终导出上传失败会使训练阶段失败。

**结果推到 GitHub**：VM 把小体积文本报告推到本仓库的 `colab-reports` 分支（不推 main），每 30 分钟一次（`P1_REPORT_MINUTES`），结束时（成功或失败、在释放 VM 之前）再推一次。我直接从 GitHub 读，不用你下载再上传：

- `v93/latest/`：`status.md`、`stages.json`、各 run 的 `status.json`/`metrics.jsonl`/`saves.jsonl` 等、`evals/results.md` 与各评测的 `summary.json`。
- `v93/sessions/<session>/`：`README.md`（跑了什么、GPU、代码版本、结果）、`session_log_tail.txt`、各阶段日志的末尾、`smoke_summary.txt`；`v93/latest_smoke_summary.txt` 是最近一次冒烟总结。
- 只传文本，单个文件有大小上限，内容里的 token、`Bearer ...`、URL 里的账号密码都会被替换成 `[REDACTED]`。

凭据格式见 [凭据文件模板](../../docs/operations/credential_templates/README.md)。一次性准备：运行 `bash experiments/qfft_v93/colab/setup_wsl.sh`，这个文件不存在时它会提示输入 token（不回显）并按下面的格式写好；也可以在 zsh 里手动写（建议用只授权本仓库、Contents 读写的 fine-grained token）：

```zsh
mkdir -p ~/.config/p1 && chmod 700 ~/.config/p1
read -rs "PAT?GitHub token: "; echo
printf 'https://XDE10:%s@github.com\n' "$PAT" > ~/.config/p1/gh_credentials; unset PAT
chmod 600 ~/.config/p1/gh_credentials
```

launch.sh 把这个文件上传到 VM 的 `/content/gh_credentials`（权限 600，任务结束后删除），git 通过 `credential.helper=store` 读取，token 不出现在命令行、远端地址、日志或推送的文件里。文件不存在时照常运行，只是不推送；`--no-report` 关闭推送。

Windows 上的一键方式（只给 Windows 上的旧副本用）：管理员 PowerShell 里 `wsl --install` 并重启一次；之后都用 `experiments\qfft_v93\windows\v93_colab.cmd`：`setup`（在 WSL 里装 uv 与 Colab CLI、检查仓库和换行符、登录）→ `ping` → `0`、`1`、`2`、`d`、`3`……（参数原样传给 pipeline.sh，例如 `v93_colab.cmd 3 --gpu G4`），另有 `status`、`attach SESSION`。双击不带参数时会提示输入。仓库根目录的 `.gitattributes` 让 `*.sh` 在任何 `core.autocrlf` 设置下都检出为 LF；在 `/mnt/d` 下若提示 Permission denied，用 `bash experiments/qfft_v93/colab/pipeline.sh ...` 运行。

1. 安装与检查：`bash experiments/qfft_v93/colab/setup_wsl.sh`。PATH 上已有的 colab CLI（uv、pipx 或 pip 装的）保留不动，版本不在 `>=0.7.5,<0.8` 范围时只提示；没有才用 uv 安装。它还检查 rclone 配置里有 `[mydrive]` 段、GitHub token 文件是否存在（缺少时提示输入）。CLI 要求 Python ≥ 3.12，uv 会自动下载。
2. 登录：第一次运行任何 `colab` 命令会打印一个 Google 授权链接，在浏览器里同意后把页面显示的授权码粘贴回终端，令牌缓存在 `~/.config/colab-cli/token.json`。`colab auth` 是给 VM 内代码授权 GCP 的，不是 CLI 登录。
3. Drive：默认用 rclone（见上），不需要再做什么；`--drive colab` 时第一次 `colab drivemount` 会打印一个"允许 Colab 访问 Drive"的链接，同意后按回车。
4. 本机自检（只读）：`bash experiments/qfft_v93/check_local.sh`。检查代理、colab CLI 与登录、两个凭据文件及权限、Drive 上的 SwanLab Key 和模型（本机装了 rclone 时）、`P1_PYTHON` 的依赖、`data_build status`、本地 manifest 是否为预览构建、最近一次 Colab 报告。
5. 先花一分钟 CPU 做端到端检查：`bash experiments/qfft_v93/colab/launch.sh --gpu cpu ping:30`（开 VM、挂 Drive、上传、执行、写日志、推报告、释放）。

```zsh
# 阶段 0–2 不依赖本地数据，可以和第一节的造数并行
bash experiments/qfft_v93/colab/pipeline.sh 0   # CPU VM：在 VM 上按固定 revision 下载评测集和模型到 Drive
bash experiments/qfft_v93/colab/pipeline.sh 1   # 冒烟（最坏长度），结束时打印并推送 SMOKE SUMMARY，见下文
bash experiments/qfft_v93/colab/pipeline.sh 2   # 锚点评测
# 本地 data_build build 完成后
bash experiments/qfft_v93/colab/pipeline.sh d   # CPU VM：上传 data/qfft1000_v93（及评测集），核对哈希
bash experiments/qfft_v93/colab/pipeline.sh 3   # DeepSeek 对：QFFT、SFT，各自训完立即评测
bash experiments/qfft_v93/colab/pipeline.sh 4   # LoRA-GRPO + 合并 + 评测
bash experiments/qfft_v93/colab/pipeline.sh 5   # R1 对
bash experiments/qfft_v93/colab/pipeline.sh 7   # CPU VM：重新判分、results.md、图
```

- GPU 默认先要 G4、分不到再要 H100（所有 GPU 阶段）：G4 显存 96 GB 比 H100 的 80 GB 大，每小时约 8.71 个计算单元，约为 H100（18.05）的一半；H100 每小时更快，赶时间时用 `--gpu H100`。费率来自公开整理的数据，开跑前 `colab usage` 会打印当前费率和余额。每种卡先按 high-RAM 规格申请，被拒时再申请标准规格；训练类阶段自己会检查主机内存。
- 冒烟按最坏情况设计，量出来的峰值就是正式训练的上限：
  - 全参：32 条恰好填满 32,768 token 的序列（`smoke_data.py`，由公开的 s1K-1.1 轨迹按训练格式拼接截取，只用于测量），真实配方跑 2 步并本地存一次 122 GB 检查点。
  - GRPO：真实批形状（8 题 × 8 采样），用 vLLM 的 `min_tokens` 把每条 completion 强制生成到 8192 token，跑 2 步并存检查点。
  - 训练类阶段开始前检查主机内存：低于 128 GiB 直接报错（ZeRO-2 offload 的优化器状态约 122 GB），低于 150 GiB 给出警告。
  - 三个冒烟阶段互不依赖：一个失败，同一台 VM 上其余的照样跑完；每次冒烟都先清空上一次的冒烟输出，数字只属于这一次。
  - 结束时（成功或失败）打印一段 `===== P1 SMOKE SUMMARY =====` 到 `===== END P1 SMOKE SUMMARY =====`：机器（GPU、主机内存、CPU 数、本地盘）、各环境的包版本、全参的每步秒数/峰值显存/主机内存/本地存 122 GB 检查点的耗时、生成吞吐、GRPO 的每步秒数与显存，以及失败阶段的报错行。这一段会推到 `colab-reports` 分支的 `v93/latest_smoke_summary.txt`，我直接读；没配置推送时就把这一段原样发给我。同一段也存在 Drive 的 `runs/project1/v93/smoke/latest.txt`，`launch.sh --gpu cpu smoke-report` 可以重新打印。
- 后面可以接 launch.sh 的参数，例如 `pipeline.sh 3 --gpu G4 --wait-min 180`：只用 G4，分不到卡时每 5 分钟重试，最多 3 小时。`--gpu` 支持逗号列表（按顺序尝试）；不认识的卡名会直接报错，因为 CLI 会把未知名称静默当成 A100。
- 单个阶段直接用 `launch.sh`，例如 `launch.sh --gpu G4 eval:run:qfft_ds_v93:full`；`launch.sh --gpu cpu status` 打印所有阶段（运行中的会显示所在 VM 和心跳）、训练进度和评测完成度。
- 每次调用：`colab new [--gpu ...] --high-mem`（被拒时不带 `--high-mem` 再试一次）→ 挂载 Drive（rclone，或 `--drive colab` 时 `colab drivemount`）并检查挂载点 → 上传代码包（大文件按 8 MiB 分片，VM 端重组并核对 SHA-256）→ `colab exec` 执行 `colab/remote.py` → 看到 `P1_RESULT` 后释放 VM（`--keep` 保留，便于 `colab console` 排查）。本地日志在 `reports/colab_logs/<session>.log`，VM 端日志在 Drive `runs/project1/v93/logs/`。

**断线与重复启动**：任务跑在 VM 的 kernel 里，不依赖本地终端。

- 网络断开时，`launch.sh` 每 15 秒重试读取 VM 的 `session_tail.json`。单次读取默认最多 10 秒。连续五次读取失败会报告状态未知，可稍后用 `--attach` 再读。`--attach` 只读取日志，由 VM 按启动配置处理释放。
- 关掉终端或按 Ctrl+C 只是脱离（退出码 75，pipeline.sh 不会接着开下一台 VM）：任务继续，结束后 VM 自己释放。重新跟踪用 `launch.sh --attach <session>`，中止用 `colab stop -s <session>`。
- 每个阶段在 Drive 上有带心跳的标记文件。另一台 VM 想跑同一阶段时，若心跳在 15 分钟内就拒绝，避免两台 VM 同时写一个 run 目录；确认原 VM 已不在时可加 `--force`。

### R1 重要性采样诊断

独立 G4 阶段已完成 64 条 rollout。它使用 R1 QFFT 第 750 步最终权重，复用 16K 长度对照的前 8 个固定题目，每题采样 8 次。直接调用锁定版本 TRL 的训练态 rollout 路径，只做生成、判分和前向重算。下方为复现命令，执行会分配 G4。

```zsh
source ~/.proxy_env.sh
bash experiments/qfft_v93/colab/launch.sh --gpu G4 --session p1-r1-is-diagnostic r1-is-diagnostic
```

验收看 `P1_RESULT=OK` 和 `r1_is_diagnostic: saved 64 completions`。逐 token 数据只保存到 Drive 的 `runs/project1/v93/runs/r1_is_diagnostic/logprobs.jsonl`。阶段日志打印两种权重的聚合结果。再次运行会尝试复用完整作答。把单份产物下载到本机后，可用装有 CPU PyTorch 的 Python 在仓库根目录比较：

```zsh
python -m experiments.qfft_v93.grpo_is_diagnostic compare /path/to/logprobs.jsonl
```

产物保存原始 token ID、采样端与训练端的 logprob、实际 completion mask、组内 advantage、奖励与停止原因，并保留 TRL 当场计算的序列权重供离线核对。比较按奖励与长度分桶，单列 IS 零权重 token 和已有截断掩码。非零权重 token 数与绝对优势加权和是更新信号指标，实测能力仍需独立评测。

### 59 道竞赛留出题的三臂评测

`r1-competition-2026` 阶段读取 Drive 的 `projects/project1/v93/data/r1_competition_2026/r1_competition_2026.jsonl`，校验冻结文件与三份模型标识的 SHA-256，并生成、评分和按题配对。保留结果在 Drive 的 `runs/project1/v93/r1_competition_2026/`。已完成的结果见 [实验总结](../../docs/status/v93_experiment_summary.md)。

### FuRA 标准 LoRA-GRPO 的单卡迁移草案

调研对照包括 [SimpleRL-Zoo](https://arxiv.org/html/2503.18892v3#A2.SS5)（每轮 1,024 题 × 8 作答）、[Dr. GRPO](https://arxiv.org/html/2503.20783v2#A7)（8 张 A100、全参）、[TinyLoRA](https://arxiv.org/html/2602.04118#S5)（同一 Instruct 起点，MATH 配方用 8,523 道题、三轮及学习率搜索）、[FutureMa](https://huggingface.co/FutureMa/Qwen2.5-7B-Instruct-GRPO-Math)（同一 Instruct 起点、单 H100，公开命令与权重，模型卡无留出成绩）和 [thawait](https://huggingface.co/thawait/qwen2.5-7b-math-reasoning-grpo#benchmark-results)（同模型家族，SFT 后的 1,000 步 GRPO 在其 MATH-500 评测上持平）。

采用 FuRA 论文中报告成绩的标准 LoRA 对照。作者在单张 H100 上从 `Qwen/Qwen2.5-7B` 开始，在三个种子上报告了 MATH-500、AMC23 和 AIME24/25。7B 启动脚本的 LoRA 默认值为 `r=64`、`alpha=32`、恒定学习率 `1e-4`，50 步，每步 32 题 × 8 作答。训练源码使用 `qwedsacf/competition_math` 的前 7,500 条、组内去均值、按作答长度归一化，每批仅更新一次。本地 vLLM 的实际 `max_tokens` 是 1,024，模型上下文是 2,048。[论文表 4 和附录 D.3](https://arxiv.org/abs/2605.22869) · [固定提交的 7B 脚本](https://github.com/yequanzhao/FuRA/blob/9c68151f3cf81bf6543bf918c011bfd98cf81220/rl/run_rl_7B.sh) · [训练源码](https://github.com/yequanzhao/FuRA/blob/9c68151f3cf81bf6543bf918c011bfd98cf81220/rl/run_rl.py)

本项目的[复现配置](configs/grpo/grpo_qfft_r1_fura_lora50.json)移植上述 rank、alpha、AdamW、学习率、50 步预算和按作答归一化的 `grpo` 损失。从 R1 QFFT 第 750 步权重出发，使用已获批并清理的 1,551 道题、每步 8 题 × 8 作答、16K 生成上限、Math-Verify 0/1 奖励、截断掩码和 `token_mask`。TRL 使用其单轮 GRPO 实现，设对称裁剪 0.2，源码在一次更新内部直接用未裁剪的比率。原版每步独立随机抽题，TRL 沿用项目的遍历式洗牌。16K 预算下，单卡只采用现有 64 条作答的批大小。此轮检验公开优化配方能否迁移到项目权重。

获得本轮 GPU 训练授权后，从 WSL 仓库根目录启动。训练以最近的 16K 实测步速估算约 15.4 个 G4 小时，另计评测。固定使用第 50 步合并权重与原始 R1 在项目竞赛 100 题上的已有作答配对，使用当前判分器，报告差值、置信区间和输出长度。这 100 道题已用于早期诊断，成绩用于与项目既有结果对标。若没有可信的增益，以已验证的 QFFT 成绩和本轮 GRPO 结果收尾。

```zsh
source ~/.proxy_env.sh
export PATH="$HOME/.local/bin:$PATH"
bash experiments/qfft_v93/colab/launch.sh --gpu G4 --session p1-r1-fura50 \
  grpo:grpo_qfft_r1_fura_lora50 r1-fura50-eval
```

训练验收检查第 50 步 `GRPO_COMPLETE.json`、`merged/p1_model_identity.json` 和三个竞赛子集的 `generation_status.json`。如 VM 已被回收，从完整 Drive 镜像用 `grpo:grpo_qfft_r1_fura_lora50:resume` 继续，并复用原评分阶段。

本地核对两份已批准的源文件、1,551 道题的顺序和题池指纹，均与前一轮 R1 一致。TRL 1.17.1 配置检查通过。微型 Qwen2 在 CPU 上实跑两次 AdamW 更新并写出完整镜像，使用模拟奖励和缩小的采样设置。评测阶段用模拟作答验证了 100 题配对和独立报告写入。真实单卡 7B 训练待本轮授权。

## 三、断点与续跑

| 对象 | 本地（VM SSD） | Drive（持久） | 新 VM 上怎么继续 |
|---|---|---|---|
| 全参训练 | `checkpoint-N`，包含优化器及训练状态 | `ckpt_mirror/checkpoint-N`，`MIRROR_COMPLETE.json` 标记 `weights_only` 或 `full_state` | 仅完整状态参与续训。旧镜像按文件检查，完整镜像可下载续跑 |
| GRPO | adapter 检查点（约 1 GB），每 10 步 | 同上，保留 3 个 | 同上 |
| 评测生成 | 每条请求结束即追加 `raw.jsonl` | 每分钟同步一次 | 合并本地与 Drive 记录，只补缺失的 (题, 第几次采样) |
| 判分 | — | `scored.jsonl`（带判分器版本） | 增量判分，`--rescore` 全部重判 |
| 环境 | `/content/p1/venvs/{infer,rl}` | `envcache/venvs/*.tar` 快照 + 冻结版本清单 | 约 1 分钟恢复 |

- 镜像先写 `.partial`，发布成功后写完成标记。完整镜像与轻量镜像各自保留，新的轻量镜像不会淘汰旧的完整镜像。`--drive colab` 时使用限速复制。
- 原 VM 仍可用且会话空闲时，用原来的 `--session` 和训练阶段重启，同一 VM 会使用完整本地检查点。会话繁忙时用 `--attach`。
- 原 VM 已消失且只剩轻量镜像时，可显式启动 `warm-start:<配置>:<来源 run>:<新 run>`。例如来源是 `qfft_r1_16k_b8_restart_v93` 时：

  ```bash
  bash experiments/qfft_v93/colab/launch.sh --gpu G4,H100 \
    warm-start:qfft_r1_16k_b8_restart:qfft_r1_16k_b8_restart_v93:qfft_r1_16k_b8_warm_v93
  ```

  入口选择来源最新可用的轻量镜像并在新 run 中固定来源。仅下载模型权重和模型配置，使用固定底座的 tokenizer。数据 SHA 和训练配方须与来源一致。若在第 125/750 步恢复，Trainer 在新 run 从 0 计数、最多更新 625 次，第一个更新用原 cosine 曲线在 125 步的学习率，后续沿同一曲线走到第 750 步。AdamW 状态重建，随机种子按配置重新初始化采样器，数据顺序会重新开始。运行身份、状态和完成报告的 `warm_start` 字段记录来源、原始偏移、剩余步数及学习率策略。结果需标为权重 warm-start。
- 训练日志（`metrics.jsonl`）、GRPO 的判分记录和各阶段日志都先写 VM 本地，再每 30–60 秒整体替换 Drive 上的副本：挂载点上一直打开着的文件要到关闭时才会上传。
- run 目录里的 `run_identity.json` 记录配方与数据 SHA；配方或数据变了会拒绝在同名 run 上续跑。
- 训练完成后导出 bf16 safetensors 到 `runs/<run>/final/`，写 `TRAIN_COMPLETE.json`，删除镜像，并把权重留在本地 SSD 上供同一 VM 的评测直接使用。

## 四、评测口径（[configs/pipeline.json](configs/pipeline.json)）

- 所有模型用同一组 prompt token：system `Please reason step by step, and put your final answer within \boxed{}.` + user 题目，用基座 tokenizer 渲染。
- vLLM，bf16，原生 RoPE，`max_model_len 32768`（Qwen2.5-Math 按其配置用 4096），`max_tokens = max_model_len − prompt`，temperature 0.6，top_p 0.95，每个 (题, 第几次) 有固定种子。
- 采样数：MATH-500 4、AIME24 16、AIME25 16、AMC23 8、GSM8K 1。
- 判分（`grading.py`）：只看最后一个 `<|im_start|>answer`（或 `</think>`）之后的最终段，取最后一个 `\boxed{}`，用 Math-Verify 0.9.0 比较；判分进程有硬超时，超时和异常单独计数，不算答错。主指标不用 LLM 复核。
- 指标（`metrics.py`）：mean@n ± 按题 bootstrap 95% CI、无偏 pass@k、maj@n、平均 token（分对错）、截断率、尾部重复率、反思词触发率、按 MATH 难度分层、不加强制收尾的"思考预算曲线"；`score.py --compare A:B` 做按题配对 bootstrap。

## 五、文件

| 文件 | 作用 |
|---|---|
| `formats.py` | prompt 与 response 布局、SFT/QFFT 编码、最终段切分（唯一来源） |
| `grading.py` | 最终段判分、带超时的进程池、多数投票聚类 |
| `metrics.py` | pass@k、bootstrap、配对检验、长度与行为统计 |
| `decontam.py` + [configs/decontam_review_v93.json](configs/decontam_review_v93.json) | 13-gram + 相似度去污，17 对已人工复核 |
| `fetch_benchmarks.py` | 固定 revision 与 SHA-256 下载评测集（本地或 VM 上的 `benchmarks` 阶段） |
| `smoke_data.py` | 冒烟用的最坏长度序列（每条恰好填满训练窗口，只用于测量） |
| `data_build.py` / `teacher.py` | 1000 题计划、准入、API 补齐、两份训练集与教师报告 |
| `train_sft.py` | 全参 SFT/QFFT（DeepSpeed ZeRO-2 offload、Liger、Drive 镜像续跑） |
| `generate.py` / `score.py` / `analyze.py` | 生成、判分汇总、图表 |
| `grpo.py` / `merge_lora.py` | TRL GRPO + LoRA + 同卡 vLLM、合并导出 |
| `checkpointing.py` / `telemetry.py` / `models.py` / `common.py` | 镜像与恢复、SwanLab、模型准备、公共工具 |
| `drive_io.py` | 本地盘与 Drive 之间的大文件传输：rclone 命令行（流式上传、永久删除旧镜像）或限速复制 |
| `colab/launch.sh` / `colab/pipeline.sh` / `colab/remote.py` / `colab/stages.py` | 一键调度 |
| `colab/report_push.py` / `colab/smoke_report.py` | 把文本报告推到 `colab-reports` 分支；冒烟总结 |
| `local_data.sh` | WSL2/Linux 上的本地造数一键入口（五步） |
| `check_local.sh` | 本机只读自检（代理、CLI、凭据与权限、Drive、解释器、数据状态、最近的 Colab 报告） |
| `colab/setup_wsl.sh`、`windows/v93_colab.cmd`、`windows/v93_local_data.cmd` | WSL 一次性安装、检查（rclone 配置、GitHub token 文件）与登录；Windows 下的 Colab 阶段入口与本地造数入口 |
| `envs/*.txt` | 训练附加包、推理与 RL 虚拟环境（2026-11-08 在 PyPI 核对过版本约束）；`envs/resolved_20261109/` 是按 Python 3.12/3.13 解析出的完整依赖，仅供参考 |

## 六、已验证与未验证

- CPU 上已实跑：模板与官方 S1_QFFT 数据逐字一致（1000/1000）；SFT/QFFT 编码不变量；判分器 10 个边界用例；微型 Qwen2 上的全参训练、模拟 VM 丢失后从 Drive 镜像恢复并续跑到结束、SFT 格式、身份校验；假引擎生成的按采样粒度续跑与判分汇总；TRL 1.17.1 GRPO + LoRA + 奖励 + 镜像 + 恢复 + 合并；合成数据上的 plan/status/fill/build 全流程；launch.sh 与 remote.py 的调度流程（按 google-colab-cli 0.7.5 行为写的模拟 CLI）；冒烟序列用真实 Qwen tokenizer 生成，QFFT 32 条、SFT 4 条都恰好 32,768 token。
- 模拟测试在 [tests/v93_harness/](../../tests/v93_harness/README.md)：`bash tests/v93_harness/run_all.sh`（8 组 214 项：launcher 场景、Drive 传输、冒烟总结、teacher、固定输入、data_build、setup_wsl.sh、check_local.sh），在工作区的临时快照上运行。
- G4 上已实测（2026-11-09 阶段 1 冒烟，数字见[交接文档第 1 节](../../archive/docs/handover/kiro_ide_onboarding.md)）：32K 最坏长度全参 142 s/步、显存峰值 28.9 GiB 已分配、主机 RSS 145.3 GiB、本地存检查点 113.5 GiB 用 386 s；GRPO 最坏长度 476 s/步、66.6 GiB。还没测：真实数据下的 s/step（每个微批变短后 ZeRO-2 offload 的梯度搬运开销）、正式评测的生成吞吐、113.5 GiB 镜像的上传耗时、H100 上的 GRPO 显存。
