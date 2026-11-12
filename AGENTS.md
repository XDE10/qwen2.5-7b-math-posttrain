# 项目一：协作约定与入口（AGENTS.md）

适用范围：本仓库根目录及子目录。2026-11-08 目录重构 v2 后重写；本文件只保留**长期有效的约定、当前状态和阅读入口**，逐版本历史已迁移到 [docs/status/history.md](docs/status/history.md)。

## 项目是什么

Qwen2.5-7B-Instruct 的数学后训练项目：SFT 取得实测提升 → GRPO 继续提升，与 Qwen2.5-Math-7B-Instruct 对标，形成可写简历的结果链；另含「同一道题：原生长轨迹 vs 教师精炼轨迹」的质量/长度/成本对照，以及 QFFT 造数链。不能预先保证涨分。

固定底座 `Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`。不要切回 R1-Distill，不自行更换数据集。

## 阅读入口（按顺序）

1. 本文件 → [docs/status/current_status.md](docs/status/current_status.md)：当前目标、实测结果和下一步。
2. [README.md](README.md)：目录职责、数据边界、命令。
3. 当前任务读 [experiments/qfft_v93/README.md](experiments/qfft_v93/README.md) 与 [三周冲刺方案 v93](docs/blueprints/三周冲刺方案_v93.md)；v26 的[训练配置](configs/project1_aimo_sft_v26.json)、[实验包](experiments/aimo_sft_v26/) 与[启动入口](tools/aimo/start_aimo_sft_v28.py) 只作历史。
4. 需要某个版本（v1…v93）的细节时再查 [docs/status/history.md](docs/status/history.md)；旧交接材料在 [archive/docs/handover/](archive/docs/handover/)，迁移路径见 [docs/migration/handover_cleanup_20261112.json](docs/migration/handover_cleanup_20261112.json)。
5. 不需要先遍历数据、缓存和全部历史报告。

## 当前主线（摘要，v93）

- 数据：1000 题（s1K-1.1 的 974 题 + v57 的 6 题替换 + v93 的 20 题替换：与 MATH-500 同题的 `s1k11/row/0918`，以及 19 道 DS 轨迹始终超出 32K 窗口的题），两位教师：s1K-1.1 的 R1 轨迹与自建 DeepSeek-V4.1-Flash 轨迹；准入为"正常结束 + 思考与回答非空 + SFT 编码 ≤ 32768"，按时间顺序取第一次可用尝试，不按对错过滤，不截断。例外：那 19 道替换题是看了 Flash 输出才选定的（对预注册规则的偏离，报告结果时必须写明，见 [history.md](docs/status/history.md) v93-P8）。由 `experiments/qfft_v93/data_build.py` 与 `teacher.py` 在本地生成 `data/qfft1000_v93/`。
- 训练：全参 SFT 与 QFFT（官方 `qwen_qfft` 格式：只训 response、无 chat 标记、无 EOS），LR 1e-5、6 epoch、cosine、warmup 10%、wd 1e-4、AdamW(0.9, 0.999)（与 QFFT 官方 yaml 相同），有效 batch 16、32K 窗口、DeepSpeed ZeRO-2 + CPU AdamW offload、Liger；矩阵为教师 × 格式 2×2。
- RL：在合并后的 QFFT 检查点上做 LoRA-GRPO（r32、LR 1e-5、150 步、8 题 × 8 采样、`dapo` 损失、β=0、ε_high 0.28、截断掩码、最终段 0/1 奖励），题池为已批准的 920 + 637 题。
- 评测：所有模型同一组 prompt token、原生 RoPE、`max_model_len 32768`、temperature 0.6、top_p 0.95；AIME24/25 各 16 次、AMC23 8 次、MATH-500 4 次、GSM8K 1 次；只判最终段最后一个 `\boxed{}`（`experiments/qfft_v93/grading.py`），主指标不用 LLM 复核。
- 调度：Colab CLI（Linux/WSL2）`experiments/qfft_v93/colab/pipeline.sh <阶段>`；所有状态在 Drive `projects/project1/v93`、`models`、`runs/project1/v93`，重跑即续。
- 环境：项目与 Colab CLI 都在 WSL2 Ubuntu：`/home/xde10/projects/qwen2.5-7b-math-posttrain`（Windows 下为 `\\wsl.localhost\Ubuntu\home\xde10\projects\qwen2.5-7b-math-posttrain`），交互 shell 是 **zsh**；给用户的命令要兼容 zsh，仓库脚本都是 bash，用 `bash 脚本` 运行；新终端先 `source ~/.proxy_env.sh`（GitHub、Colab CLI、rclone 走代理）。Colab VM 里（`colab console`）是 **bash**。旧的 Windows 目录 `D:/Users/Mr.Z/Desktop/项目一_7B数学后训练` 与 `D:/anaconda3/envs/genai-test/python.exe` 只是旧位置。用户自己操作 Colab（不给我登录）。GPU 默认先 G4（RTX PRO 6000 Blackwell 96GB，约 8.71 计算单元/小时）后 H100 80GB（约 18.05）；G4 high-RAM 规格冒烟实测主机内存 177 GiB、48 CPU、本地盘 236 GiB。
- Drive：用户用 rclone，远端 `mydrive:` 挂到 VM 的 `/content/mydrive`（`--vfs-cache-mode writes --vfs-cache-max-size 10G`，launch.sh 另加 `--poll-interval 15s`），这是 launch.sh 的默认（`--drive rclone`，配置文件 `P1_RCLONE_CONF`，默认 `~/.config/rclone/rclone.conf`）；`--drive colab` 才用 Colab 自带挂载 `/content/drive/MyDrive`。大文件（检查点、导出权重、模型、环境快照）走 rclone 命令行（`experiments/qfft_v93/drive_io.py`），不经挂载的写缓存；被新镜像取代的检查点永久删除（不进回收站）。Drive 每个账号每 24 小时约 750 GiB 上传上限：全参每 2 个 epoch 存一次（`save_steps` 126，每个 run 约 241 GiB），24 小时内最多 2 个全参 run，撞上限时 rclone 立即报错（`--drive-stop-on-upload-limit`）。
- 结果回传：VM 把小体积文本报告推到本仓库的 `colab-reports` 分支（永远不推 main），每 30 分钟一次并在结束时一次；凭据是用户本机的 `~/.config/p1/gh_credentials`（一行 `https://USER:TOKEN@github.com`，launch.sh 上传到 VM 的 `/content/gh_credentials`，用完删除）。我从 GitHub 直接读 `v93/latest/` 与 `v93/sessions/<session>/`，不需要用户下载再上传。
- 冒烟（阶段 1）结束时打印 `P1 SMOKE SUMMARY` 一段（也在 Drive `runs/project1/v93/smoke/latest.txt`，并推到 `colab-reports` 分支的 `v93/latest_smoke_summary.txt`）。2026-11-09 已在 G4 跑通（41 分钟）：32K 最坏长度全参 142 s/步、显存 28.9 GiB、主机 RSS 145.3 GiB / 177 GiB，GRPO 476 s/步、66.6 GiB。
- 23 道缺口题（DeepSeek 在 high/max 下的轨迹一直跑出 32K 窗口）：预注册方案是先用 `reasoning_effort=low` 追加尝试、仍缺的成对剔除。实际（2026-11-10）：4 题补齐；剩下 19 题用户决定保留 1000 题、不做成对剔除，改为用已批准的 OpenR1 原题替换。不用 R1 补 DS，不训练只缺一边的数据。预览构建（`--allow-gaps`）不能上传，阶段 `d` 会拒收。
- DS 思考段里关于隐藏参数 "oververbosity 5" 的句子由 `data_build.py` 的 `META` 删除，原文留在每行 `teacher_detail.meta_edits`。
- **进度（2026-11-12）**：阶段 0、1、d、2 与 `qfft_ds_v93` 的正式训练和评测已完成。当前优先跑 R1、16K、batch 8 的自训对照，正式 GRPO 仍待运行。实测成绩和最新运行进度见 [docs/status/current_status.md](docs/status/current_status.md) 与 `colab-reports`。

## 仓库约定（必须遵守）

- **不删除历史文件。** 被取代的脚本/测试/配置/报告统一移入 `archive/`，不删除；`archive/` 是**只读历史快照**，不保证可直接运行（见 [archive/README.md](archive/README.md)）。
- 现行主线 = 从活跃入口出发的**导入闭包 + 文档明确引用**的代码；不在闭包里的不要擅自提升为主线，也不要为了「整洁」删除。
- 改代码前先看 `git status` 和实际文件；文档中的路径必须真实存在（归档路径要写明 `archive/` 前缀）。
- 数据准入：只看对应 manifest 的 `approved_training_files` / `approved_evaluation_files` 与逐行授权；候选 ≠ 批准，AI 审核/答案等价 ≠ 人工推导金标。不能手改批准字段或哈希绕过准入。
- 不重复全量去污/审计/扩量；验证聚焦当前报错和身份关键项，避免防御性检查阻碍执行。
- 不擅自启动 GPU 任务、不擅自发起付费/订阅 API 请求、不自动扩量或入训；这些需要用户明确授权，并把「已授权范围」写进报告。
- 凭证（SwanLab / DeepSeek Key）只放私密文件，不打印、不回显、不入库、不写进交接文档或公开运行包。
- 大体积数据/证据/报告只在本地：`data/`、`artifacts/`、`reports/`、`archive/data/`、`archive/reports/`、`.pilot_deps/`、`*.zip`、`.pytest_tmp/` 不入 git（各目录保留 README 例外）。详见 [.gitignore](.gitignore)。唯一例外是 Colab 推到 `colab-reports` 分支的小体积文本报告（有大小上限、去掉密钥，不进 main）。
- 文档与代码保留来源和验证记录；**CPU 通过不等于 GPU 结果**，「已实现 / CPU 通过 / 真实 GPU 跑通 / 能力提升」要分开陈述。
- 迁移/清理/重构的每次移动都留映射与记录（`docs/migration/`），移动后同步 `docs/metadata/migration_manifest.json` 并跑 `python tools/verification/verify_migration.py`。
- WSL2 下多步或带嵌套引号的操作写进 `.sh` 再用 `bash` 执行。远程短步骤设超时并流式输出，长任务用 `launch.sh --attach` 重连。改动 Colab 调度、教师调用或 Drive 传输后，运行 `bash tests/v93_harness/run_all.sh` 中对应的套件。
- **响应与防误杀硬约束**：(1) Agent 执行任何外部/Git 网络命令（如 `git fetch origin colab-reports`）必须加超时保护（`timeout 10s`）或先检查本地日志，严禁无超时阻塞让用户等待；(2) 给用户命令前必须核对精确参数（`launch.sh` 为 `--session` 而非 `-s`，`colab-cli 0.8.0` 为 `colab sessions`），绝不触碰用户并发运行的其他会话（如 `opd`、`p3`）；(3) `remote.py` 通过 `active_run_id` 令牌防止前一次报错的 120s 延迟 `runtime.unassign()` 销毁同一 VM 上复用的新任务，`launch.sh` 默认 `POLL_S=15` 秒且支持直接复用已存在的 `--session`。

## 目录地图

| 目录 | 职责 |
|---|---|
| `configs/` | 现行工程配置，仅 `project1_aimo_sft_v26.json`；历史配置在 `archive/configs/` |
| `experiments/aimo_sft_v26/` | 现行 AIMO SFT 实验包（train/evaluate/export/telemetry 等） |
| `scripts/{training,evaluation}/` | 6 个现行命令行/worker 入口；其余历史脚本在 `archive/scripts/` |
| `src/{data,evaluation,losses,training}/` | 16 个现行库模块（批准检查、评分器、损失参考、HF 训练入口） |
| `tests/` | 25 个现行 pytest 文件（172 个用例），按 data / evaluation / losses / training / qfft 分组；`tests/v93_harness/` 是 v93 调度、教师、Drive 的模拟测试（`bash tests/v93_harness/run_all.sh`，214 项，非 pytest） |
| `tools/{aimo,evaluation,qfft,review,verification}/` | 现行工具：AIMO 启动、评测、QFFT 造数与监控、审查包、迁移校验 |
| `data/` | 本地数据（`aimo_sft_v26/`、`math/`、`portable_release/`、`tokenizers/`），git 忽略 |
| `artifacts/` | 造数与评测原始证据，git 忽略 |
| `reports/{current,runs}/` | 现行报告（7 个目录），git 忽略（仅 README）；历史报告在 `archive/reports/` |
| `docs/` | `status/`、`blueprints/`、`migration/`、`metadata/`、`operations/` |
| `archive/` | 只读历史快照：`configs/ data/ docs/ experiments/ mapping/ reports/ scripts/ src/ tests/ tools/` |
| `.pilot_deps/` | 数学验证器（math_verify 0.9.0 等）纯 Python 运行时依赖，git 忽略 |

## 路径、启动与恢复

- 本地命令一律在仓库根目录执行（WSL2）：本地造数 `bash experiments/qfft_v93/local_data.sh`（解释器 `P1_PYTHON`，默认 `python3`，需要 pandas、pyarrow、tokenizers），`pytest` 用同一解释器。`experiments/qfft_v93/windows/*.cmd` 只给 Windows 上的旧副本用。`.gitattributes` 固定 v93 的 `*.sh` 为 LF、`*.cmd` 为 CRLF（`tests/v93_harness/` 全部 LF），不要改回。本机只读自检：`bash experiments/qfft_v93/check_local.sh`。
- Drive 持久目录：v93 用 `projects/project1/v93`（数据、代码包、环境快照）、`models`（与 v26 共用，基座目录不变）、`runs/project1/v93`（runs、evals、stages、logs、figures）；VM 本地工作区 `/content/p1`。这些都是相对 Drive 根的路径；VM 上的 Drive 根是 `/content/mydrive`（rclone）或 `/content/drive/MyDrive`（`--drive colab`）。v26 的工作副本 `/content/workspaces/aimo-sft-v26` 与结果 `runs/project1/aimo_sft_v26` 只作历史。
- v93 起不再用 VS Code 远程终端：在 Linux/WSL2 上装 Colab CLI，用 `experiments/qfft_v93/colab/launch.sh` 或 `pipeline.sh` 启动；每次调用新建 VM、挂载 Drive、上传代码包、跑阶段、结束后释放。任务跑在 VM 的 kernel 里：断线、关终端或 Ctrl+C 不会中止它（`launch.sh --attach <session>` 重新跟踪，`colab stop -s <session>` 才是中止）。`launch.sh --gpu cpu status` 查看所有阶段与训练进度。
- v26 的旧入口 `python3 /content/drive/MyDrive/projects/project1/start_aimo_sft_v28.py` 只用于复现历史。
- **有任务在跑时先看最新状态（`status` 阶段或 Drive 上的 `runs/project1/v93/stages/`），不重复启动同一阶段**；阶段标记带 5 分钟心跳，另一台 VM 在心跳 15 分钟内会拒绝同一阶段（确认原 VM 已不在才用 `--force`），run 目录还有身份校验。

## 已知兼容项必须落实到入口

- torchao 0.10.0 与当前 PEFT 不兼容：v27 自动把已安装且低于 0.16.0 的版本定向升级到 0.18.0，用 `--no-deps`，不连带升级 torch。
- Transformers 5.20.2 不接受 `warmup_ratio`：v27 改用整数 `warmup_steps`（本轮 22 步），在模型加载前验证 TrainingArguments；长度分组同时适配旧 `group_by_length` 与新 `train_sampling_strategy`。
- 历史 `pyroaring<1.0.4` 曾解决 SwanLab 安装问题；按具体错误处理，不无条件重装整套依赖。
- SwanLab 必须在线、详细记录、自动认证：Key 存 Drive 根下的 `.project1_credentials/swanlab.json`（字段 `SWANLAB_API_KEY`；rclone 挂载时即 `/content/mydrive/.project1_credentials/swanlab.json`），读取后注入子进程环境，不要求用户每次输入，不静默转 offline。
- 只补缺失依赖和已证实的兼容修复，保留预装 Python/PyTorch/Transformers。
- v93 环境分三份：训练沿用 Colab 镜像自带的 torch/transformers（与系统 CUDA 一致，DeepSpeed 在运行时编译 CPU-Adam），只补 `envs/train-extra.txt`；推理（vllm 0.33.0）与 RL（vllm 0.32.0 + trl 1.17.1）各用独立 uv 虚拟环境，首次装好后快照到 Drive 并冻结版本清单。不要把 vLLM 装进训练用的系统环境。
- Colab CLI（google-colab-cli 0.7.5 已按源码核对）不支持原生 Windows，launch.sh 在 WSL2/Linux 上运行；CLI 需要 Python ≥ 3.12（`uv tool install` 自动处理）。CLI 登录是"打开链接、粘贴授权码"，`colab auth` 只给 VM 内代码授权 GCP。`colab exec --timeout` 是本地客户端的总时限（默认 30 s），统一传 86400；超时或断线不会中断 VM 上的任务。未知的 `--gpu` 名称会被 CLI 静默当成 A100，launch.sh 已做校验。

## 数据与评测口径

- 训练/评测范围以对应 manifest 的 approved 列表及逐行授权为准；旧 1500 raw / 1500 compact 是同一批题的两臂，不是 3000 道题。
- 根目录曾有的 4810 条 RL 候选文件是旧候选；E0 每题 4 次 rollout 后获批的 920 条在 Drive `runs/project1/rl_approved_v8_n4`。不把该通过率当新 SFT 模型的通过率。
- 637 题是 E0 四次全错的条件探针，不是独立最终测试。
- v93 起评测与 RL 奖励都用 `experiments/qfft_v93/grading.py`（只判最终段最后一个框）；`src/training/math_reward_v2.py` 与 strict-v3 仅用于复现 v93 之前的历史数字。答案等价不是逐步证明正确，解析失败、超时、判分异常单列，不算答错。
- 不按评测结果挑 checkpoint：v93 预注册"用最后一个 epoch / 最后一步"；旧 2048 预算、旧 8K+256 收尾预算的成绩不能直接充当 v93 基线。
- v93 评测口径是原生 RoPE、`max_model_len 32768`、`max_tokens = 32768 − prompt`（Qwen2.5-Math 用 4096）；v26 的 32768 生成 + 40960 上下文 + YaRN factor 4 只属于历史结果。

## 历史与工作方式

- 逐版本（v1…v92）的日期、实测数字、授权范围和结论都在 [docs/status/history.md](docs/status/history.md)；那是**历史记录**，冲突时以本文件 + `docs/status/current_status.md` + 最新报告为准。
- 工作方式：优先执行，给完整可复制的重连/挂载/恢复命令，明确结束标记与何时可停 GPU；不重复索要已授权的许可；不把历史方案或 smoke 写成已完成的正式训练结果。
- 维护本文件时保持简短：只放长期约定与指向，不追加逐版本流水账（流水账进 `docs/status/history.md`）。
