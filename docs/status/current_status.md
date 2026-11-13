# 项目一当前状态（2026-11-13，v93）

> 本页只写「现在做到哪」的摘要；逐版本历史与精确数字见 [history.md](history.md)，仓库约定见根目录 [AGENTS.md](../../AGENTS.md)，v93 方案见 [三周冲刺方案_v93.md](../blueprints/三周冲刺方案_v93.md)。WSL2 启动和恢复命令见 [v93 运行手册](../../experiments/qfft_v93/README.md)。

## 目标与主线

Qwen2.5-7B-Instruct → 全参 SFT / QFFT（两位教师）→ LoRA-GRPO，与公开 checkpoint 和 Qwen2.5-Math-7B-Instruct 在同一评测口径下比较，形成可写简历、可复现的结果链。**不能预先保证涨分。**

- 固定底座：`Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- 实现：[experiments/qfft_v93/](../../experiments/qfft_v93/README.md)。研究问题、配方、判定规则已预注册在方案文档里；之后的偏离记在方案第 10 节。

## Colab 上已跑通（用户在 WSL2 上用 Colab CLI 运行，结果由 VM 推到 `colab-reports` 分支）

- 2026-11-09：`ping:30`（CPU）、阶段 0（CPU，约 11 分钟）：5 个评测集与 4 个模型（base、qwen_math、qfft_official、sft_repro）已在 Drive。
- 2026-11-09：阶段 1 最坏长度冒烟（G4 = RTX PRO 6000 Blackwell 96 GB，主机内存 177 GiB，48 CPU，41 分钟，`colab-reports` commit `d6a9608`）：全参 16 × 32,768 token 每步 148/142 s，显存峰值已分配 28.9 GiB / 保留 42.6 GiB，主机 RSS 最高 145.3 GiB；本地存检查点 113.5 GiB 用 386 s；评测 24 条全部保存，vLLM 启动 217 s；GRPO（64 条都强制到 8192 token）每步 476/474 s，显存 66.6 GiB；SwanLab 在线。这些是冒烟（2 步）的数字，不是正式训练结果。
- 2026-11-10：修好 `launch.sh` 挂载脚本的 f-string 错误（PR #9，`40f2b68`）后，用户在 CPU VM 上 `ping:30` 真实挂载成功（会话 `p1-1110-135452`）。
- 2026-11-10：阶段 d（CPU VM，会话 `p1-1110-230033`，约 7 分钟）把 `data/qfft1000_v93` 与评测集传到 Drive `projects/project1/v93/data/`；本机 `rclone check` 核对 Drive 上的文件与本地逐个相同。
- 2026-11-11：**阶段 2（4 个锚点模型 × 5 个评测集，共 4 × 4,599 条生成）全部完成**（会话 `p1-1110-234216`），结果已归档至 Drive 与 `colab-reports`。
- 2026-11-11～11-12：**阶段 3 上半场 `train:qfft_ds`（378 步全参 QFFT，DeepSeek-V4.1-Flash 轨迹）+ `eval:run:qfft_ds_v93:full`（4,599 条全量评测）全部完成**。会话 `p1-1111-231532` 从带优化器状态的 checkpoint-126 续训到第 378 步，导出最终权重并评完五个题集。真实训练中每步约 40～66 s，显存峰值 28.8 GiB。排除优化器状态的 checkpoint-252 镜像实测上传 28.4 GiB，用时 318 s，未用于本轮恢复。
- 2026-11-12：旧 R1 对照 `qfft_r1_16k_b8_v93` 停在第 209/750 步，Drive 留有第 125 步轻量镜像。原 VM 消失后，用户授权从底座用独立 run 重跑。[旧会话报告](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/aca5248/v93/sessions/p1-1112-183656/README.md)
- 2026-11-13：**R1 16K batch 8 对照 `qfft_r1_16k_b8_restart_v93` 完成 750 步及五项评测**。G4 会话 `p1-r1-16k-restart-1112` 从固定底座开始，完成 6 个 epoch，未使用 warm-start。最终权重 05:49 写完，4,599 条生成全部判分，09:17 结束（UTC+8）。[会话报告](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/71a78e7c69989e7ced436a8e228356ccd4402737/v93/sessions/p1-r1-16k-restart-1112/README.md)

## R1 五项实测成绩（v93 统一口径）

四个模型使用相同题目、生成配置和判分器。各题先对多次生成取平均，再按题等权汇总。竞赛 100 题为 AIME24（30 题）、AIME25（30 题）及 AMC23（40 题）。公开 QFFT 为作者发布的 `lwl-uestc/QFFT-S1-7B`。[五项评分明细](https://github.com/XDE10/qwen2.5-7b-math-posttrain/tree/71a78e7c69989e7ced436a8e228356ccd4402737/v93/latest/evals)

| 评测 | 底座 | 公开 QFFT | Flash QFFT | 本轮 R1 QFFT | R1 比底座（百分点） |
|---|---:|---:|---:|---:|---:|
| **竞赛 100 题** | **26.06%** | **35.56%** | **19.31%** | **33.63%** | **+7.56** |
| AIME24 (mean@16) | 11.25% | 20.83% | 5.00% | 18.54% | +7.29 |
| AIME25 (mean@16) | 7.71% | 16.04% | 3.96% | 14.79% | +7.08 |
| AMC23 (mean@8) | 50.94% | 61.25% | 41.56% | 59.06% | +8.13 |
| MATH-500 (mean@4) | 76.80% | 81.10% | 67.00% | 80.50% | +3.70 |
| GSM8K (mean@1) | 91.58% | 92.27% | 88.70% | 91.43% | −0.15 |

竞赛 100 题按 `sample_id` 配对，使用项目的 bootstrap 实现重采样 10,000 次（seed 0）。相对底座的 **+7.56 个百分点，95% CI [+3.88, +11.44]**，达到「点估计至少 +5 个百分点且区间下界高于 0」的目标。区间仍包含低于 +5 的增益。相对 Flash 为 **+14.31**，95% CI **[+9.94, +18.88]**。相对公开 QFFT 为 **−1.94**，95% CI **[−4.06, +0.06]**。[配对方法](../../experiments/qfft_v93/metrics.py)

本轮 R1 平均输出 token / 截断率依次为 AIME24 **14,651 / 10.42%**、AIME25 **11,125 / 2.50%**、AMC23 **7,498 / 1.25%**、MATH-500 **3,115 / 0.95%**、GSM8K **476 / 0.00%**。与 Flash 的比较同时涉及教师轨迹、训练窗口和有效 batch，无法单独归因于教师。

## 阶段 2 锚点与阶段 3 `qfft_ds_v93` 历史实测（v93 统一口径）

评测口径：`v93-final-segment-last-box/math-verify-0.9.0`，temperature=0.6，top_p=0.95，`max_model_len=32768`（`qwen_math` 为 4096）；AIME24×16、AIME25×16、AMC23×8、MATH-500×4、GSM8K×1（共 4,599 条/模型）。

| 模型 | 训练数据 / 格式 | MATH-500 (mean@4) | AIME24 (mean@16) | AIME25 (mean@16) | AMC23 (mean@8) | GSM8K (mean@1) | AIME24 平均长度 / 截断率 | AIME25 平均长度 / 截断率 |
|---|---|---|---|---|---|---|---|---|
| `base` (`Qwen2.5-7B-Instruct`) | 原始基座 | 76.8% [73.7, 80.0] | 11.3% [3.1, 21.5] | 7.7% [2.9, 13.1] | 50.9% [39.1, 62.5] | 91.6% [90.1, 93.0] | 1,479 / 1.5% | 1,506 / 1.9% |
| `qfft_official`（QFFT 作者发布） | s1K-1.1 R1 + QFFT | **81.1%** [78.3, 83.9] | **20.8%** [10.2, 32.9] | **16.0%** [6.9, 26.9] | **61.3%** [49.7, 72.8] | **92.3%** [90.8, 93.7] | 14,604 / 9.4% | 11,467 / 4.4% |
| `sft_repro` (`RomEscape`) | s1K-1.1 R1 + SFT | **81.7%** [78.9, 84.4] | **18.8%** [7.9, 31.5] | **16.9%** [6.5, 29.4] | **61.3%** [49.7, 72.5] | **92.3%** [90.9, 93.7] | 22,040 / 18.8% | 20,946 / 15.8% |
| `qwen_math` (`Qwen2.5-Math-7B`) | 官方数学专用模型 | **83.4%** [80.5, 86.3] | 12.3% [4.2, 22.3] | 11.5% [4.4, 20.2] | 55.6% [43.8, 67.2] | **95.3%** [94.2, 96.4] | 1,595 / 19.4% | 1,360 / 9.8% |
| **`qfft_ds_v93`（本轮实测）** | **DS-V4.1-Flash + QFFT** | **67.0%** [63.7, 70.4] | **5.0%** [1.7, 9.4] | **4.0%** [0.8, 8.1] | **41.6%** [29.7, 53.8] | **88.7%** [87.0, 90.4] | **19,566 / 44.6%** | **18,567 / 41.9%** |

- **关键诊断**：
  1. 同样采用 QFFT 格式（无 chat 标记、不训 `<|im_end|>` EOS 结束符），在 s1K-1.1 R1 轨迹上训出的 `qfft_official` 比基座全面涨分（AIME24 `11.3% → 20.8%`，MATH-500 `76.8% → 81.1%`，AIME24 截断率仅 `9.4%`）。
  2. `qfft_ds_v93` 的截断率为 **AIME24 44.6%、AIME25 41.9%、AMC23 17.2%、MATH-500 8.0%**。MATH-500 即使给全部截断样本满分也只有 74.9%，低于基座的 76.8%。离线评分允许缺少最终段标签时从全文抽取答案。教师轨迹、训练窗口和有效 batch 都与作者配方有差异，单项影响待对照验证。

## 数据（2026-11-10 构建并上传）

- `data/qfft1000_v93/`：r1、ds、ds_verified 各 1000 行，`data_build status` 为 1000/1000、gaps 0，manifest 的 SHA-256 与文件一致。
- 1000 题 = s1K-1.1 的 974 题 + v57 的 6 题 + v93 的 20 题替换：0918（与 MATH-500 同题）和 19 道 DS 轨迹始终超出 32K 窗口的题。
- **对预注册方案的偏离**：原定"low 追加尝试后仍缺的题成对剔除"。low 补齐 3 题，另 1 题由第 14 次 high 补齐；剩下 19 题用户决定保留 1000 题，改为用已批准的 OpenR1 原题替换。选这 19 道替换题时看了 Flash 的输出（第一轮因轨迹偏短被放弃，第二轮要求思考 ≥500 词，3 题还挑过答对的或更长的），所以这 19 题的 DS 轨迹全部答对。其余 981 题的准入规则不变。结果报告要写明这一点；经过见 [历史记录 v93-P8](history.md) 与 [归档交接稿第 2.1 节](../../archive/docs/handover/kiro_ide_onboarding.md)。
- 教师报告：可核对 750 题上 R1 64.3%、DS 67.3%；只看 s1K-1.1 原题（724 道可核对）为 63.0%、66.3%。DS response 均值 10,016.7 token、最大 32,496；R1 均值 9,636.0、最大 26,835。DS 判分偏低的已知原因：DS 的最后一个框常带文字或单位（1000 行里 372 行，R1 76 行），判分器只看最后一个框。
- DS 思考段里关于隐藏参数 "oververbosity 5" 的句子共删 552 处（479 行），原文在每行 `teacher_detail.meta_edits`。
- 证据：`artifacts/qfft_teacher_v93/`（全部 DeepSeek 原始返回）与本地归档的 `archive/reports/local_cleanup_20261112/scratch/`（替换题的选题脚本和流式日志）备份在 Drive `projects/project1/v93/local_backup/`，rclone check 0 差异。

## v93 已完成（代码与修复）

- 数据：`fetch_benchmarks.py`（5 个评测集固定 revision + SHA-256）、`data_build.py`（1000 题计划、准入、两份训练集、教师报告；`--paired-exclusion` 成对剔除、每行轨迹的来源统计、元信息清洗）、`teacher.py`（DeepSeek 补齐、可选答对重采样、`--efforts low`）。
- 去污：13-gram + 相似度对 5 个评测集检查 s1K-1.1，17 对命中已逐一复核；`s1k11/row/0918` 与 MATH-500 `test/intermediate_algebra/1063.json` 同题，已替换；20 道 v93 替换题也过了同样的检查。
- 训练与评测：`train_sft.py`（全参 SFT/QFFT，ZeRO-2 offload、Liger、本地检查点 + Drive 后台镜像 + 自动恢复，每 2 个 epoch 存一次；2026-11-11 修复 `exclude=['*optim*']` 仅上传模型权重至 Drive）、`grpo.py` + `merge_lora.py`（2026-11-11 恢复 `mask_truncated_completions: true` 与 `strict=True` 判分对齐）、`generate.py`、`score.py`、`analyze.py`。
- 调度：`colab/launch.sh`、`pipeline.sh`、`remote.py`、`stages.py`（断线不中止、心跳租约、GPU 回退、rclone 挂载 5 次重试、分片上传 5 次重试、`active_run_id` 防止旧任务延迟 `unassign` 误杀、报告推送到 `colab-reports`）。

## 明确未完成

- 预注册矩阵的阶段 3 下半场（`train:sft_ds` + `eval:run:sft_ds_v93:full`）、阶段 4（Flash GRPO）、阶段 5（32K、batch 16 的 R1 双格式臂）、阶段 6、阶段 7 尚未完成。上面的 R1 16K、batch 8 对照已完成。
- 判分：Math-Verify 自身的 5 秒比较超时会返回"不相等"，现在被记成答错，而不是按约定单列为超时；修复后要重判，见 history v93-P8。

## 历史结论（仍有效，口径与 v93 不同，不能当 v93 基线）

- v26 LoRA SFT（AIMO2、8K 窗口）MATH-500 原生 72.4%，低于 E0 同预算 8 轮均值 76.6%；dev256 32K 贪心有 122/256 截断。
- 同一 8K 口径、原生判分下，公开 RomEscape SFT-7B 与 QFFT-7B 都是 389/500，平均输出 4,224 vs 1,936 token。
- 严格判分（strict-v3）在 dev256 上比原生 Math-Verify 低 18 题。
- R1 教师在 s1K-1.1 的 742 道可核对题上答对 62.7%；R1 response 最长 26,835 token，32K 窗口可全部容纳。

## 下一步：从 R1 最终权重做 GRPO（待运行）

- 初始化用 `init_model: "run:qfft_r1_16k_b8_restart_v93"`，解析到 Drive 的 `runs/project1/v93/runs/qfft_r1_16k_b8_restart_v93/final/`。已核对权重、tokenizer 和模型身份文件，身份记录为 750 步。[重跑配置](../../experiments/qfft_v93/configs/train/qfft_r1_16k_b8_restart.json) · [模型路径解析](../../experiments/qfft_v93/models.py)
- 独立 GRPO run `grpo_qfft_r1_16k_b8_v93` 沿用 v93 配方，LoRA r32、LR `1e-5`、8 题 × 8 采样、DAPO 损失、严格最终段 0/1 奖励及截断掩码。2026-11-13 用户明确授权将 637 条 E0 难题全部加入本轮 R1 题池，与已批准的 920 条合计 1,557 条源记录。批准清单固定两份文件的 SHA-256、行数和原始批准状态，源探针的 `training_approved: false` 留作历史。训练入口仍执行去重和评测集去污。[R1 GRPO 配置](../../experiments/qfft_v93/configs/grpo/grpo_qfft_r1_16k_b8.json) · [题池批准清单](../../experiments/qfft_v93/configs/grpo/r1_pool_approval.json)
- CPU 实测使用 Drive 的两份原文件。去重后 1,555 题，去除 4 道与评测集相似的题后剩 **1,551 题（918 道原批准题 + 633 道难题）**。
- 先用独立 run 做 2 步真实 rollout，观察显存、截断率和组内是否同时出现 0/1 奖励。R1 在离线 AIME24 的平均输出已达 14.7K token，现行 8,192 token rollout 的可用奖励密度需要实测。冒烟沿用正式采样和截断掩码，warmup 缩为 1 步，每步存检查点，结束时只导出 adapter。[两步冒烟配置](../../experiments/qfft_v93/configs/grpo/smoke_grpo_qfft_r1_16k_b8.json)
- 正式跑前验证 GRPO 的跨 VM 恢复。当前 adapter 镜像默认排除优化器，精确恢复需要包含优化器的 LoRA 检查点。正式训练计划为 150 步，用最后一步的 adapter 合并到 R1 权重，再按同一五项口径评测，以本轮 R1 QFFT 的 **33.63%** 为 GRPO 的竞赛基线，报告配对差值、输出长度和截断率。[GRPO 训练与合并入口](../../experiments/qfft_v93/grpo.py)

正式流程使用 `grpo:grpo_qfft_r1_16k_b8` → `eval:run:grpo_qfft_r1_16k_b8_v93/merged:full`。配置已准备，GPU 待用户授权。

两步冒烟命令（WSL2 仓库根目录，待运行）：

```bash
source ~/.proxy_env.sh
bash experiments/qfft_v93/colab/launch.sh --gpu G4,H100 grpo:smoke_grpo_qfft_r1_16k_b8
```

成功时会输出 `P1_RESULT=OK`，Drive 的 `runs/project1/v93/runs/smoke_grpo_qfft_r1_16k_b8_v93/GRPO_COMPLETE.json` 应记录 2 步。显存、`completions/clipped_ratio` 和 `frac_reward_zero_std` 见同一目录的 `metrics.jsonl`。
