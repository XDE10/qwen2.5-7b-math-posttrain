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

## R1 GRPO（8K 冒烟完成，正式训练待运行）

- 初始化用 `init_model: "run:qfft_r1_16k_b8_restart_v93"`，解析到 Drive 的 `runs/project1/v93/runs/qfft_r1_16k_b8_restart_v93/final/`。已核对权重、tokenizer 和模型身份文件，身份记录为 750 步。[重跑配置](../../experiments/qfft_v93/configs/train/qfft_r1_16k_b8_restart.json) · [模型路径解析](../../experiments/qfft_v93/models.py)
- 独立 GRPO run `grpo_qfft_r1_16k_b8_v93` 使用 LoRA r32、LR `1e-5`、8 题 × 8 采样、8,192 token 生成上限、DAPO 损失、0/1 奖励及截断掩码。奖励优先判最终段最后一个完整答案框，仅在两个最终段标签都缺失时回退到全文最后一个完整框。2026-11-13 用户授权把 637 条 E0 难题全部加入本轮 R1 题池，与已批准的 920 条合计 1,557 条源记录。批准清单固定两份文件的 SHA-256、行数和原始批准状态，训练入口执行去重与评测集去污。[R1 GRPO 配置](../../experiments/qfft_v93/configs/grpo/grpo_qfft_r1_16k_b8.json) · [题池批准清单](../../experiments/qfft_v93/configs/grpo/r1_pool_approval.json)
- CPU 实测使用 Drive 的两份原文件。去重后 1,555 题，去除 4 道与评测集相似的题后剩 **1,551 题（918 道原批准题 + 633 道难题）**。
- 冒烟用独立 run，保留正式采样和截断掩码，warmup 缩为 1 步，每步存检查点，结束时导出 adapter。

严格奖励基线（2 步，已完成）：

```bash
source ~/.proxy_env.sh
bash experiments/qfft_v93/colab/launch.sh --gpu G4,H100 grpo:smoke_grpo_qfft_r1_16k_b8
```

会话 `p1-r1-grpo-smoke8k-1113` 完成两步，128 条中 113 条为 `no_answer`，37 条被 TRL 掩码，8/16 题组奖励零方差。[会话报告](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/ac48a7e6202dc8e15ecb56005882c5b6d372df36/v93/sessions/p1-r1-grpo-smoke8k-1113/README.md)

2026-11-13 用户授权从同一 R1 权重另跑 2 步 8,192-token 诊断，沿用严格奖励。诊断入口为 `grpo:diagnose_grpo_qfft_r1_8k`。逐条原文、vLLM 停止原因、TRL 截断掩码和判分原因保存在该 run 的 Drive 文件 `rollout_diagnostics.jsonl`。下载后可用 `python -m experiments.qfft_v93.rollout_diagnostics <文件>` 在 CPU 上重放严格判分，并仅对缺最终段标签的回答使用最后一个完整答案框做对照。原文仅写入 Drive。

诊断已完成 2 步（会话 `p1-r1-grpo-diag8k-1113`，代码包 `dad352384f53`）。128 条中 31 条达到长度上限，另有 1 条因字符串 `\n\nProblem:` 提前结束并被 TRL 掩码。97 条未达长度上限的回答里，82 条严格判为 `no_answer`，全部缺最终段标签。80 条有完整答案框，回退重判后 44 对、36 错。按相同题组离线重算，零方差组从 9/16 降到 3/16。该轮训练使用严格奖励。[诊断会话报告](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/278ba2c/v93/sessions/p1-r1-grpo-diag8k-1113/README.md)

2026-11-13 用户授权启用 R1 的答案框回退，并用同一第 750 步权重跑 8K 两步冒烟（`grpo:smoke_grpo_qfft_r1_fallback_8k`）。G4 会话 `p1-r1-grpo-box8k-1113` 完成，代码包 `80fb86403290`。同批 128 条上，严格判分为 CPU 重放，回退是 GPU 训练实际使用的奖励。[会话报告](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/278ba2c/v93/sessions/p1-r1-grpo-box8k-1113/README.md) · [SwanLab](https://swanlab.cn/@XDE10/project1-7b-math/runs/lm8ta3jd)

| 同批指标 | 严格判分 | 答案框回退 | 差值（百分点） |
|---|---:|---:|---:|
| `no_answer` | 113/128（88.28%） | 30/128（23.44%） | −64.84 |
| 零方差题组 | 8/16（50.00%） | 3/16（18.75%） | −31.25 |
| 触及长度上限 | 37/128（28.91%） | 37/128（28.91%） | 0 |

76 条未截断回答因回退恢复可判分，其中 50 对、26 错。37 条触及长度上限的回答仍被掩码。两步用时约 422 / 402 秒，PyTorch 峰值分配 66.6 GiB。诊断与回退的逐条原文保存在各自 Drive run 下，GitHub 只存汇总。CPU 回归 181 通过、1 跳过。16K 和正式 150 步均未运行。

### 正式 150 步前的准备

- **跨 VM 恢复。** 正式 R1 每 10 步将完整 LoRA 检查点镜像到 Drive，保留最近 3 份已发布的完整镜像。镜像包含 adapter、优化器、调度器、RNG 和 Trainer 状态。之前的 G4 冒烟中，完整本地检查点约 935 MiB。CPU 用真实 TRL + LoRA、微型模型和本地 Drive 替身验证，从空的第二个 VM 目录下载第 1 步镜像，逐项核对 adapter、AdamW、调度器及 Python/NumPy/PyTorch RNG，然后从第 2 步跑到第 3 步。实际 Colab 换 VM 的恢复仍待验证，vLLM 采样序列也未验证能逐 token 重现。[检查点实现](../../experiments/qfft_v93/checkpointing.py)
- **沿用 8K 回退冒烟的奖励。** `no_answer` 表示按当前规则没提取到可判分的答案，奖励为 0。有最终段标签时只判该段最后一个完整答案框；两种标签都缺失时才取全文最后一个完整框，并与参考答案比较。TRL 1.17.1 仍掩掉截断回答的 loss token，组内奖励均值沿用冒烟算法。
- **将训练与评测拆成两个会话。** 按两步实测速率外推，150 步约 **17.2 小时**，另有环境启动、保存、合并和评测。正式训练使用 `grpo:grpo_qfft_r1_16k_b8`。训练结束确认 `GRPO_COMPLETE.json` 记录 150 步，并核对 `merged/` 的最终权重，再用 `eval:run:grpo_qfft_r1_16k_b8_v93/merged:full` 跑五项评测。以这轮 R1 QFFT 的竞赛 **33.63%** 为基线，报告按题配对差值、输出长度和截断率。[GRPO 入口](../../experiments/qfft_v93/grpo.py)

### 8K 换 VM 验证（命令已准备，GPU 待授权）

两个会话共享独立的 [两步验证配置](../../experiments/qfft_v93/configs/grpo/smoke_grpo_qfft_r1_resume_8k.json)，从 R1 第 750 步最终权重启动，并沿用答案框回退与截断掩码。训练预算固定为两步。

```bash
cd /home/xde10/projects/qwen2.5-7b-math-posttrain
source ~/.proxy_env.sh
bash experiments/qfft_v93/colab/launch.sh --gpu G4,H100 --session p1-r1-resume8k-a \
  grpo:smoke_grpo_qfft_r1_resume_8k:pause-after=1
```

等日志显示 `paused after step 1; full Drive mirror confirmed` 和第一台 VM 已释放，再启动第二台：

```bash
bash experiments/qfft_v93/colab/launch.sh --gpu G4,H100 --session p1-r1-resume8k-b \
  grpo:smoke_grpo_qfft_r1_resume_8k:resume
```

第二台需找到完整断点才能启动训练。验收看 `Resuming from restored .../checkpoint-1`、`step 2/2` 和 `P1_RESULT=OK`，以及 Drive 的 `runs/project1/v93/runs/smoke_grpo_qfft_r1_resume_8k_v93/GRPO_COMPLETE.json` 中的 `steps: 2` 和 `resumed_from`。两条命令用同一阶段租约。

### 8K / 16K 生成长度对照（每组两步）

[8K 配置](../../experiments/qfft_v93/configs/grpo/compare_grpo_qfft_r1_8k.json)和 [16K 配置](../../experiments/qfft_v93/configs/grpo/compare_grpo_qfft_r1_16k.json)均从 R1 第 750 步最终权重独立启动。用固定种子从去重、去污后的两份获批来源各抽 8 题，固定顺序，每步 4 道 E0 中等题（4 次答对 1–3 次）和 4 道 E0 难题（4 次全错），每题生成 8 条。CPU 核对了源文件哈希和题目顺序，用本地 Qwen2.5-7B-Instruct tokenizer 核对，模板编码后 prompt 为 60–167 token。两组只改生成上限（8,192 / 16,384），沿用答案框回退、截断掩码及同一训练参数。

用户授权在 G4 上依次运行这两组各两步的对照。正式 150 步和接近 32K 的配置待另行决定。CPU 测试结果为 187 通过、1 跳过（另有 4 个子测试通过），覆盖配置等价、TRL 题组顺序、掩码后的奖励题组计数和阶段计时。GPU 汇总由运行结束后的 `colab-reports` 会话报告回传。

在 WSL 终端先进入仓库并加载代理，依次运行两条命令，使用 G4，等第一条结束后再运行第二条。

```bash
cd /home/xde10/projects/qwen2.5-7b-math-posttrain
source ~/.proxy_env.sh
bash experiments/qfft_v93/colab/launch.sh --gpu G4 --session p1-r1-len8k grpo:compare_grpo_qfft_r1_8k
bash experiments/qfft_v93/colab/launch.sh --gpu G4 --session p1-r1-len16k grpo:compare_grpo_qfft_r1_16k
```

分别检查 `P1_RESULT=OK` 和 Drive 对应 `compare_grpo_qfft_r1_{8k,16k}_v93` 下的 `GRPO_COMPLETE.json`（`steps: 2`）、`rl_pool_report.json`（相同题目 ID 与顺序）及 `metrics.jsonl` 的 `comparison/` 指标。

- 长度截断率用 vLLM 的 `finish_reason=length`。另记录 TRL 掩码率，字符串停止也可能被掩码。
- 有效奖励题组记录未掩码回答中同时有 0 和 1 的题数。另记录仍有更新信号的题数，包含只有掩码回答提供奖励差异、但仍有未掩码 token 获得非零优势的组。
- 每约一秒采样整卡显存与 GPU 利用率，输出每步及各阶段的采样峰值、平均利用率和样本数。生成耗时只计 vLLM 生成。其余 rollout 包括权重同步、判分及 logprob 计算。更新耗时包括前向、反向、优化器和训练循环开销。每步耗时从训练步开始算到优化器更新结束，不含模型加载、保存和镜像。
- R1 最终权重原生窗口为 32,768 token，含题目和生成。保留 1,024 token 的 prompt 预算时，32K 上下文最多配 31,744 个生成 token。先用 16K 实测截断和整卡显存，再判断要不要做接近 32K 的对照。两步对照用于选生成预算，分数需经固定评测验证。
