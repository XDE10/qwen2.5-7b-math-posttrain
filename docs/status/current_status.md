# 项目一当前状态（2026-11-12，v93）

> 本页只写「现在做到哪」的摘要；逐版本历史与精确数字见 [history.md](history.md)，仓库约定见根目录 [AGENTS.md](../../AGENTS.md)，v93 方案见 [三周冲刺方案_v93.md](../blueprints/三周冲刺方案_v93.md)。接手当前工作（本机 Kiro IDE / WSL2）先读 [交接文档](../handover/kiro_ide_onboarding.md)。

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
- 2026-11-12 20:14（UTC+8）：R1 16K batch 8 对照 `qfft_r1_16k_b8_v93` 的第 196/750 步已见于 `colab-reports` 提交 `aca5248`，正式评测结果待回传。

## 阶段 2 锚点与阶段 3 `qfft_ds_v93` 实测结果汇总（v93 统一口径）

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
- **对预注册方案的偏离**：原定"low 追加尝试后仍缺的题成对剔除"。low 补齐 3 题，另 1 题由第 14 次 high 补齐；剩下 19 题用户决定保留 1000 题，改为用已批准的 OpenR1 原题替换。选这 19 道替换题时看了 Flash 的输出（第一轮因轨迹偏短被放弃，第二轮要求思考 ≥500 词，3 题还挑过答对的或更长的），所以这 19 题的 DS 轨迹全部答对。其余 981 题的准入规则不变。结果报告要写明这一点；经过见交接文档 2.1 与 [history.md](history.md) v93-P8。
- 教师报告：可核对 750 题上 R1 64.3%、DS 67.3%；只看 s1K-1.1 原题（724 道可核对）为 63.0%、66.3%。DS response 均值 10,016.7 token、最大 32,496；R1 均值 9,636.0、最大 26,835。DS 判分偏低的已知原因：DS 的最后一个框常带文字或单位（1000 行里 372 行，R1 76 行），判分器只看最后一个框。
- DS 思考段里关于隐藏参数 "oververbosity 5" 的句子共删 552 处（479 行），原文在每行 `teacher_detail.meta_edits`。
- 证据：`artifacts/qfft_teacher_v93/`（全部 DeepSeek 原始返回）与 `scratch/`（替换题的选题脚本和流式日志）备份在 Drive `projects/project1/v93/local_backup/`，rclone check 0 差异。

## v93 已完成（代码与修复）

- 数据：`fetch_benchmarks.py`（5 个评测集固定 revision + SHA-256）、`data_build.py`（1000 题计划、准入、两份训练集、教师报告；`--paired-exclusion` 成对剔除、每行轨迹的来源统计、元信息清洗）、`teacher.py`（DeepSeek 补齐、可选答对重采样、`--efforts low`）。
- 去污：13-gram + 相似度对 5 个评测集检查 s1K-1.1，17 对命中已逐一复核；`s1k11/row/0918` 与 MATH-500 `test/intermediate_algebra/1063.json` 同题，已替换；20 道 v93 替换题也过了同样的检查。
- 训练与评测：`train_sft.py`（全参 SFT/QFFT，ZeRO-2 offload、Liger、本地检查点 + Drive 后台镜像 + 自动恢复，每 2 个 epoch 存一次；2026-11-11 修复 `exclude=['*optim*']` 仅上传模型权重至 Drive）、`grpo.py` + `merge_lora.py`（2026-11-11 恢复 `mask_truncated_completions: true` 与 `strict=True` 判分对齐）、`generate.py`、`score.py`、`analyze.py`。
- 调度：`colab/launch.sh`、`pipeline.sh`、`remote.py`、`stages.py`（断线不中止、心跳租约、GPU 回退、rclone 挂载 5 次重试、分片上传 5 次重试、`active_run_id` 防止旧任务延迟 `unassign` 误杀、报告推送到 `colab-reports`）。

## 明确未完成

- 阶段 3 下半场（`train:sft_ds` + `eval:run:sft_ds_v93:full`）、阶段 4（`grpo:grpo_qfft_ds`）、阶段 5（R1 对 `qfft_r1` / `sft_r1`）、阶段 6、阶段 7 尚未完成。
- 判分：Math-Verify 自身的 5 秒比较超时会返回"不相等"，现在被记成答错，而不是按约定单列为超时；修复后要重判，见 history v93-P8。

## 历史结论（仍有效，口径与 v93 不同，不能当 v93 基线）

- v26 LoRA SFT（AIMO2、8K 窗口）MATH-500 原生 72.4%，低于 E0 同预算 8 轮均值 76.6%；dev256 32K 贪心有 122/256 截断。
- 同一 8K 口径、原生判分下，公开 RomEscape SFT-7B 与 QFFT-7B 都是 389/500，平均输出 4,224 vs 1,936 token。
- 严格判分（strict-v3）在 dev256 上比原生 Math-Verify 低 18 题。
- R1 教师在 s1K-1.1 的 742 道可核对题上答对 62.7%；R1 response 最长 26,835 token，32K 窗口可全部容纳。

## 当前优先目标（2026-11-12）

- 实习项目优先拿到一组可复现的自训增益。先用 `qfft_r1_16k_b8_v93` 在固定 Qwen2.5-7B-Instruct 底座上做全参 QFFT，使用已上传的 v93 去污 R1 数据、16K 截断、有效 batch 8、6 epoch。用第 750 步最终权重完成五项评测，再根据结果决定是否投入 GRPO。这轮命令只包含监督训练和评测，由用户启动 GPU。[训练配置](../../experiments/qfft_v93/configs/train/qfft_r1_16k_b8.json)
- 主指标是 AIME24 + AIME25 + AMC23 的 100 题按题平均分。当前基座 **26.0625%**，建议目标是比基座高至少 **5 个百分点**（约 **31.1%**），且按题配对 bootstrap 的 95% 置信区间下界高于 0。同步报告 MATH-500、GSM8K、平均输出 token 和截断率。目标待实测，简历只引用实测成绩和明确的评测口径。
- RomEscape 的 `qfft_repro` 是独立复现权重，作为复现参考。`qfft_official` 是 QFFT 作者权重。自训增益以同一 v93 评测下的 `base` 为基准。[模型配置](../../experiments/qfft_v93/configs/pipeline.json)
