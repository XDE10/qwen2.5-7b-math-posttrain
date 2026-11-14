# 项目一 v93 数学后训练现状与实验总结

**截至 2026-11-14，UTC+8。** 项目主线是用开源 7B 指令模型做数学后训练，在相同评测协议下分别量化监督微调、强化学习和推理长度的变化。本文覆盖当前主线的配置、已完成实验、主要历史结论和剩余决策。仓库的逐版本工程记录见 `docs/status/history.md`。代码以本地 `cf8a805` 及后续工作区改动为准，最近一次竞赛留出评测以 `colab-reports` 的 `40c2f8a` 为准。部分仓库状态文档早于这次评测。

## 1. 已经得到的结果

- **全参 QFFT 有实测提分。** 固定从 `Qwen/Qwen2.5-7B-Instruct` 开始，用 1,000 道题的 R1 长推理轨迹做全参 QFFT。最终模型在竞赛 100 题上从 **26.06% 到 33.63%，+7.56 个百分点，按题配对 95% CI [+3.88, +11.44]**。MATH-500 从 76.80% 到 80.50%。[五套评测及配对结果](https://github.com/XDE10/qwen2.5-7b-math-posttrain/tree/71a78e7c69989e7ced436a8e228356ccd4402737/v93/latest/evals)
- **现有 GRPO 试点尚未证实额外提分。** `sequence_mask` 第 30 步在 MATH 开发留出 249 题上为 **+1.41 pp [−1.00, +4.02]**，在竞赛 100 题上为 **−1.50 pp [−3.75, +0.75]**，均相对 R1 QFFT。第 30 步只跑了三个竞赛子集和开发留出。[开发评测](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/456404ccd66134ad5eaad7fc8fd6cb5d747ab8f8/v93/latest/grpo_dev/results.json) · [竞赛评测](https://github.com/XDE10/qwen2.5-7b-math-posttrain/tree/ad8266037db114f5a811544ad8cfdd88c0488dea/v93/latest/evals/grpo_qfft_r1_16k_continue30_v93_merged)
- **`token_mask` 第 10 步的可复述结果是输出变短。** 独立竞赛留出 59 题上，比 `sequence_mask` 第 10 步 **+0.21 pp [−2.54, +2.97]**，对应 472 次生成中多答对 1 次。平均输出减少约 **1,744 token（16.8%）**。相对原始 R1 为 **+0.85 pp [−2.33, +4.03]**。单种子短跑无法建立稳定的准确率收益。[59 题最终报告](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/40c2f8a/v93/latest/r1_competition_2026/results.json)

## 2. 模型、数据和数据偏离

| 项目 | 当前取值和用途 |
|---|---|
| 初始底座 | `Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`。内部简写 `base`，其模型类型是 Instruct。 |
| 原始 R1 | 项目自训的 `qfft_r1_16k_b8_restart_v93` 第 750 步完整导出。R1 指 s1K-1.1 的教师轨迹。初始权重来自上面的 Qwen Instruct。 |
| Flash QFFT | 同一底座用 DeepSeek-V4.1-Flash 轨迹做 QFFT，产物 `qfft_ds_v93`。 |
| 公共锚点 | QFFT 作者 `lwl-uestc/QFFT-S1-7B`，RomEscape 的 `PJiNH/QFFT-repro-S1.1-SFT-7B`，以及数学专用 `Qwen/Qwen2.5-Math-7B-Instruct`。锚点 revision 固定于 `configs/pipeline.json`。 |

监督阶段的两份对齐训练数据各 1,000 道，来自 s1K-1.1 的 974 道、v57 的 6 道替换和 v93 的 20 道替换。一道 v93 替换源于与 MATH-500 同题（`s1k11/row/0918`）。其余 19 道替换源于 Flash 轨迹无法在 32K 内结束。原有 23 道缺口，3 道通过 low 补齐，1 道通过追加 high 补齐。对五项评测做 13-gram 与相似度检查并人工复核命中。最终数据目录为 `data/qfft1000_v93/`，R1、Flash 和 `ds_verified` 各有 1,000 行，manifest 与文件哈希一致。

通常按时间顺序取首条正常结束、思考和最终答案非空、按 SFT 编码不超过 32,768 token 的尝试。**19 道例外属于预注册偏离**。原计划用 low 补齐后仍缺的题成对剔除，后来为维持 1,000 题改用获批的 OpenR1 原题。选择时已经看过 Flash 输出，并使用过轨迹长度以及少数题的对错条件。因此这 19 道 Flash 轨迹全对，其余 981 道沿用首条合格轨迹规则。建库准入和实际 R1 16K 训练时发生的截断分别记录。

可核对答案的 750 道题上，R1 教师轨迹正确率 64.3%，Flash 67.3%。仅看可核对的 724 道 s1K-1.1 原题，为 63.0% 和 66.3%。R1 平均响应 9,636 token，最大 26,835。Flash 平均 10,016.7，最大 32,496。Flash 的 479 行去除了 552 处与隐藏参数有关的元叙述，原文记录在行级修订元数据中。教师的单条轨迹分数无法替代训练后学生的独立评测。[s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1) · [项目状态与数据说明](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/cf8a805/docs/status/current_status.md)

## 3. 监督训练配置和实跑

| 参数 | Flash QFFT 实跑 | R1 QFFT 实跑 |
|---|---|---|
| 训练方式 | 全参数，QFFT | 全参数，QFFT |
| 输入格式 | response-only，无 chat 标记和终止 EOS | 同左 |
| 训练窗口 | 32,768，不截断 | 16,384，允许截断 |
| 单卡 micro batch × 累积 | 1 × 16 | 1 × 8 |
| 优化器和调度 | AdamW，β₁ 0.9、β₂ 0.999，weight decay 1e-4，LR 1e-5，cosine，warmup 10%，grad norm 1.0 | 同左 |
| 训练长度 | 6 epochs，378 步 | 6 epochs，750 步 |
| 工程实现 | BF16，gradient checkpointing，DeepSpeed ZeRO-2 + CPU AdamW offload，Liger，SDPA，seed 42 | 同左 |
| 最终结果 | 从带优化器的 checkpoint-126 恢复后完成，五套评测已回传 | 旧 run 停在第 209 步，Drive 只有第 125 步轻量镜像。新 run 从底座重跑，完成 750 步和五套评测。 |

R1 的 16K 训练有 86/1,000 条轨迹触发截断，共移除 129,034 token，57 条截断后缺少最终回答段。预注册还设计了标准 SFT 格式（保留 chat 格式，mask prompt 的 loss，训练结束标记）以及教师 × 格式的 32K、batch 16 矩阵。实际完成的是 Flash QFFT 32K/batch 16 与 R1 QFFT 16K/batch 8。两臂的长度与 batch 同时变化，分差无法单独归因于教师。[训练配置与运行手册](https://github.com/XDE10/qwen2.5-7b-math-posttrain/tree/cf8a805/experiments/qfft_v93) · [R1 重跑会话](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/71a78e7c69989e7ced436a8e228356ccd4402737/v93/sessions/p1-r1-16k-restart-1112/README.md)

## 4. 正式评测口径和分数

使用固定的 Qwen token 模板，题目放在 user 消息中。system prompt 为 `Please reason step by step, and put your final answer within \boxed{}.`。原生 RoPE，无 YaRN，temperature 0.6、top_p 0.95。默认上下文 32,768，生成上限为窗口减 prompt token 数。数学专用锚点保留其 4,096 原生窗口。AIME24、AIME25 各 30 题、每题 16 次，AMC23 40 题、每题 8 次，MATH-500 每题 4 次，GSM8K 每题 1 次。五套共 4,599 条生成/模型。竞赛 100 题为 AIME24、AIME25、AMC23 合并。先取每题的采样正确率，再按题等权计算 mean@n，区分 pass@k 或多数票。差值置信区间按 `sample_id` 逐题配对、bootstrap 10,000 次（seed 0）。

离线评分用 Math-Verify 0.9.0，优先从最终段取最后一个答案框，现行非严格分支存在全文或未加框的解析回退。训练奖励使用更严格的答案框规则。评测主指标使用自动判分，保留 `no_answer`、超时和判分故障统计。旧锚点评测的 grader 标签早于 timeout-status 修复，新 59 题三臂使用同一新版且判分故障为零。[配置](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/cf8a805/experiments/qfft_v93/configs/pipeline.json) · [评分实现](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/cf8a805/experiments/qfft_v93/grading.py)

| 模型 | AIME24 | AIME25 | AMC23 | 竞赛 100 题 | MATH-500 | GSM8K |
|---|---:|---:|---:|---:|---:|---:|
| 原始 Qwen2.5-7B-Instruct | 11.25% | 7.71% | 50.94% | 26.06% | 76.80% | 91.58% |
| QFFT 作者权重 | 20.83% | 16.04% | 61.25% | 35.56% | 81.10% | 92.27% |
| RomEscape SFT 权重 | 18.75% | 16.88% | 61.25% | 35.19% | 81.70% | 92.34% |
| Qwen2.5-Math-7B-Instruct | 12.29% | 11.46% | 55.63% | 29.38% | 83.40% | 95.30% |
| 项目 Flash QFFT | 5.00% | 3.96% | 41.56% | 19.31% | 67.00% | 88.70% |
| **项目 R1 QFFT** | **18.54%** | **14.79%** | **59.06%** | **33.63%** | **80.50%** | **91.43%** |
| 项目 R1 + GRPO，sequence 第 30 步 | 18.33% | 16.25% | 54.38% | 32.13% | 未测 | 未测 |

- R1 QFFT 竞赛 100 题相对底座 **+7.56 pp [+3.88, +11.44]**，满足预设的「点估计至少 +5 pp 且区间下界大于 0」。区间包含小于 +5 pp 的增益。相对 Flash **+14.31 pp [+9.94, +18.88]**，相对 QFFT 作者权重 **−1.94 pp [−4.06, +0.06]**。数学专用锚点在 MATH-500、GSM8K 两套分数高于 R1。[主结果](https://github.com/XDE10/qwen2.5-7b-math-posttrain/tree/71a78e7c69989e7ced436a8e228356ccd4402737/v93/latest/evals)
- R1 在 AIME24、AIME25、AMC23、MATH-500、GSM8K 的平均输出 token/截断率依次为 14,651/10.42%、11,125/2.50%、7,498/1.25%、3,115/0.95%、476/0%。Flash 的 AIME24/25 截断为 44.6%/41.9%，AMC23 为 17.2%，MATH-500 为 8.0%。把 Flash MATH-500 所有截断作答补到全对，最多也只有约 74.9%。[实验状态](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/cf8a805/docs/status/current_status.md)

## 5. GRPO 实际配方

所有 R1 试点从 R1 QFFT 的第 750 步完整权重开始。累计 30 步是同一条 `sequence_mask` 轨迹的继续训练。`token_mask` 10 步从原始 R1 重新开始，两臂记录的前 80 个题组及顺序一致。两次使用不同代码包，sequence 臂第 5 步有跨 VM 恢复，采样与训练仅完成单种子对照。

| 组别 | 已完成 16K 试点的参数 |
|---|---|
| 题池 | 920 条 E0 四次答对 1–3 次的获批题，加入另行获批的 637 条 E0 四次全错题。1,557 源记录 → 1,555 唯一题 → 1,551 道训练题（918 + 633），训练前核对批准文件与评测重叠。难度标签来自旧 E0，当前 R1 的可学性要看实测。 |
| LoRA 与学习率 | r32，alpha32，dropout 0，all-linear，BF16，gradient checkpointing，LR 1e-5，warmup 5 步后恒定，seed 42。 |
| 更新批次 | 每步 8 个 prompt × 8 条生成，micro batch 2 × gradient accumulation 32，`num_iterations=1`。 |
| Rollout | prompt 上限 1,024，completion 上限 16,384，temperature 1.0，top_p 1.0。EOS ID 151645/151643，字符串停止包含下一道 Question/Problem 或用户消息标记。 |
| 奖励 | 仅 0/1 答案正确性。存在最终段标签时严格取该段最后一个完整框，两种标签都缺失时回退到全文最后一个完整框。判分基础设施失败重试，仍有失败时在更新前停止。 |
| 损失与掩码 | `loss_type=dapo`，按有效 token 归一化，组内中心化、`scale_rewards=none`，β=0，ε_low 0.2、ε_high 0.28，`mask_truncated_completions=true`。另由 vLLM 采样端到训练端的 IS 模式决定权重校正（sequence 或 token），两个机制分别记录。 |
| 运行与保存 | TRL 1.17.1 + vLLM 0.32.0 colocate，G4 vLLM 显存比例 0.45，vLLM 不启用 sleep mode。试点每 5 步保存，结束导出 adapter 和合并权重。 |

源配置为 `experiments/qfft_v93/configs/grpo/` 下的 `_defaults.json`、`grpo_qfft_r1_16k_pilot.json` 等。旧文件 `grpo_qfft_r1_16k_b8.json` 尽管带 16K 字样，实际继承的生成上限是 8,192。以展开后的配置和 run identity 核对实际参数。[GRPO 入口和配置](https://github.com/XDE10/qwen2.5-7b-math-posttrain/tree/cf8a805/experiments/qfft_v93)

## 6. RL 诊断与留出评测

### 6.1 奖励解析与训练生成上限

最初的 8K 两步 G4 冒烟发现 R1 经常给出答案框却缺最终段标签。同一批 128 条回放，启用完整答案框回退后 `no_answer` 从 **113/128 降至 30/128**，零方差题组从 **8/16 降至 3/16**。76 条未截断回答恢复可评分，其中 50 对、26 错。触及上限的 37 条保持原掩码。[8K 回退实测](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/278ba2c/v93/sessions/p1-r1-grpo-box8k-1113/README.md)

另一批固定 16 道混合难度题的三组两步对照只改变训练生成上限。以 vLLM `finish_reason=length` 计截断。实际 loss 掩码还可能包含某些字符串停止。

| 生成上限 | 达上限的生成（共 128 条） | 未掩码回答有 0/1 两种奖励的组（共 16 组） | 每步约用时 | 整卡显存峰值 |
|---|---:|---:|---:|---:|
| 8K | 46（35.94%） | 8 | 7.1 分钟 | 68.63 GiB |
| 12K | 31（24.22%） | 9 | 12.4 分钟 | 73.42 GiB |
| 16K | 9（7.03%） | 10 | 18.3 分钟 | 78.91 GiB |

这些实验量化了训练信号和计算开销。后续训练使用 16K，评测保留 32K 原生上下文。[固定题组的长度实验](https://github.com/XDE10/qwen2.5-7b-math-posttrain/tree/456404ccd66134ad5eaad7fc8fd6cb5d747ab8f8/v93/latest/runs)

### 6.2 sequence_mask 第 10 步和第 30 步

先在 G4 跑到第 5 步，发布完整检查点，在另一台 G4 核对身份与文件后恢复，跑到第 10 步。随后从完整第 10 步断点继续到第 30 步。恢复包含 adapter、AdamW、调度器、Trainer 与 RNG 状态，尚未验证 vLLM 的采样能跨 VM 逐 token 重现。

历史受保护的 MATH dev256 去除与 R1 QFFT 数据、GRPO 源及五套评测重叠或相似的 7 道题，冻结为 249 道、每题 2 次。同一协议下，原始 R1、sequence 第 10 步、第 30 步分别为 81.93%、82.93%、83.33%。第 10 步相对 R1 **+1.00 pp [−1.61, +3.61]**，第 30 步相对 R1 **+1.41 pp [−1.00, +4.02]**，第 30 步相对第 10 步 **+0.40 pp [−2.21, +3.01]**。[第 10 步](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/5c08c4a3b58967a1a98f986f1d0b9fceedd0fd12/v93/latest/grpo_dev/results.json) · [第 30 步](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/456404ccd66134ad5eaad7fc8fd6cb5d747ab8f8/v93/latest/grpo_dev/results.json)

竞赛 100 题第 30 步相对 R1 **−1.50 pp [−3.75, +0.75]**。AMC23 由 189/320 降到 174/320，**−4.69 pp [−9.69, 0.00]**。双方均未截断的 308 对作答净少 13 条正确。两臂的 AMC23 作答均无 `no_answer`，按同一版本重判 640 条复现原对错。这个分解解释了掉分来自哪些样本，无法识别训练退步的原因。[竞赛 30 步报告](https://github.com/XDE10/qwen2.5-7b-math-posttrain/tree/ad8266037db114f5a811544ad8cfdd88c0488dea/v93/latest/evals/grpo_qfft_r1_16k_continue30_v93_merged)

前 30 步中，127 道 E0 中等题与 113 道 E0 难题的全零奖励组分别为 6/127 和 49/113，未掩码混合奖励组为 84/127 和 61/113。难题消耗 **64.3% 的生成 token**，分别有 48/1,016 和 171/904 条回答触及 16K 上限。重判 1,693 条未掩码作答，二元奖励只差 1 条。第 11–30 步 LR 为 1e-5，梯度范数无持续走高，缺少相对原始 R1 的 KL 轨迹。题池难度与计算效率的关系已有观测，采样重配和学习率对最终得分的效果没有受控对照。[训练与诊断汇总](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/cf8a805/docs/status/current_status.md)

### 6.3 IS 诊断和 token_mask 独立留出

固定 64 条 rollout，在无优化器更新的一次 G4 诊断里对比 sequence 与 token 的 IS 权重。其中 8 条被长度相关规则掩码，另有 17 条未截断的 8K+ 回答，其 sequence 权重小于 0.01。两条正确长回答的 sequence 权重小于 1.5e-18，token 方案对这两条只将 1 个 token 权重设零。全批 344,325 个候选 token 中 token IS 设零 22 个，sequence IS 的精确零权重为 0。序列权重非常小与精确设零是不同现象；它们的能力影响由后续留出评测检验。

实际 10 步对照中，sequence 与 token 两臂在同一题目顺序下各生成 640 条，训练答对 **294 和 329 条**，未掩码混合奖励组 **49 和 51**，触及生成上限 **80 和 76**，平均每步约 **18.55 和 18.50 分钟**。训练作答是在各自不断更新的策略下生成，不能据此估计泛化增益。[IS 诊断回传](https://github.com/XDE10/qwen2.5-7b-math-posttrain/tree/168c6d5a3076dd3403ce5b984da168d1c7792747/v93/sessions/p1-r1-is-diagnostic) · [token_mask 训练完成记录](https://github.com/XDE10/qwen2.5-7b-math-posttrain/tree/40c2f8a/v93/latest/runs/grpo_qfft_r1_16k_tokenmask10_v93)

2026 年竞赛 59 题留出构建时从 80 个原始题位去除 14 个图形题位、7 个重复题位，最终包括 AIME I 14 道、AIME II 14 道、AMC 10A 19 道和 AMC 12A 12 道。对照项目 QFFT/GRPO 数据、过去五套评测和 MATH 开发集按规范化文本精确匹配和近重复（8-gram、difflib ≥0.50）复核。2026-11-14 17:10（UTC+8）冻结，SHA-256 为 `c7b67bacc08851ac9f85935c3459dc973520c3ce3d1a78ecbc2408f027e113cc`。AMC 改成短答，用选项对应的数值或表达式判分，AMC 10A 第 8 题因题意需要保留原选项值。成绩用于三臂内部比较。三臂均每题 8 次、共 472 条生成，零缺题、零 `no_answer` 和零判分故障。

| 模型 | 正确生成 | mean@8 | 平均输出 token | 截断 |
|---|---:|---:|---:|---:|
| 原始 R1 QFFT | 139/472 | 29.45% | 10,760 | 20/472 |
| sequence_mask 第 10 步 | 142/472 | 30.08% | 10,399 | 15/472 |
| token_mask 第 10 步 | 143/472 | 30.30% | 8,656 | 15/472 |

| 按题配对差值 | Δ（百分点） | 95% CI |
|---|---:|---:|
| token_mask − sequence_mask | **+0.21** | **[−2.54, +2.97]** |
| token_mask − 原始 R1 | **+0.85** | **[−2.33, +4.03]** |
| sequence_mask − 原始 R1 | **+0.64** | **[−2.75, +3.81]** |

以每题平均输出 token 做配对，token 相对 sequence 为 **−1,743.5 token**，重采样 95% CI **[−2,681.61, −841.92]**。准确率差值的区间跨 0，长度下降是在这次协议与这对固定权重下观察到的效果。最近评测由 G4 会话 `p1-1114-192550` 完成，最终标记为 OK。59 题以及竞赛 100 题分数都已经看过，后续可用于既有结果对标。若按这些结果挑选方案，需要另设独立最终留出或明确把分析标成探索性。[会话](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/40c2f8a/v93/sessions/p1-1114-192550/README.md) · [逐模型汇总与准确率配对](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/40c2f8a/v93/latest/r1_competition_2026/results.json)

## 7. 工程、运行状态和历史参考

- **硬件和环境。** 主卡为 Colab G4（RTX PRO 6000 Blackwell 96 GB），冒烟机器具备 177 GiB 主机内存、48 个 CPU、约 236 GiB 本地盘，H100 80 GB 是项目备选。SFT 沿用 Colab 系统 PyTorch/Transformers 并补充训练依赖。推理 vLLM 0.33.0 和 RL vLLM 0.32.0/TRL 1.17.1 分开冻结。WSL2 下用 Colab CLI 调度，结果同步 Drive，小体积报告回传到 `colab-reports`，SwanLab 在线记录。
- **真实工程通过项。** 32K 最坏长度全参冒烟约 142–148 秒/步，PyTorch 峰值分配 28.9 GiB，主机 RSS 峰值 145.3 GiB。强制 64 条均生成 8K 的 RL 冒烟约 474–476 秒/步、PyTorch 峰值 66.6 GiB。Flash 实际全参训练约 40–66 秒/步，16K 十步 RL 实跑约 18.5 分钟/步。全参完整本地保存实测约 113.5 GiB、386 秒，曾有 28.4 GiB 轻量镜像约 318 秒。完整镜像才支持跨 VM 恢复优化器等状态。评测支持按已落盘记录断点续跑。[冒烟与训练状态](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/cf8a805/docs/status/current_status.md)
- **结束物。** GRPO 看 `GRPO_COMPLETE.json` 的步数与恢复来源、`merged/` 的模型身份和评分报告中的样本数。调度还有心跳租约和 run identity，用于防止重复启动及不兼容恢复。CPU 测试、实际 GPU 出件、独立评测结论分别记录。
- **历史 v26。** 旧 AIMO2 LoRA SFT 的 8K 口径下 MATH-500 为 72.4%，同期 E0 八轮均值 76.6%。旧同一 8K 口径下，公开 RomEscape SFT 与 QFFT 都为 389/500，平均输出约 4,224 vs 1,936 token。旧分数受历史 prompt、生成窗口、收尾和判分口径影响，应与 v93 分列。同题原生长轨迹与教师精炼轨迹的完整效益分析仍有待收尾。[历史记录](https://github.com/XDE10/qwen2.5-7b-math-posttrain/blob/cf8a805/docs/status/history.md)

## 8. 公开配方的进展和接下来的选择

FuRA 的 **7B** 实验以 `Qwen/Qwen2.5-7B` Base 起步。最相关的是论文中标准 LoRA 对照。作者在单 H100 上跑 50 步、每步 32 题 × 8 条，r64/alpha32、AdamW、恒定 LR 1e-4、无预热、0/1 末框奖励、按序列平均损失。训练代码的 vLLM `max_tokens=1024`，总上下文 2048。表 4 的 Base→标准 LoRA 约为 MATH-500 48.4→58.5、AMC23 40.0→49.2、AIME24 5.0→11.5、AIME25 2.9→4.3。评测有贪心和多采样之分。项目 R1 是数学后训练模型，MATH-500 80.50% 来自本项目 32K mean@4 协议，跨论文的绝对分数及提升幅度无法直接映射。[FuRA 论文表 4 和附录 D.3](https://arxiv.org/html/2605.22869v1#S6.SS2) · [固定提交的启动脚本](https://github.com/yequanzhao/FuRA/blob/9c68151f3cf81bf6543bf918c011bfd98cf81220/rl/run_rl_7B.sh) · [训练源码](https://github.com/yequanzhao/FuRA/blob/9c68151f3cf81bf6543bf918c011bfd98cf81220/rl/run_rl.py)

本地 `grpo_qfft_r1_fura_lora50.json` 是迁移草稿。它从项目 R1 起步，保留 1,551 题、8×8、16K、Math-Verify 0/1、截断掩码和 token_mask，移植 r64/alpha32、七类线性投影、1e-4 恒定 LR、无预热、AdamW、序列归一化 `grpo` 损失、对称 0.2 与 50 步。CPU 微型模型检查和模拟评测已验证走通，真实 7B G4 未运行。按现有 16K 步速估算约 **15.4 G4 小时**，另计环境、检查点和评测。论文展示了单卡实现，能否从强数学起点继续提分仍需其他证据。

| 状态 | 内容 |
|---|---|
| 有证据可以收尾 | 全参 R1 QFFT 的竞赛与 MATH-500 提升，Flash QFFT 的实测负结果和混杂因素，GRPO 训练与恢复工程，30 步泛化结果和 token_mask 的长度结果。 |
| 暂缓 | 150 步正式 GRPO、FuRA 50 步迁移、6 道中等＋2 道难题的采样草案。既有授权覆盖已执行试点和评测。 |
| 预注册矩阵中尚缺 | Flash 标准 SFT 和 Flash GRPO、R1 的 32K/batch 16 双格式对照及矩阵末段分析。 |
| 当前决策 | 从公开证据里筛选是否存在能迁移到「已做数学 SFT/QFFT/蒸馏的强 7B → 数学 RL」的单卡配方。如果没有支持投入的证据，以已有结果收尾。 |

本项目的正向结果可表述为「在固定 Qwen2.5-7B-Instruct 上用 1,000 道题的 R1 轨迹完成全参 QFFT，竞赛 100 题 mean@n **26.06% → 33.63%**，按题配对 **+7.56 pp [3.88, 11.44]**」。GRPO 可以作为带独立评测和复现记录的后续实验呈现，现有准确率结果不足以声称额外提升。
