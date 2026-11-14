# 给 ChatGPT 6 Pro 的数学 RL 配方调研提示词

请阅读随附的《项目一 v93 数学后训练现状与实验总结》。我要决定，这个项目还值得跑一次数学 RL，还是现在就用已经验证的 QFFT 成绩收尾。请直接浏览 **GitHub、Hugging Face、论文原文和作者公开训练日志**，找出实际可查证的配方，替我作一个有依据、可以执行的决定。输出用中文，保留模型 ID、参数名和原始指标名称。

## 我们的项目和已完成结果

我们固定从 `Qwen/Qwen2.5-7B-Instruct`（revision `a09a35458c702b33eeacc393d103063234e8bc28`）出发。以 1,000 道题的 s1K-1.1 R1 教师长轨迹做全参 QFFT。QFFT 只训 response，去掉 chat 标记和终止 EOS。训练 16,384 窗口，有效 batch 8，AdamW，LR 1e-5，cosine，10% warmup，6 epochs/750 步。最终权重是 `qfft_r1_16k_b8_restart_v93`，下面称原始 R1。这个名字指我们的 Qwen Instruct 后训练模型。

| 项目内受控评测 | 结果 |
|---|---|
| 竞赛 100 题（AIME24/25 各 30，AMC23 40） | 底座 26.06%，原始 R1 33.63%，**+7.56 pp，逐题配对 95% CI [+3.88, +11.44]**。公开 QFFT 锚点 35.56%，Qwen2.5-Math-7B-Instruct 29.38%。 |
| MATH-500 | 底座 76.80%，原始 R1 80.50%，数学专用锚点 83.40%。 |
| MATH 开发 249 题，mean@2 | sequence 第 10 步相对 R1 +1.00 pp [−1.61, +3.61]。第 30 步 +1.41 pp [−1.00, +4.02]。 |
| 竞赛 100 题，第 30 步 | sequence 第 30 步相对 R1 **−1.50 pp [−3.75, +0.75]**。AMC23 −4.69 pp [−9.69, 0.00]，大部分净差值来自双方均未截断的回答，重判复现原对错。 |
| 独立竞赛 59 题，mean@8 | 原始 R1 139/472（29.45%），sequence 10 步 142/472（30.08%），token 10 步 143/472（30.30%）。token 比 sequence **+0.21 pp [−2.54, +2.97]**，比原始 R1 **+0.85 pp [−2.33, +4.03]**。 |
| 59 题输出长度 | 原始 R1 10,760、sequence 10,399、token 8,656 token/回答。token 相对 sequence 少约 1,744 token（16.8%）。按题配对长度差 95% CI [−2,682, −842] token。 |

项目评测使用原生 32,768 上下文（prompt 加生成），temperature 0.6、top_p 0.95，Math-Verify 0.9.0 的最终答案规则。AIME24/25 为 mean@16、AMC23 mean@8、MATH-500 mean@4，先按题取平均再等权聚合。Qwen2.5-Math 锚点保留自己的 4K 原生窗口。100 题和 59 题成绩都已经看过，后续可以对标历史结果。它们无法充当根据这些成绩选方案后的全新最终盲测。

已有 LoRA-GRPO 试点用固定获批题池 1,551 道（918 道旧 E0 中等题、633 道旧 E0 四次全错题），每步 8 道 × 每道 8 条生成，LoRA r32/alpha32/dropout 0/all-linear，LR 1e-5、5 步 warmup 后恒定，DAPO token-normalized loss，ε_low 0.20/ε_high 0.28，β=0，`scale_rewards=none`，每批更新 1 次，seed 42。Rollout 用 16,384 completion + 1,024 prompt 上限，temperature 1.0、top_p 1.0、截断掩码、最后完整答案框 0/1 奖励。使用 TRL 1.17.1 + vLLM 0.32.0 colocate。可用硬件是一张 Colab G4（RTX PRO 6000 Blackwell 96 GB），H100 80 GB 可作备选。16K 的一次 64 rollout 更新约 18.5 分钟，另计环境、检查点与评测。

已做过的诊断也需要纳入判断。答案框回退把同批 `no_answer` 从 113/128 降到 30/128。固定题组的 8K/12K/16K 截断分别为 46/128、31/128、9/128。sequence 前 30 步里，E0 难题用了 64.3% 的生成 token，全零组 49/113，中等题是 6/127。一次无更新的 IS 诊断发现某些未截断长回答的序列权重很小。后来的独立留出对照观察到 token 方案输出明显变短，准确率差值区间跨零。目前没有受控证据确认 LR 1e-5 或难题比例导致了掉分。

前一轮提到 [FuRA](https://arxiv.org/abs/2605.22869)。它确实有 7B 单卡对照，起点是 `Qwen/Qwen2.5-7B` **Base**，标准 LoRA 的 MATH-500 大致 48.4%→58.5%。作者实际训练源码约 1K 生成、32 题 × 8 采样、50 步、r64/alpha32、恒定 LR 1e-4。我们已有借鉴它的 50 步配置草稿，仍用原始 R1、固定题池、16K、8×8，只有 CPU 检查。以现有步速约需 15.4 G4 小时。只有迁移到强数学起点的证据足够，才值得考虑这类投入。请独立核对论文和固定版本源码。[FuRA 7B 启动脚本](https://github.com/yequanzhao/FuRA/blob/9c68151f3cf81bf6543bf918c011bfd98cf81220/rl/run_rl_7B.sh) · [训练源码](https://github.com/yequanzhao/FuRA/blob/9c68151f3cf81bf6543bf918c011bfd98cf81220/rl/run_rl.py)

## 1. 找什么项目

按相似程度分层，重点覆盖 6–9B 数学模型。更大或更小模型的研究可以提供机制证据，列出硬件、rollout 长度和起点差异。

1. **最贴近**是 Qwen2.5-7B-Instruct 先做数学 SFT、QFFT、长 CoT 或 R1 蒸馏，再对同一个 checkpoint 做可验证数学 RL。优先长推理轨迹、GRPO/RLOO/REINFORCE++ 或类似策略梯度，且作者给了明确的 RL 前后同口径留出分数。
2. **其次**是从 7B 数学 Instruct、可确认已做数学 SFT 的公开 checkpoint、R1-Distill-Qwen-7B 或其他强 6–9B 数学模型开始。确认该起点是否也做过 RL。我们自己的起点仍固定为原始 R1。
3. **工程参照**包括普通 Instruct 直接 RL、纯 Base RL、1.5B/14B/32B、需要多卡的配方。用来参考成本和机制，单列其所能支持的结论。
4. **主动找零收益或退化。** 找和我们更接近但 RL 后几乎持平、只降长度、只提高 pass@1 却伤 pass@k、只涨训练集或较容易题的公开结果。跟踪后续复现、issue、作者说明的评测偏差或反例。

分别从代码仓库、模型卡/训练集和论文三条路径追溯来源。可先组合这些检索词，再沿候选模型的上游和下游继续找。

- `"Qwen2.5-7B-Instruct" "SFT" "GRPO" "AIME" before after`、`"Qwen2.5-7B" "continued RL" "MATH-500" LoRA`、`"Qwen2.5-Math-7B-Instruct" "GRPO" warm start`。
- `"DeepSeek-R1-Distill-Qwen-7B" "GRPO" "AIME" baseline`、`"R1-Distill-Qwen-7B" "RL" "AMC23" training config`、`"s1.1-7B" "QFFT" "GRPO" reinforcement learning`。
- `"math SFT checkpoint" "RLVR" "7B" "AIME"`、`"math post-trained" "GRPO" "single GPU" qwen`、`"7B" "LoRA" "GRPO" "H100" "AIME" "16k"`，以及同义算法名称和中文关键词。
- 在 GitHub 搜索实际入口的 `model_name_or_path`、`actor_rollout_ref.model.path`、`init_model`、`sft_model_path`、`num_generations`、`max_completion_length`、`clip_ratio_high`、`loss_type`、`lora_rank`。从 Hugging Face 的 `base_model`、`adapter_config.json`、训练配置和 `trainer_state.json` 反查。模型卡和脚本冲突时，追溯到实际 run 的参数或日志。
- 起始线索有 [TinyLoRA](https://arxiv.org/html/2602.04118)、[Open R1](https://github.com/huggingface/open-r1)、[FutureMa](https://huggingface.co/FutureMa/Qwen2.5-7B-Instruct-GRPO-Math)、[thawait](https://huggingface.co/thawait/qwen2.5-7b-math-reasoning-grpo)、[PRIME/Eurus-2-7B](https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME)、[SimpleRL](https://github.com/hkust-nlp/simpleRL-reason)、[Dr. GRPO](https://github.com/sail-sg/understand-r1-zero)、[Light-R1](https://arxiv.org/abs/2503.10460) 和 [Tina](https://arxiv.org/abs/2504.15777)。请再找这些入口以外的作者项目和可信复现。重点核对 PRIME 的奖励、TinyLoRA 的实际起点及算力、Open R1 的已执行 run 与样例配置、Tina 和 Light-R1 的实际模型尺寸、FutureMa 有无配对留出成绩、thawait 的数学提升发生在哪个训练阶段。

## 2. 对每个候选查哪些证据

主表可以列 6–12 个有明确取舍的候选，证据不足时少列。对最贴近的两三个做源码级核查。优先引用作者论文表格、固定 commit 的脚本/配置行、确切 Hugging Face revision 和训练/eval 输出。每个关键数字附近附上能直接证明它的链接。分别标明作者报告、第三方复现和基于源码的推断。未知参数写「未公开」或「无法确认」，区分代码默认值、样例配置和实际 run 生效值。

- **起点和对照。** 列精确 checkpoint ID、参数量、revision、Base/Instruct/数学 SFT/蒸馏/既有 RL 历史。对照必须追溯到同一初始权重。模型名称里有 7B、HF 的标签或项目介绍不足以证明实际起点。
- **数据和奖励。** 列训练数据、revision、规模、数学分布、过滤规则、答案格式、训练/验证/测试去污、按当前 policy 筛题或动态过滤的机制，以及全零/全一组如何处理。说明碰到截断、无箱、解析失败、超时的奖励和 loss 权重。明确有无长度、格式、过程奖励。
- **优化和采样。** 列框架版本、训练入口、全参/LoRA、r/alpha/target/量化、优化器、LR/调度/预热/weight decay、步数或训练 token、有效组数 × 每组生成数、micro batch/梯度累积、同一个 rollout 的更新次数、算法与 loss 归一化、reward/advantage 缩放、clip 方向与范围、KL/参考策略、IS 校正、截断掩码、train prompt/生成上限、真实平均生成长度、temperature/top_p 和 seed。同名参数也可能有不同语义，必要时引用实际 trainer 实现和配置覆盖链。
- **评测。** 列初始权重→RL 最终权重在各个竞赛数据集上的起点、终点和绝对 pp。注明题数、采样数、seed 数、prompt、温度、预算、判分器、数据版本、评测是否用同一题、checkpoint 是最后还是按验证集选优。分别记录 greedy、mean@n、pass@k、maj@n。找配对原始结果或置信区间，只有公开数据足够时才重算。单一分数或不同 seed 的均值无法给出逐题稳定性。
- **开销和迁移。** 列卡型号/数量、GPU 小时、生成数或 token 数、CPU offload、生成/训练是否共卡、显存峰值。区分单卡优化和整套训练确实在单卡完成。没有 GPU 时长时可按公开步数和 rollout 估计，写出假设和误差来源。比较我们的 8×8、16K 与外部方案的实际吞吐。
- **支持力度。** 对每个候选回答三个问题。是否真实展示强数学起点的增量。涨分是否来自同一题集及推理预算。若只移植到单卡、保持当前题池、检查点和评测，需要改变哪些东西。说明哪些外部结果无法外推到本项目。

禁止相减不同评测协议下的 48.4% 与 80.50%。任何因果判断都要说明实验控制。留意训练集包含 benchmark 原题、数据难度变化、budget forcing、模型模板差异、只公布最佳 checkpoint 或挑选评测 seed，逐项解释它们对候选结论的影响。

## 3. 给一个有截止条件的决定

先按证据排序，解释现有日志对奖励解析、长度、IS 和题池效率分别证实了什么。请回答「有没有人从与我们类似的强数学 7B 起点，在真实单卡预算下把 AIME/AMC/MATH 提分做出来？」和「最接近但失败的公开对照是什么？」。给至多两个候选，最终只选 **一次有上限的迁移** 或 **现在收尾**。如果资料仅支持纯 Base 提分，或者完整方案要用超出条件的硬件，请把它写进决策依据。

若推荐一次迁移，给可以人工审核的计划：

1. 从原始 R1 QFFT 权重初始化。保留现有获批题池与 LR 1e-5 的当前执行约束。若最有根据的配方需要换学习率、采样或题池，写出哪项需要我另行同意，分析维持原约束后迁移还剩哪些改动。并排给出公开原版与本项目迁移设置，列最小参数差异、各字段的公开依据和要观察的指标。
2. 固定最终步数、最大 GPU 小时、生成预算、检查点频率和一次验收的入口。用约 18.5 分钟/步作预算基准，说明哪些改动会改变步时，单计评测成本。如果效果只能靠正式 150 步或多轮扫参才能判断，评估证据是否值这笔成本。当前 150 步暂缓。
3. 预先固定题集、主指标、按题配对 bootstrap CI、最小有实质意义的点估计、可接受的回退幅度及结论规则。既有 59/100 题可以作历史对标。若需另建最终留出，只说明如何从独立竞赛源查重并冻结，检查样本量与可分辨的效果量。复用原始 R1 生成时逐项对齐模型、题、prompt、生成参数、scorer。健康监控看训练指标，能力验收看预先指定的最后一步。
4. 给出增益证据够、数据仍不确定、出现退化三种情况下的最终写法、产物和是否收尾。单训练种子的逐题置信区间刻画固定这对权重在题目层面的不确定性。公开方案如果使用多个 seed，说明单种子对「稳定提分」的证据少了哪一部分。

若选择收尾，用已验证的 QFFT 和 GRPO 证据给两条可放进报告或简历的项目描述，明确哪个权重作为最终交付。说明什么新外部证据才足以改变决策。无需继续盲扫。

## 4. 输出结构与完成标准

1. **结论摘要。** 一屏内给出是否有真正接近的项目、哪项证据可迁移、最终选择、需要多少 GPU 小时。
2. **证据总表。** 每行包含起点类型、前后分数及同口径程度、有效配方、硬件、证据质量与直接链接。列出最容易误认成近似项目的候选及排除原因。
3. **最接近配方逐项还原。** 用固定版本 GitHub、HF、论文和日志复核实际起始 checkpoint、训练 CLI/YAML、配置覆盖、评测表。发现文件、文档或日志冲突时，在相应行标明。
4. **本项目的唯一决定。** 给有限计划或收尾结论，明确已有证据和所依赖的外部证据。交付前复核关键来源，确保没有把终点当起点、Base 当 Instruct、pass@k 当 mean@n、训练正确率当泛化、CPU 验证当 7B GPU 结果。

这一轮是只读调研。附件足以完成项目情况对照，项目内部 GitHub 链接可能需要权限。若当前对话已连接获准的工作区，可补看下列仓库相对路径。内部链接不可访问时，标明内部事实来自附件，继续公开检索。

- `docs/status/current_status.md`、`docs/blueprints/三周冲刺方案_v93.md`、`experiments/qfft_v93/README.md`、`experiments/qfft_v93/configs/pipeline.json`。
- `experiments/qfft_v93/configs/grpo/` 中的 `_defaults.json`、`grpo_qfft_r1_16k_pilot.json`、`grpo_qfft_r1_16k_tokenmask10.json`、`grpo_qfft_r1_fura_lora50.json`、`r1_competition_2026.json`。
- `colab-reports` 分支 `40c2f8a` 的 `v93/latest/r1_competition_2026/results.json`，以及总结里链接的固定版本五套评测、249 题评分和 30 步竞赛评测。

请在这一轮给出完整报告。检索做到足以支撑投入或收尾的决定即可。公开证据回答不了的部分，请指出缺少哪一项关键证据，并据此作出选择。
