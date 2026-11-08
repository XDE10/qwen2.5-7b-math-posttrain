# 项目一当前状态（2026-11-08，v93）

> 本页只写「现在做到哪」的摘要；逐版本历史与精确数字见 [history.md](history.md)，仓库约定见根目录 [AGENTS.md](../../AGENTS.md)，v93 方案见 [三周冲刺方案_v93.md](../blueprints/三周冲刺方案_v93.md)。

## 目标与主线

Qwen2.5-7B-Instruct → 全参 SFT / QFFT（两位教师）→ LoRA-GRPO，与公开 checkpoint 和 Qwen2.5-Math-7B-Instruct 在同一评测口径下比较，形成可写简历、可复现的结果链。**不能预先保证涨分。**

- 固定底座：`Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- 实现：[experiments/qfft_v93/](../../experiments/qfft_v93/README.md)。研究问题、配方、判定规则已预注册在方案文档里。

## v93 已完成（代码与 CPU 验证）

- 数据：`fetch_benchmarks.py`（5 个评测集固定 revision + SHA-256，MATH-500 与项目封存摘要一致）、`data_build.py`（1000 题计划、准入、两份训练集、教师报告）、`teacher.py`（DeepSeek 补齐与可选的答对重采样）。
- 去污：13-gram + 相似度对 5 个评测集检查 s1K-1.1，17 对命中已逐一复核（`experiments/qfft_v93/configs/decontam_review_v93.json`）；新发现 `s1k11/row/0918` 与 MATH-500 `test/intermediate_algebra/1063.json` 同题，v93 计划中替换。
- 训练：`train_sft.py`（全参 SFT/QFFT，ZeRO-2 offload、Liger、本地检查点 + Drive 后台镜像 + 自动恢复）、`grpo.py` + `merge_lora.py`（TRL GRPO + LoRA + 同卡 vLLM）。
- 评测：`generate.py`（连续批处理、逐条落盘、按采样粒度续跑）、`score.py`（最终段判分、mean@n/pass@k/maj@n、配对 bootstrap、results.md）、`analyze.py`（五类图）。
- 调度：`colab/launch.sh`、`colab/pipeline.sh`、`colab/remote.py`、`colab/stages.py`。
- 修复的 P0：`src/training/approved_hf.py` 与 `src/training/math_reward_v2.py` 的判分 worker 路径（11-08 重构后失效，approved_hf 会静默把所有奖励算成 0），并改为基础设施失败直接报错。
- CPU 实跑验证：模板与官方 S1_QFFT 数据逐字一致（1000/1000）；判分器 10 个边界用例；微型模型上的全参训练、模拟 VM 丢失后从 Drive 恢复续跑、身份校验；假引擎生成续跑与判分；TRL GRPO + LoRA + 恢复 + 合并；合成数据上的 plan/status/fill/build；launch.sh/remote.py 调度流程。原有测试结果不变（本环境缺本地数据导致的 1 failed + 4 errors 与改动前相同）。
- 参照数字：R1 教师在 s1K-1.1 的 742 道可核对题上答对 62.7%；R1 response 最长 26,835 token，32K 窗口可全部容纳。

## 明确未完成

- 本地：`fetch_benchmarks`、`data_build plan/status`、`teacher fill`（补齐 DeepSeek 到 1000/1000）、`data_build build` 都还没在工作站上跑；教师报告与是否需要 `ds_verified` 待定。
- GPU：冒烟、锚点评测、四个全参 run、GRPO 均未运行；DeepSpeed CPU-Adam 速度、32K 峰值显存、vLLM 在 G4（Blackwell）上的吞吐、122 GB 检查点镜像到 Drive 的耗时都要在阶段 1 冒烟里实测。
- 任何结论（QFFT vs SFT、教师对比、RL 收益）都还没有数据。

## 历史结论（仍有效，口径与 v93 不同，不能当 v93 基线）

- v26 LoRA SFT（AIMO2、8K 窗口）MATH-500 原生 72.4%，低于 E0 同预算 8 轮均值 76.6%；dev256 32K 贪心有 122/256 截断。
- 同一 8K 口径、原生判分下，公开 RomEscape SFT-7B 与 QFFT-7B 都是 389/500，平均输出 4,224 vs 1,936 token。
- 严格判分（strict-v3）在 dev256 上比原生 Math-Verify 低 18 题。

## 下一步

1. 工作站：`python -m experiments.qfft_v93.fetch_benchmarks --sources` → `data_build plan` → `data_build status` → `teacher fill --fallback-official` → `data_build build --ds-verified`，读 `data/qfft1000_v93/teacher_report.md`。
2. WSL2 上装 Colab CLI，`pipeline.sh 0` → `pipeline.sh 1 --gpu G4`（或 H100），核对 s/step、峰值显存、主机内存后再开阶段 2–5。
