# 项目一当前状态（2026-11-09，v93）

> 本页只写「现在做到哪」的摘要；逐版本历史与精确数字见 [history.md](history.md)，仓库约定见根目录 [AGENTS.md](../../AGENTS.md)，v93 方案见 [三周冲刺方案_v93.md](../blueprints/三周冲刺方案_v93.md)。接手当前工作（本机 Kiro IDE / WSL2）先读 [交接文档](../handover/kiro_ide_onboarding.md)。

## 目标与主线

Qwen2.5-7B-Instruct → 全参 SFT / QFFT（两位教师）→ LoRA-GRPO，与公开 checkpoint 和 Qwen2.5-Math-7B-Instruct 在同一评测口径下比较，形成可写简历、可复现的结果链。**不能预先保证涨分。**

- 固定底座：`Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- 实现：[experiments/qfft_v93/](../../experiments/qfft_v93/README.md)。研究问题、配方、判定规则已预注册在方案文档里。

## GPU 上已跑通（2026-11-09，用户在 WSL2 上用 Colab CLI 运行，结果由 VM 推到 `colab-reports` 分支）

- `ping:30`（CPU）、阶段 0（CPU，约 11 分钟）：5 个评测集与 4 个模型（base、qwen_math、qfft_official、sft_repro）已在 Drive。
- 阶段 1 最坏长度冒烟（G4 = RTX PRO 6000 Blackwell 96 GB，主机内存 177 GiB，48 CPU，41 分钟，`colab-reports` commit `d6a9608`）：全参 16 × 32,768 token 每步 148/142 s，显存峰值已分配 28.9 GiB / 保留 42.6 GiB，主机 RSS 最高 145.3 GiB；本地存检查点 113.5 GiB 用 386 s；评测 24 条全部保存，vLLM 启动 217 s；GRPO（64 条都强制到 8192 token）每步 476/474 s，显存 66.6 GiB；SwanLab 在线。这些是冒烟（2 步）的数字，不是正式训练结果。

## 本地数据

- DeepSeek-V4.1-Flash 轨迹 977/1000 可用（用户本机 `data_build status`）。23 题在 v66/v86/v91 与 v93 的 8 次 high/max 尝试里都超出 32K 窗口或被截断；处理方案（先 `reasoning_effort=low` 追加尝试，仍缺的成对剔除）见交接文档第 2 节，`teacher fill` 由用户在本机启动（付费 API）。
- 用户在预览数据上看到的教师对比（728 道可核对题，Math-Verify）：DS 66.8%（486/728，平均约 4,181 token），R1 63.6%（463/728，平均约 9,810 token）。正式数字以补齐或剔除后 `data_build build` 的 `teacher_report.md` 为准。

## v93 已完成（代码与 CPU / 模拟验证）

- 数据：`fetch_benchmarks.py`（5 个评测集固定 revision + SHA-256）、`data_build.py`（1000 题计划、准入、两份训练集、教师报告；`--paired-exclusion` 成对剔除、每行轨迹的来源统计）、`teacher.py`（DeepSeek 补齐、可选答对重采样、`--efforts low`）。
- 去污：13-gram + 相似度对 5 个评测集检查 s1K-1.1，17 对命中已逐一复核；`s1k11/row/0918` 与 MATH-500 `test/intermediate_algebra/1063.json` 同题，v93 计划中替换。
- 训练与评测：`train_sft.py`（全参 SFT/QFFT，ZeRO-2 offload、Liger、本地检查点 + Drive 后台镜像 + 自动恢复，每 2 个 epoch 存一次）、`grpo.py` + `merge_lora.py`、`generate.py`、`score.py`、`analyze.py`。
- 调度：`colab/launch.sh`、`pipeline.sh`、`remote.py`、`stages.py`（断线不中止、心跳租约、GPU 回退、rclone 挂载、报告推送到 `colab-reports`）；阶段 `d` 拒收预览构建；rclone 上传在 Drive 每日上限处立即报错。
- 本机工具：`local_data.sh`（造数五步）、`colab/setup_wsl.sh`（安装与检查）、`check_local.sh`（只读自检）。
- 测试：`tests/v93_harness/`（从云端沙箱迁入仓库）8 组 214 项全部通过，在工作区快照上运行；pytest 原有结果不变（缺本地数据的用例除外）。

## 明确未完成

- 23 题的 low effort 追加尝试与最终 `data_build build`；阶段 `d`（上传数据）。
- 阶段 2（锚点评测）、3（`qfft_ds`、`sft_ds`）、4（GRPO）、5（R1 对）、6、7 均未运行；真实数据下的 s/step、正式评测吞吐、113.5 GiB 镜像的上传耗时、H100 上的 GRPO 显存都没有实测。
- 任何结论（QFFT vs SFT、教师对比、RL 收益）都还没有数据。

## 历史结论（仍有效，口径与 v93 不同，不能当 v93 基线）

- v26 LoRA SFT（AIMO2、8K 窗口）MATH-500 原生 72.4%，低于 E0 同预算 8 轮均值 76.6%；dev256 32K 贪心有 122/256 截断。
- 同一 8K 口径、原生判分下，公开 RomEscape SFT-7B 与 QFFT-7B 都是 389/500，平均输出 4,224 vs 1,936 token。
- 严格判分（strict-v3）在 dev256 上比原生 Math-Verify 低 18 题。
- R1 教师在 s1K-1.1 的 742 道可核对题上答对 62.7%；R1 response 最长 26,835 token，32K 窗口可全部容纳。

## 下一步

见 [交接文档第 4 节](../handover/kiro_ide_onboarding.md)：阶段 2（不依赖数据，现在就能跑）→ 23 题 low effort 补齐 → `data_build build` → 阶段 `d` → 阶段 3（第一个训练的 s/step 出来后重算耗时与计算单元）→ 阶段 4、5。
