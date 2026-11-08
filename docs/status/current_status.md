# 项目一当前状态（2026-11-08）

> 本页只写「现在做到哪」的摘要；逐版本历史与精确数字见 [history.md](history.md)，仓库约定见根目录 [AGENTS.md](../../AGENTS.md)。

## 目标与主线

Qwen2.5-7B-Instruct → SFT 取得实测提升 → GRPO 继续提升，与 Qwen2.5-Math-7B-Instruct 对标，形成可写简历的结果链；另含「同一道题：原生长轨迹 vs 教师精炼轨迹」的质量/长度/成本对照，以及 QFFT 造数链。**不能预先保证涨分。**

- 固定底座：`Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- 训练数据：原始 AIMO2 `sft-4k` 的 8192 完整窗口子集，**2275 条 / 2263 题**；全 4033 条扫描，不压缩、不截断、不再限 1000 条；有/无 think 标签的原始推导都保留。
- 训练配方：**v26**（LoRA r32 / alpha64 / dropout 0.05，LR 2e-05，3 epochs，batch 2×累积 8，`max_length 8192`，逐 epoch 保存，在线 SwanLab）。
- 启动入口：**v28**（`tools/aimo/start_aimo_sft_v28.py`，自动读 SwanLab Key 后启动 v26 配方）；**v27** 是前置兼容补丁（torchao、warmup_ratio、在线认证）。

## 环境与路径

- 本地根目录：`D:/Users/Mr.Z/Desktop/项目一_7B数学后训练`；CPU 解释器 `D:/anaconda3/envs/genai-test/python.exe`。
- 训练/评测：Colab 单卡 A100 80GB；Drive 持久目录 `projects/project1`、`models`、`runs/project1`。
- Colab 工作副本 `/content/workspaces/aimo-sft-v26`；默认结果 `/content/drive/MyDrive/runs/project1/aimo_sft_v26`。
- 最新启动命令（Drive 已挂载且无同任务运行）：

```bash
python3 /content/drive/MyDrive/projects/project1/start_aimo_sft_v28.py
```

## 目录重构（2026-11-08 完成）

- 根目录已按职责重整：`configs/`（1 个现行配置）、`experiments/aimo_sft_v26/`、`scripts/{training,evaluation}/`（6 个）、`src/`（16 个模块）、`tools/{aimo,evaluation,qfft,review,verification}/`（23 个）、`tests/`（25 文件 / 172 用例）；数据、证据、报告仍在 `data/`、`artifacts/`、`reports/{current,runs}/`（git 忽略）。
- v1…v92 被取代的脚本/测试/配置/实验/报告整体移入 `archive/`（570 次移动，364 个 tracked 走 `git mv`），**未删除任何文件**；原路径可在 `archive/mapping/restructure_v2_mapping.json` 反查。
- 验证：`pytest` **172 passed / 0 failed**；`python tools/verification/verify_migration.py` 输出 `errors: []`。
- 已知兼容项：本地 `bundle_info.json`、`delivery.json` 与 v27 补丁 zip 是重构前布局的交付快照；下一次 GPU 训练前需用 `tools/aimo/package_aimo_sft_v26.py` 重新打包（生成新布局 zip 并刷新校验台账/交付记录），不要直接沿用旧包。

## 评测口径

- 训练窗口 8192 与评测生成预算 32768 分开。
- `max_new_tokens 32768`、`max_model_len 40960`、YaRN factor 4、temperature 0.6、top_p 0.95。
- AIME 2024/2025 主测，每题 64 次取平均单次正确率（pass@1 估计，不是 pass@64）；MATH-500 单次辅助；dev256 用于选 checkpoint。
- 评分走 `src/training/math_reward_v2.py` 与冻结 strict-v3；答案等价 ≠ 逐步证明正确，解析失败/不支持类别单列。

## QFFT 造数 / 审查链

- 最新为 **v91**（补跑不完整题，目标补齐 1000 完整且可入训）与 **v92**（外部深度审查包），时间 **2026-11-03**。
- 证据在 `artifacts/qfft_ds_dataset_v66/`、`artifacts/qfft_length_retry_v86/`、`artifacts/qfft_incomplete_high_v91/`、`artifacts/qfft_all_supervisor_v71/`。
- 报告在 `reports/runs/qfft_ds_dataset_20261029_v66/`、`reports/runs/qfft_incomplete_high_20261031_v91/`、`reports/runs/qfft_followups_20261029_v70/`；审查材料在 `reports/current/`。
- 生成中已实际落盘的候选不等于质量批准、不等同可入训；全部 `training_approved=false` 的边界仍成立。

## 明确未完成

- 正式双组（SFT / QFFT）训练与最终收益结论：未完成。
- 训练器完整的保存/恢复/释放闭环：未就绪（v46 容量短测、E0/SFT 评测等历史实测已完成，但不能替代正式训练验收）。
- 正式 RL 题池与 E0～E4 全套配置：未冻结；637 条件探针不是独立最终测试。
- 任一「已授权」范围之外的 GPU 任务、付费/订阅 API 扩量、数据重采：均需用户重新明确授权。

## 近期仍有效的实测结论（摘要，详见 history.md）

- SFT 评测：v41 MATH-500 原生 362/500（72.4%），本地 DS 复核 364/500（72.8%）；作者成品 s1.1-7B 同口径复核 406/500（81.2%）。
- RomEscape QFFT 成品：原生 389/500（77.8%），辅助复核 395/500（79.0%）；P-ALIGN 复核 372/500（74.4%）。
- 以上均为当时提示/预算/判分口径下的结果，不是新协议基线，也不能当作「SFT 已提分」的结论。

## 下一步（供接手者判断，非自动执行）

1. 先读 [AGENTS.md](../../AGENTS.md) 与 [README.md](../../README.md)，核对 `configs/project1_aimo_sft_v26.json`、`data/aimo_sft_v26/manifest.json` 的实际身份。
2. 在目标 A100 环境按 v28 入口核对依赖、凭证与恢复点，跑通一次完整 SFT + 评测闭环（含中断恢复）。
3. 冻结正式双组方案与 RL 题池后再谈正式扩量；不要重复全量去污/审计。
