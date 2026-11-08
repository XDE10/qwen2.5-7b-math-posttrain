# data/ —— 本地数据（不进 git）

训练与评测用的数据、tokenizer 与发布包。**整个目录被 `.gitignore` 忽略**，只有本 README 入库；数据本体只在本地，不做远程备份承诺。

## 顶层内容

| 路径 | 大小 | 内容 |
|---|---|---|
| `aimo_sft_v26/` | 约 31 MB | 现行 AIMO SFT 训练 + 评测数据（原 `reports/aimo_sft_20261026_v26` 整体移入） |
| `math/` | 约 27 MB | 全量数学数据（gsm8k / math / MATH-500 等） |
| `portable_release/` | 约 22 MB | 便携发布包：raw/compact 配对、RL 优先级、保留 dev id 清单 |
| `tokenizers/` | 约 51 MB | 5 个 tokenizer 快照（用于离线复现 token 统计） |
| `math_manifest.json` | — | 数学数据集的来源、去污与用途清单 |

## aimo_sft_v26/

| 文件 | 大小 | 说明 |
|---|---|---|
| `train.jsonl` | 31,991,127 B | 8192 完整窗口子集，**2275 条 / 2263 题**；有/无 think 标签的原始推导都保留 |
| `manifest.json` | 1,161 B | 批准训练文件、条数、筛选规则（`scanned_until_target 4033`、`selection_seed 42`、数据源 `imagination-research/aimo2-datasets sft-4k` rev `237f575f59fda9b79de5f4f9e9c7860099f84349`） |
| `validation.json` | — | 该批数据的校验结果 |
| `dev.jsonl` | 72,252 B | dev256 选模集 |
| `math500.jsonl` | 136,567 B | MATH-500 评测题（单次辅助） |
| `aime2024.jsonl`、`aime2025.jsonl` | — | AIME 主测，每题 64 次 |
| `evaluation_sources.json` | 1,271 B | 评测题来源与身份 |
| `bundle_checksums.json` | 15,437 B | 现行 bundle 逐文件 SHA 台账，`experiments/aimo_sft_v26/run.py` 启动时校验；2026-11-08 目录重构后按新路径重算（重构前副本存 `archive/reports/_loose/audits/bundle_checksums_pre_refactor_20261108.json`） |
| `bundle_info.json`、`delivery.json` | — | 2026-10-26 Colab 交付快照（zip 体积/SHA/Drive 文件 ID），保留原样作历史记录；下次跑 GPU 前用 `tools/aimo/package_aimo_sft_v26.py` 重新打包会刷新它们 |
| `training_example.txt` | 11,004 B | 一条完整训练样本（含完整 think 轨迹）示例 |
| `训练与评测说明.md` | 6,105 B | 数据构建与评测口径说明 |

## math/ 与 portable_release/

- `math/`：`gsm8k_train.jsonl`、`gsm8k_test.jsonl`、`math_train.jsonl`、`math_dev.jsonl`、`math500_test.jsonl`、`math_source_test.jsonl`、`math_quarantine.jsonl`。
- `portable_release/`：`manifest.json`、`project1_raw_paired.jsonl`、`project1_compact_paired.jsonl`、`project1_rl_priority_not_rollout_selected.jsonl`、`reserved_dev_ids_never_train.json`。1500 raw / 1500 compact 是同一批题的两臂，不是 3000 道题。

## 准入与边界

- 训练/评测范围以对应 `manifest.json` 的 `approved_training_files` / `approved_evaluation_files` 与逐行授权为准；`math_manifest.json` 同样记录 `approved_training_files` / `approved_evaluation_files` / `quarantined_files`。
- 候选 ≠ 批准；AI 审核或答案等价 ≠ 人工推导金标。不能手改批准字段或哈希绕过准入。
- 最终测试题不能变成生成种子或选模依据；`reserved_dev_ids_never_train.json` 列出永不训练的 dev id。
- 历史数据（25 项，含 project1/2/3 各代批次、旧 distillation / strict / topup 数据集）已整体移入 `archive/data/`，**不进 git**，只作追溯；旧代码引用的 `data/...` 路径在其中已失效，可用 `archive/mapping/restructure_v2_mapping.json` 找回原名。
