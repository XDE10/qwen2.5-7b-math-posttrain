# reports/ —— 现行报告索引（不进 git）

只保留**当前仍被引用**的报告：`current/`（审查材料）与 `runs/`（造数/复现运行记录）。**整个目录被 `.gitignore` 忽略**，只有本 README 入库；体积约 168 MB。历史报告在 `archive/reports/`。

## current/（2 个目录）

| 目录 | 内容 |
|---|---|
| `claude_project1_review_20261031_v90/` | 供外部审查的总结、审查提示词、最新事实快照、精确教师提示词、31 文件 ZIP 与 SHA 清单 |
| `external_review_20261103_v92/` | v92 外部深度审查包（由 `tools/review/build_external_review_zip_v92.py` 生成） |

## runs/（5 个目录）

| 目录 | 内容 |
|---|---|
| `qfft_ds_dataset_20261029_v66/` | QFFT 1000 题造数主报告：分批总结、断连/熔断/恢复记录、截断重采说明、审查分片索引 |
| `qfft_incomplete_high_20261031_v91/` | 不完整题 high 补跑（v91）的运行报告 |
| `qfft_followups_20261029_v70/` | 超预算/耗尽/超长题归档：全量台账、单题目录、原文件路径 SHA、后续状态 |
| `romescape_reproduction_20261028_v57/` | RomEscape 复现与 QFFT 对照评测记录 |
| `colab_compat_20261026_v27/` | Colab 环境兼容（torchao / warmup_steps / SwanLab）问题与修复记录 |

## 查找约定

- 报告正文与 JSON 证据分开：正文/索引在本目录，原始请求/响应/候选在 `artifacts/`（同名目录）。例如造数报告对应 `artifacts/qfft_ds_dataset_v66/`。
- 报告里的分数必须带口径（提示、预算、采样次数、判分器）；`solution` 回退、答案等价、人工复核与严格金标要分开陈述。
- 历史报告（63 个按日期分组的报告目录 + `_loose/{audits,logs,snapshots}` 的 87 个散落文件）已整体移入 `archive/reports/`，**不进 git**，只作追溯；旧路径可用 `archive/mapping/restructure_v2_mapping.json` 找回。
- 旧 `reports/handover_*` 已移到 `docs/handover/`；旧 `reports/aimo_sft_20261026_v26` 已移到 `data/aimo_sft_v26/`；旧 `reports/claude_*` / `external_*` 现在 `current/` 下。
