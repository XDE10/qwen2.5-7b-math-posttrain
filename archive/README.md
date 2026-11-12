# archive/ —— 只读历史快照

本目录存放被取代的脚本、测试、配置、实验、数据、报告与交接材料。目录重构 v2（2026-11-08）把不再属于现行主线的内容移到这里保存。

> **archive/ 是只读历史快照，不保证可直接运行。** 内部相对路径、数据位置、配置键名都可能与当前结构不一致；需要运行旧代码时，请先按 mapping 找回原名并自行判断依赖是否还存在。

## 结构

| 子目录 | 内容 | 规模 |
|---|---|---|
| `configs/` | 各代训练/评测/smoke 配置 | 20 个文件 |
| `scripts/` | 各代 AIMO / project1 / project2 / project3 训练、评测、造数、审查入口 | 134 个文件 |
| `tools/` | 各代启动器、评测器、造数/审查/监控工具 | 40 个文件（另有 2 个 `start_aimo_sft_v26*` 已被移回 `tools/aimo/`） |
| `src/` | 各代库模块（数据处理、奖励、训练、验证器） | 59 个文件 |
| `tests/` | 各代测试（重构前基线 583 passed / 12 failed 中已归档的部分） | 70 个文件 |
| `experiments/` | `aimo_sft_v25/`、`projection_pilot_v21/v23/v24/` 等旧实验包 | 4 个目录 / 23 个文件 |
| `data/` | 25 个历史数据批次（project1/2/3、旧 distillation/strict/topup 等） | 2867 个文件，**不进 git** |
| `reports/` | 63 个按日期分组的报告目录 + `_loose/{audits,logs,snapshots}` 共 87 个散落文件 | 1174 个文件，**不进 git** |
| `docs/` | [旧交接包](docs/handover/)和 [Kiro 配置快照](docs/kiro/steering/v93-local-execution.md) | [2026-11-12 移动记录](../docs/migration/handover_cleanup_20261112.json) |
| `mapping/` | `restructure_v2_mapping.json`：本次 570 次移动的 `from` / `to` / `tracked` | 1 个文件 |

## 如何找回原路径

1. 打开 `mapping/restructure_v2_mapping.json`，它是一组记录，每条含：
   - `from`：移动前路径（相对仓库根，如 `reports/qfft_ds_dataset_20261029_v66/...`）；
   - `to`：移动后路径（如 `archive/reports/2026-10-29/...`）；
   - `tracked`：是否被 git 跟踪（`true` 为 `git mv`，`false` 为未被跟踪的历史文件，移动后仍不入库）。
2. 反向查找：在 JSON 里搜索你记得的旧文件名或旧目录名，取对应 `to`。
3. 规划文件 `docs/migration/restructure_v2_plan.json` 含同源 `moves` 列表；重放脚本 `docs/migration/restructure_v2_move.py` 支持 `--dry`（只打印）与 `--run`（执行）。
4. 更早的文档层移动（`project_blueprints→docs/blueprints` 等）见 `docs/migration/重构记录_20261108.md` 与 `docs/metadata/migration_manifest.json`。

## 使用原则

- **不要从 archive 里直接跑训练/评测**：路径、数据、配置可能已失效，且结果无法与现行基线比较。
- **不要把 archive 内容复制回顶层**「让旧脚本跑通」；要恢复某能力，应在现行结构里重写并留测试。
- 只读：不要在 archive 内新建/覆盖证据；新证据写 `artifacts/`、新报告写 `reports/`。
- `archive/data/` 与 `archive/reports/` 被 `.gitignore` 忽略，只存在于本地，不做备份承诺。
- 归档不改变历史事实：旧报告里的数字与结论按当时口径理解，不要用当前口径重算覆盖。
