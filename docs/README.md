# docs 索引

项目文档集中在此，按用途分层；**当前状态与协作约定以 [status/current_status.md](status/current_status.md) 和根目录 [AGENTS.md](../AGENTS.md) 为准**，本目录其余部分是方案原文、迁移/运维记录与历史。

| 目录 | 内容 | 何时读 |
|---|---|---|
| [status/](status/) | [status/current_status.md](status/current_status.md)（当前主线/状态摘要）、[status/history.md](status/history.md)（v1…v93 历史条目，原 AGENTS.md 长历史整体迁移） | 想快速了解「现在做到哪」，或追溯某版本细节 |
| [blueprints/](blueprints/) | 研究方案、共用规范、数据封板与上卡说明、批准配置留档（[blueprints/configs/approved_release_20261024/](blueprints/configs/approved_release_20261024/)）；现行方案 [blueprints/三周冲刺方案_v93.md](blueprints/三周冲刺方案_v93.md) | 需要追溯「当初为什么这样设计」 |
| [interview/](interview/) | [interview/面试追问与参考答案.md](interview/面试追问与参考答案.md)：简历条目、STAR 讲述、约 50 道追问与参考答案、数字速查 | 准备面试 |
| [handover/](handover/) | 6 个 `handover_*` 交接目录（原 `reports/handover_*`） | 需要历次对话的完整接手材料 |
| [migration/](migration/) | 迁移/清理/重构记录、[migration/restructure_v2_plan.json](migration/restructure_v2_plan.json)、可重放的 [migration/restructure_v2_move.py](migration/restructure_v2_move.py)、[migration/pytest_baseline_before_restructure.txt](migration/pytest_baseline_before_restructure.txt) | 追溯文件删除/移动理由与统计 |
| [metadata/](metadata/) | [metadata/migration_manifest.json](metadata/migration_manifest.json)（文件大小与 SHA 清单）、[metadata/requirements-cpu-snapshot.txt](metadata/requirements-cpu-snapshot.txt)（历史 CPU 环境）、Colab 文档校验清单 | 校验完整性、复现历史环境 |
| [operations/](operations/) | CPU 夜间运行手册 | 需要跑离线数据工程 |

## 与根目录文档的关系

- [AGENTS.md](../AGENTS.md)：仓库约定、当前主线摘要、阅读入口、目录地图。
- [README.md](../README.md)：工程入口——目标目录结构、代码地图、数据边界、测试与归档说明。
- [status/history.md](status/history.md)：历史记录，只作追溯；冲突时以 AGENTS.md + current_status.md + 最新报告为准。
- `blueprints/` 中的方案是 2026-10 的原稿，可能含已被后续实测取代的数字与旧路径；旧路径可用 `archive/mapping/restructure_v2_mapping.json` 找回。

## 维护约定

- 移动/重命名本目录内文件时，同步 `docs/metadata/migration_manifest.json` 中的路径与大小，并运行 `python tools/verification/verify_migration.py` 保持 `errors: []`。
- 重构/清理必须留「原因 + 映射 + 统计」三件套，参考 [migration/重构记录_20261108_v2.md](migration/重构记录_20261108_v2.md)。
- 新增历史条目不要塞回 AGENTS.md：写入 `status/history.md`，AGENTS.md 只保留长期约定。
