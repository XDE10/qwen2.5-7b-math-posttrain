# docs 索引

项目文档集中在此，按用途分层；**最新状态与协作约定始终以根目录 [AGENTS.md](../AGENTS.md) 为准**，本目录存放的是方案原文、迁移与运维记录。

| 目录 | 内容 | 何时读 |
|---|---|---|
| [blueprints/](blueprints/) | 研究方案、共用规范、数据封板与上卡说明、批准配置留档（`configs/approved_release_20261024/`） | 需要追溯"当初为什么这样设计"时 |
| [migration/](migration/) | 迁移验收（`迁移验收.md`）、清理记录（2026-11-08 三批）、重构记录（2026-11-08） | 需要追溯文件删除/移动理由时 |
| [metadata/](metadata/) | `migration_manifest.json`（文件大小与 SHA 清单）、`requirements-cpu-snapshot.txt`（历史 CPU 环境）、Colab 文档校验清单 | 校验完整性、复现历史环境时 |
| [operations/](operations/) | CPU 夜间运行手册 | 需要跑离线数据工程时 |

## 与根目录文档的关系

- [AGENTS.md](../AGENTS.md)：项目身份、协作约定、最新批复与未完成事项（会持续更新）。
- [README.md](../README.md)：工程入口——目录结构、数据边界、命令与历史基线快照。
- 本目录 `blueprints/` 中的方案是 2026-10 的原稿，可能含已被后续实测取代的数字；冲突时以 AGENTS.md 与 `reports/` 最新报告为准。

## 维护约定

移动/重命名本目录内文件时，需同步 `metadata/migration_manifest.json` 中的路径与大小，并运行 `python tools/verify_migration.py` 保持 `errors: []`。
