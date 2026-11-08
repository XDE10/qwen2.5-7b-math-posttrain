# 项目入口

适用范围：本项目根目录及其子目录。

- 当前项目：项目一：7B 数学后训练与同题轨迹对照。
- 主线是 Qwen2.5-7B-Instruct；旧 0.5B/1.5B 只作历史。1500 对是同题两臂；4810 条 RL 文件仍为候选。当前 RL smoke 从合并 SFT 基座开始，不覆盖完整 E0～E4。
- 先读 [README.md](README.md)，再按任务读取 [configs/](configs) 和 [data/portable_release/manifest.json](data/portable_release/manifest.json)。无需先遍历全部原始数据、缓存或历史报告。
- 当前执行配置在根目录 configs/；project_blueprints/ 和原始发布目录中的旧配置保留溯源，可能带旧绝对路径。
- 本项目独立于旧 modern_genai 和其他项目目录；旧原仓库项目环境已清理，不能引用其 .venv。
- 数据范围以 approved_training_files / approved_evaluation_files 和逐行批准为准。候选不等于批准，AI 审核不等于人工金标；不要手改批准或哈希以绕过检查。
- 代码能力查看实际源码，状态查看 README 与版本化报告。当前 GPU 尚未验收，2 步 smoke 和历史 CPU 通过不代表正式训练成功。
- 命令从项目根目录执行；离线入口为 python tools/verify_migration.py。依赖与相关测试命令见 README。
- 文档/代码变更保留来源和验证记录；固定模型、数据、划分和评测身份，报告未运行的检查。
