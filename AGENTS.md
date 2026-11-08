# 项目一：协作约定与入口（AGENTS.md）

适用范围：本仓库根目录及子目录。2026-11-08 目录重构 v2 后重写；本文件只保留**长期有效的约定、当前状态和阅读入口**，逐版本历史已迁移到 [docs/status/history.md](docs/status/history.md)。

## 项目是什么

Qwen2.5-7B-Instruct 的数学后训练项目：SFT 取得实测提升 → GRPO 继续提升，与 Qwen2.5-Math-7B-Instruct 对标，形成可写简历的结果链；另含「同一道题：原生长轨迹 vs 教师精炼轨迹」的质量/长度/成本对照，以及 QFFT 造数链。不能预先保证涨分。

固定底座 `Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`。不要切回 R1-Distill，不自行更换数据集。

## 阅读入口（按顺序）

1. 本文件 → [docs/status/current_status.md](docs/status/current_status.md)：当前做到哪、哪些未完成。
2. [README.md](README.md)：目录职责、数据边界、命令。
3. 当前任务读 [训练配置](configs/project1_aimo_sft_v26.json)、[数据 manifest](data/aimo_sft_v26/manifest.json)、[实验包](experiments/aimo_sft_v26/) 与 [最新启动入口](tools/aimo/start_aimo_sft_v28.py)。
4. 需要某个版本（v1…v92）的细节时再查 [docs/status/history.md](docs/status/history.md)；查旧路径用 `archive/mapping/restructure_v2_mapping.json`。
5. 不需要先遍历数据、缓存和全部历史报告。

## 当前主线（摘要）

- 数据：原始 AIMO2 `sft-4k` 的 8192 完整窗口子集，训练 **2275 条 / 2263 题**（扫描 4033 条，不压缩、不截断、不再限 1000 条；有/无 think 标签的原始推导都保留）。
- 训练配方 **v26**：LoRA r32/alpha64/dropout0.05，LR 2e-05，3 epochs，batch 2×累积 8（有效 16），`max_length 8192`，逐 epoch 保存，在线 SwanLab。
- 启动入口 **v28**：读取私密 SwanLab Key → 自动准备依赖 → 启动 v26 配方；v27 为前置兼容补丁（torchao / warmup_ratio / 在线认证）。
- 环境：本地根目录 `D:/Users/Mr.Z/Desktop/项目一_7B数学后训练`，CPU 解释器 `D:/anaconda3/envs/genai-test/python.exe`；训练/评测在 Colab 单卡 A100 80GB，Drive 持久目录 `projects/project1`、`models`、`runs/project1`。
- 评测口径：训练窗口 8192 与评测生成 32768 分开；`max_new_tokens 32768`、`max_model_len 40960`、YaRN factor 4、temperature 0.6、top_p 0.95；AIME 每题 64 次，MATH-500 单次辅助。
- QFFT 造数/审查链最新到 v91（补跑不完整题）/v92（外部审查包），时间 2026-11-03；证据在 `artifacts/qfft_*`，报告在 [reports/runs/](reports/runs/)。
- **未完成**：正式双组（SFT / QFFT）训练、最终收益结论、完整恢复/保存/释放闭环。详见 [docs/status/current_status.md](docs/status/current_status.md)。

## 仓库约定（必须遵守）

- **不删除历史文件。** 被取代的脚本/测试/配置/报告统一移入 `archive/`，不删除；`archive/` 是**只读历史快照**，不保证可直接运行（见 [archive/README.md](archive/README.md)）。
- 现行主线 = 从活跃入口出发的**导入闭包 + 文档明确引用**的代码；不在闭包里的不要擅自提升为主线，也不要为了「整洁」删除。
- 改代码前先看 `git status` 和实际文件；文档中的路径必须真实存在（归档路径要写明 `archive/` 前缀）。
- 数据准入：只看对应 manifest 的 `approved_training_files` / `approved_evaluation_files` 与逐行授权；候选 ≠ 批准，AI 审核/答案等价 ≠ 人工推导金标。不能手改批准字段或哈希绕过准入。
- 不重复全量去污/审计/扩量；验证聚焦当前报错和身份关键项，避免防御性检查阻碍执行。
- 不擅自启动 GPU 任务、不擅自发起付费/订阅 API 请求、不自动扩量或入训；这些需要用户明确授权，并把「已授权范围」写进报告。
- 凭证（SwanLab / DeepSeek Key）只放私密文件，不打印、不回显、不入库、不写进交接文档或公开运行包。
- 大体积数据/证据/报告只在本地：`data/`、`artifacts/`、`reports/`、`archive/data/`、`archive/reports/`、`.pilot_deps/`、`*.zip`、`.pytest_tmp/` 不入 git（各目录保留 README 例外）。详见 [.gitignore](.gitignore)。
- 文档与代码保留来源和验证记录；**CPU 通过不等于 GPU 结果**，「已实现 / CPU 通过 / 真实 GPU 跑通 / 能力提升」要分开陈述。
- 迁移/清理/重构的每次移动都留映射与记录（`docs/migration/`），移动后同步 `docs/metadata/migration_manifest.json` 并跑 `python tools/verification/verify_migration.py`。

## 目录地图

| 目录 | 职责 |
|---|---|
| `configs/` | 现行工程配置，仅 `project1_aimo_sft_v26.json`；历史配置在 `archive/configs/` |
| `experiments/aimo_sft_v26/` | 现行 AIMO SFT 实验包（train/evaluate/export/telemetry 等） |
| `scripts/{training,evaluation}/` | 6 个现行命令行/worker 入口；其余历史脚本在 `archive/scripts/` |
| `src/{data,evaluation,losses,training}/` | 16 个现行库模块（批准检查、评分器、损失参考、HF 训练入口） |
| `tests/` | 25 个现行测试文件（172 个用例），按 data / evaluation / losses / training / qfft 分组 |
| `tools/{aimo,evaluation,qfft,review,verification}/` | 现行工具：AIMO 启动、评测、QFFT 造数与监控、审查包、迁移校验 |
| `data/` | 本地数据（`aimo_sft_v26/`、`math/`、`portable_release/`、`tokenizers/`），git 忽略 |
| `artifacts/` | 造数与评测原始证据，git 忽略 |
| `reports/{current,runs}/` | 现行报告（7 个目录），git 忽略（仅 README）；历史报告在 `archive/reports/` |
| `docs/` | `status/`、`blueprints/`、`handover/`、`migration/`、`metadata/`、`operations/` |
| `archive/` | 只读历史快照：`configs/ data/ experiments/ mapping/ reports/ scripts/ src/ tests/ tools/` |
| `.pilot_deps/` | 数学验证器（math_verify 0.9.0 等）纯 Python 运行时依赖，git 忽略 |

## 路径、启动与恢复

- 本地命令一律在仓库根目录执行；`pytest` 用 `D:/anaconda3/envs/genai-test/python.exe -m pytest`。
- Colab 采用 VS Code 远程终端；notebook 只用于连接和挂载 Drive，不把 Python 挂载代码贴进 Bash。
- Drive 持久目录：`projects/project1`、`models`、`runs/project1`；工作副本 `/content/workspaces/aimo-sft-v26`；默认结果 `/content/drive/MyDrive/runs/project1/aimo_sft_v26`。
- Drive 挂载后且没有同任务运行时，最新命令：

```bash
python3 /content/drive/MyDrive/projects/project1/start_aimo_sft_v28.py
```

- `--batch-size 4`、`--output-root` 可继续传入；恢复沿用原值。入口自动读凭证、还原旧包、应用补丁、处理依赖并启动，不用手工拼分片或重下已有模型。
- **有任务在跑时先看最新状态，不重启窗口、不杀进程、不重复启动 GPU 任务。**

## 已知兼容项必须落实到入口

- torchao 0.10.0 与当前 PEFT 不兼容：v27 自动把已安装且低于 0.16.0 的版本定向升级到 0.18.0，用 `--no-deps`，不连带升级 torch。
- Transformers 5.20.2 不接受 `warmup_ratio`：v27 改用整数 `warmup_steps`（本轮 22 步），在模型加载前验证 TrainingArguments；长度分组同时适配旧 `group_by_length` 与新 `train_sampling_strategy`。
- 历史 `pyroaring<1.0.4` 曾解决 SwanLab 安装问题；按具体错误处理，不无条件重装整套依赖。
- SwanLab 必须在线、详细记录、自动认证：Key 存 `/content/drive/MyDrive/.project1_credentials/swanlab.json`（字段 `SWANLAB_API_KEY`），v28 读取后注入子进程环境，不要求用户每次输入，不静默转 offline。
- 只补缺失依赖和已证实的兼容修复，保留预装 Python/PyTorch/Transformers。

## 数据与评测口径

- 训练/评测范围以对应 manifest 的 approved 列表及逐行授权为准；旧 1500 raw / 1500 compact 是同一批题的两臂，不是 3000 道题。
- 根目录曾有的 4810 条 RL 候选文件是旧候选；E0 每题 4 次 rollout 后获批的 920 条在 Drive `runs/project1/rl_approved_v8_n4`。不把该通过率当新 SFT 模型的通过率。
- 637 题是 E0 四次全错的条件探针，不是独立最终测试。
- 评分用 `src/training/math_reward_v2.py` 与冻结 strict-v3 路径；答案等价不是逐步证明正确，解析失败/不支持类别单列。
- 最后一轮 merged 不自动称最佳：先比较 dev，再正式测试；旧 2048 预算成绩不能直接充当新协议基线。
- 生成上限 32768、上下文 40960、YaRN factor 4 是现行评测口径；不要把推理并发 4 混同训练 batch 4。

## 历史与工作方式

- 逐版本（v1…v92）的日期、实测数字、授权范围和结论都在 [docs/status/history.md](docs/status/history.md)；那是**历史记录**，冲突时以本文件 + `docs/status/current_status.md` + 最新报告为准。
- 工作方式：优先执行，给完整可复制的重连/挂载/恢复命令，明确结束标记与何时可停 GPU；不重复索要已授权的许可；不把历史方案或 smoke 写成已完成的正式训练结果。
- 维护本文件时保持简短：只放长期约定与指向，不追加逐版本流水账（流水账进 `docs/status/history.md`）。
