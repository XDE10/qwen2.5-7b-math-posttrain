# 项目一：7B 数学后训练与同题轨迹对照

研究 Qwen2.5-7B-Instruct 在数学任务上的 SFT、GRPO 与 Dr.GRPO，以及"同一道题：原生长轨迹 vs 教师精炼轨迹"的质量、长度与成本对照。当前主模型固定 `Qwen/Qwen2.5-7B-Instruct`（revision `a09a35458c702b33eeacc393d103063234e8bc28`）。

**最新状态、授权与未完成事项以 [AGENTS.md](AGENTS.md) 为准**（截至 2026-11-03，v91/v92 造数与外部审查）。本文件其余章节是 2026-10-25～10-28 的工程与迁移基线快照，历史状态不在此重复；两者冲突时以 AGENTS.md 与 `reports/` 中最新报告为准。


## 目录结构

| 路径 | 内容 | 说明 |
|---|---|---|
| `configs/` | 工程配置 | 相对路径，根目录执行 |
| `src/` | 训练、数据、评分、损失实现 | 主线代码 |
| `scripts/` | 命令行入口与流水线 | 历史与现行混放，按报告中的命令定位 |
| `tools/` | 离线校验与监控 | `verify_migration.py`、造数监视器 |
| `tests/` | pytest 用例 | |
| `data/` | 原始筛选源、发布数据、tokenizer | 大体积，不入 git |
| `artifacts/` | 造数与评测原始证据 | 大体积，不入 git |
| `reports/` | 逐次实验报告 | 过程目录，不入 git；索引见 [reports/README.md](reports/README.md) |
| `docs/blueprints/` | 研究方案、共用规范、上卡说明、批准配置留档 | |
| `docs/migration/` | 迁移验收、清理与重构记录 | |
| `docs/metadata/` | 迁移清单、CPU 环境快照、Colab 校验清单 | |
| `docs/operations/` | CPU 夜间运行手册 | |
| `.pilot_deps/` | 数学验证器纯 Python 依赖 | 含许可证，git 忽略 |

---

## 项目目标

研究 7B 指令模型在数学任务上的 SFT、GRPO 与 Dr.GRPO，以及同一道题的原生长轨迹和教师精炼轨迹是否带来不同的质量、长度与训练成本。当前主模型是 **Qwen/Qwen2.5-7B-Instruct**。这是面向实习展示的受控后训练项目；旧 0.5B/1.5B TinyPostTrain 试点已于 2026-11-08 删除（见 [清理记录](docs/migration/清理记录_20261108_独立化.md)）。

已产出数据、审计、训练入口、公开模型评测与 QFFT 造数证据；正式双组训练与最终收益结论仍未完成，未验证事项见 [AGENTS.md](AGENTS.md)。

## 给 Codex 的最小阅读顺序

1. [AGENTS.md](AGENTS.md)：本目录的项目身份和阅读入口。
2. 本 README：当前状态、代码地图和实际命令。
3. [当前运行数据清单](data/portable_release/manifest.json) 与 [当前配置目录](configs)：核对实际文件、批准范围、模型和输出身份。
4. 按任务阅读下方相关源文件；再查 [完整研究方案](docs/blueprints/项目一_7B数学后训练_详细实施方案.md) 与 [Strict v8 实测报告](reports/strict_v8_20261024/执行报告.md)。

`data/portable_release/manifest.json` 的 `approved_training_files` / `approved_evaluation_files` 是本包当前数据准入范围。清单里的跨项目 `summary` 是原联合报告的背景，不会自动扩大授权。代码功能以实际源码为准；旧方案中的“待实现”“暂停写脚本”或旧数据数量可能已被后续实现和发布取代。具体冲突先看文件和版本，不能按段落里的“最新”二字推断。

## 当前事实与未完成事项

- 当前发布为 **1500 道同题配对**：`project1_raw_paired.jsonl` 和 `project1_compact_paired.jsonl` 各 1500 行。两臂是同一批题目，不能把它们当成 3000 道独立题。
- 另有 `project1_rl_priority_not_rollout_selected.jsonl`，共 **4810 道 RL 优先候选**。它不在 manifest 的 `approved_training_files` 中，也不是已用 7B rollout 筛选出的正式 RL 题池。
- 当前 smoke 的 SFT 两臂及 RL 工程入口已存在；所有 `configs/` 配置都是 `max_steps=2`，用于工程接线，不能当作正式训练预算。
- 原生 SFT smoke 的 `max_length=4096`，精炼 SFT smoke 为 3800；RL 生成上限 2048、每题 4 条回答、`beta=0.04`。这些是当前配置值，不能与旧方案建议值混用，也不足以自动建立公平的正式对照。
- 原生/精炼的数据差异同时涉及教师、提示词与轨迹长度；没有模型实测前，不能归因为“消除了过度思考”。
- 最近的包内相关 CPU 测试记录为 **17 passed**；完整迁移文件 SHA 校验通过。GPU/A100 加载、前后向、rollout 和恢复仍未验收。

当前只有工程 smoke 配置，没有本项目 GPU 训练产物、完整正式实验结果或已实测的 Linux/CUDA 环境锁文件。原仓库借用过的 Windows 虚拟环境已经清理；本地 CPU 复现使用独立环境，当前 Colab 则优先检查并复用预装环境。

## 模型与数据来源

- 初始模型：`Qwen/Qwen2.5-7B-Instruct`，固定 revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- tokenizer：[本地 7B tokenizer](data/tokenizers/Qwen2.5-7B-Instruct)。不从另一个模型或旧 TinyPostTrain 借用模板。
- 原生/精炼轨迹来自版本化来源筛选与盲生成后验证流程；批准方法是逐项 AI 审核加确定性检查，**不是人工金标认证**。
- 通用数学文件在 [data/math](data/math)，包括 train、dev、MATH-500、GSM8K-test。训练、调参和最终测试用途不同；最终测试不能变成生成种子或选模依据。

## 当前数据与使用边界

- [原生轨迹](data/portable_release/project1_raw_paired.jsonl)：SFT 原生臂，批准训练。
- [精炼轨迹](data/portable_release/project1_compact_paired.jsonl)：SFT 精炼臂，批准训练。
- [RL 优先候选](data/portable_release/project1_rl_priority_not_rollout_selected.jsonl)：候选，仅供后续正式筛选，未被当前 manifest 批准训练。
- [dev 保护 ID](data/portable_release/reserved_dev_ids_never_train.json)：不得混入训练或变式题生成种子。

数学 SFT 的模型输入只有题目，目标是相应 `solution`；标准答案和审核字段不进入问题提示。当前 RL smoke 复用已批准同题子集做工程检查，正式 RL 独立题池与实验预算尚需另行冻结。

所有已批准训练行仍需要通过文件 SHA、逐行 `training_approved`、唯一 ID 和模型输入白名单检查。候选、隔离、存疑数据不可通过手改批准字段或仅更新哈希进入训练。新数据版本应保留原发布与来源证据，验证/测试内容不能进入合成生成器。

## 目录与代码地图

- [configs](configs)：本项目使用相对路径的最新工程配置，执行时从本目录根部运行。
- [data/portable_release](data/portable_release)：当前实际运行副本；原始发布和历史版本在 `data/` 其他目录中留档。
- [src](src)、[scripts](scripts)、[tests](tests)：训练、数据、评分与测试实现。迁移保留了一些共用模块和其他项目脚本作为依赖/证据；存在于本包不代表它属于本项目主线或能不加选择地运行。
- [docs/blueprints](docs/blueprints)：详细研究方案、共用规范和历史配置。里面旧路径配置不能替代根目录 `configs/`。
- [reports](reports)、[docs/migration](docs/migration)：逐项审核、失败/隔离证据、版本化测试与迁移验收。
- [tools/verify_migration.py](tools/verify_migration.py)：无需模型或网络的离线完整性校验。
- 原 `math`、`agentic_rl`、`video_mapping`、`tiny_posttrain` 参考学习材料与旧试点已于 2026-11-08 按用户要求删除，见 [清理记录](docs/migration/清理记录_20261108_独立化.md)。

本项目主要实现：

- [scripts/train_project1_sft.py](scripts/train_project1_sft.py)：SFT 命令入口。
- [scripts/train_project1_rl.py](scripts/train_project1_rl.py)：GRPO/Dr.GRPO 命令入口。
- [src/training/approved_hf.py](src/training/approved_hf.py)：批准检查、assistant-only SFT、LoRA、TRL GRPO、checkpoint 身份和合并 SFT 基座。
- [src/data/approved_training_inputs.py](src/data/approved_training_inputs.py)：文件授权、SHA、逐行批准和模型字段白名单。
- [src/losses/reference.py](src/losses/reference.py)：SFT、GRPO/Dr.GRPO 与 KL 的小张量参考算子。
- [scripts/evaluate_approved_math.py](scripts/evaluate_approved_math.py)：固定 greedy 数学评测及逐题输出；调用保守答案验证器。
- [scripts/math_verify_v3_strict_worker.py](scripts/math_verify_v3_strict_worker.py)：数学答案保守类型门槛；不支持项须单列。数据构造阶段的较新类型核验与训练评分器不是同一套覆盖范围。

## 环境与首次检查

**当前 Colab 用户先执行 [v5 指南](reports/colab_preinstalled_20261025_v5/Colab预装环境指南_v5.md)，跳过本节创建环境和安装固定快照的命令。** 轻量上传包不含完整历史迁移树，不要在 Colab 上传包里运行下面的全项目迁移校验；它应在本地完整项目执行。上传包使用自己的 `colab_upload_checksums.json` 核验，文档更新包另有增量清单。

所有下方相对路径命令均先进入本项目根目录；不使用旧 `modern_genai` 路径或旧 venv。Python 3.12 是之前 CPU 模块验证所用版本。首先只需标准库即可检查迁移完整性：

```bash
python --version
python tools/verify_migration.py
python tools/verify_migration.py --full
```

第一条校验模式检查清单文件存在性/大小、当前 release SHA、批准范围、配置依赖和图片 SHA；`--full` 额外检查所有记录文件的 SHA。脚本可在别处通过其绝对路径调用，它会自行定位包根。校验通过不证明 CUDA 训练或新环境依赖已安装。

以下是本地 CPU 历史复现步骤（不是当前 Colab 安装步骤；先使用你安装的 Python 3.12 创建虚拟环境）：

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活：

```bash
source .venv/bin/activate
```

激活后恢复模块检查环境并运行相关测试：

```bash
python -m pip install -r docs/metadata/requirements-cpu-snapshot.txt
python -m pytest tests/test_approved_training_inputs.py tests/test_training_entries_cpu.py tests/test_losses_and_masks.py -q
```

[requirements-cpu-snapshot.txt](docs/metadata/requirements-cpu-snapshot.txt) 是旧 CPU 环境的主要包版本记录，包含 PyTorch、Transformers、TRL、PEFT 等，**不是完整依赖锁或已实测的 CUDA 安装方案**。目标 GPU 环境应单独固定与驱动匹配的 PyTorch/CUDA 和框架版本并做 import/API 验收，不要据 CPU 包安装成功宣称 GPU 可运行。[.pilot_deps](.pilot_deps) 保留数学验证器及相关纯 Python 依赖/许可证；其中 SymPy 等仍依赖实际环境安装。

最近已保存的相关 CPU 检查为 17 passed，证据见 [迁移验收记录](docs/migration/迁移验收.md) 和 [机器可读结果](docs/migration/迁移验收.json)。本次 README 更新只运行文档与离线完整性检查，没有重跑模型测试或 GPU 训练。

## 当前 GPU 入口与输出

以下为既有工程入口，当前预装环境尚未验收。先按 v5 指南完成依赖、tokenizer 和 CPU/API 检查；v4 采样器有显式旧版本拦截，不可直接在新镜像运行，更不能删除拦截后沿用旧实验身份。

先分别完成两个 SFT smoke，再运行依赖对应合并基座的 RL。下面是当前文件中实际存在的入口，不代表已经在 GPU 跑通。

```bash
python -m scripts.train_project1_sft --config configs/project1_raw_sft_smoke.json
python -m scripts.train_project1_sft --config configs/project1_compact_sft_smoke.json
python -m scripts.train_project1_rl --config configs/project1_raw_grpo_smoke.json
python -m scripts.train_project1_rl --config configs/project1_raw_dr_grpo_smoke.json
python -m scripts.train_project1_rl --config configs/project1_compact_grpo_smoke.json
python -m scripts.train_project1_rl --config configs/project1_compact_dr_grpo_smoke.json
```

SFT 完成后分别导出 `runs/strict_v8_project1_raw_sft_smoke/merged/` 和 `runs/strict_v8_project1_compact_sft_smoke/merged/`。RL 需要其中的 `merged_sft_identity.json`、`weight_manifest.json` 及实际权重；这些产物尚未生成，未附在迁移包里。

RL 在合并 SFT 基座上新建 LoRA，以便固定 SFT reference；不能把 SFT adapter 直接当作 RL adapter，再通过禁用它误退回初始模型。GRPO 使用组内奖励标准差缩放，Dr.GRPO 关闭该缩放；实际 TRL 行为仍要用目标 GPU 日志验证。

正式模型命令需要目标 CUDA 环境和配置中的固定模型 revision 权重；本次文档更新不会自动执行这些训练/下载命令。当前入口检查单张可见 GPU，不能直接当成多卡配置。重新开始实验使用新输出目录；恢复必须核对完整 checkpoint 身份，不能只加载 adapter 就声称恢复了优化过程。

## 完整实验与评测计划

完整核心方案是 E0 原始模型、E1 SFT、E2 原始模型直接 GRPO、E3 同一 SFT 起点的 GRPO、E4 同一 SFT 起点的 Dr.GRPO。当前配置只覆盖 SFT 与 SFT 后 RL 的工程 smoke，**不等于 E0～E4 已全部配置和完成**，直接 RL 起点及正式配置仍需补齐。

同题原生/精炼轨迹是另一条配对消融轴。正式运行需预登记相同题集、模型起点、训练/生成 token 预算、评测提示、长度和评分版本；不要把 smoke 的不同长度上限直接解释为严格公平的最终实验。

最终数学评测计划为 MATH-500、GSM8K-test；dev 用于调参与预算选择。报告准确率、格式/解析失败、截断、回答长度、实际训练与 rollout tokens、峰值显存和时间。验证器对关系式/集合等不支持或不确定的情况单列，不能统称模型推理错误。

## 建议继续工作的顺序

1. 用本包配置和实际 tokenizer 核对输入、assistant labels、EOS 与最长样本；确认数学评分器的覆盖与不支持类别。
2. 建立目标 Linux/CUDA 环境，固定 GPU/依赖版本；跑 E0 dev 基线与包含最长样本的 2 步 SFT 工程检查，记录 loss、梯度与峰值显存。
3. 主动中断并恢复一次，核对 optimizer、scheduler、RNG、step 与模型身份；再检查新 RL LoRA 对应的固定 SFT reference。
4. 跑有效组奖励及截断检查，通过后再冻结正式 RL 题池、E0～E4 配置与原生/精炼比较方案。20 更新 pilot、100/200 等检查点是计划，不能写成已达到收敛。
5. 完成受控训练、最终评测、改善/退化案例与负结果分析，形成可复现报告。

每个阶段保存配置、数据/模型身份、原始预测、评分版本、恢复记录、显存和成本。将“已实现代码”“CPU 通过”“真实模型跑通”“最终能力提升”分别陈述；未运行的检查保留未验证状态。

## 历史材料、迁移与后续维护

旧 TinyPostTrain 完整源码压缩包（原 `历史归档/`）与 `tiny_posttrain/` 目录已于 2026-11-08 按用户要求删除（旧 1.5B 试点已被 7B 主线取代），见 [清理记录](docs/migration/清理记录_20261108_独立化.md)；其中旧模型结果本就不能当作 7B 证据。

共用 API 审核缓存的 15119 个小文件完整保存在 [shared_api_cache.zip](reports/data_release_preparation_20261024/shared_api_cache.zip)。现有训练/离线校验无需解压；需要查询原始请求或恢复历史数据审核时再解压：

```bash
python -m zipfile -e reports/data_release_preparation_20261024/shared_api_cache.zip reports/data_release_preparation_20261024/shared_api_cache
```

历史成本账本是审计事实，不是新任务的调用额度；恢复 API 数据流水线前应依据当前任务的实际范围、密钥与预算。不要自动重新生成或扩量已封存数据。

[migration_manifest.json](docs/metadata/migration_manifest.json) 跟踪迁移负载文件的大小和哈希，数据 release 的 manifest 另行保护训练批准。本次只同步 README/AGENTS 对应的文档清单项，并在 `docs/migration/README交接补充前_20261024.md`、`docs/migration/migration_manifest_交接补充前_20261024.json` 保留旧版本。以后合法修改文档/代码应说明变更和相关校验；不能用批量重算训练数据 SHA 的方式绕过数据准入。

## 新对话可直接使用的开场说明

> 请先读取本项目 AGENTS.md、README.md、configs/ 和 data/portable_release/manifest.json。结合实际源码说明项目目标、已完成工作、未验证事项和建议的下一步。继续本项目主线，保留既有数据与审核记录；不要把历史方案或 2 步 smoke 写成已完成的正式训练结果。
