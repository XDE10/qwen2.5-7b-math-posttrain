# 项目一：7B 数学后训练

Qwen2.5-7B-Instruct 的数学后训练项目：SFT 取得实测提升 → GRPO 继续提升，与 Qwen2.5-Math-7B-Instruct 对标，形成可写简历的结果链；另含「同一道题：原生长轨迹 vs 教师精炼轨迹」的质量/长度/成本对照，以及 QFFT 造数链。不能预先保证涨分。

- **当前状态**：[docs/status/current_status.md](docs/status/current_status.md)
- **协作约定与阅读入口**：[AGENTS.md](AGENTS.md)
- **逐版本历史（v1…v93）**：[docs/status/history.md](docs/status/history.md)
- **接手当前工作（本机 Kiro IDE / WSL2）**：[docs/handover/kiro_ide_onboarding.md](docs/handover/kiro_ide_onboarding.md)
- **v93 方案与面试材料**：[docs/blueprints/三周冲刺方案_v93.md](docs/blueprints/三周冲刺方案_v93.md)、[docs/interview/面试追问与参考答案.md](docs/interview/面试追问与参考答案.md)

## 当前主线从哪开始读（v93）

1. [experiments/qfft_v93/README.md](experiments/qfft_v93/README.md)：实验矩阵、本地数据步骤、Colab 一键阶段、断点续跑、评测口径。
2. [configs/pipeline.json](experiments/qfft_v93/configs/pipeline.json)、[configs/train/_defaults.json](experiments/qfft_v93/configs/train/_defaults.json)、[configs/grpo/_defaults.json](experiments/qfft_v93/configs/grpo/_defaults.json)：模型 revision、评测口径、全参与 GRPO 配方。
3. [formats.py](experiments/qfft_v93/formats.py) 与 [grading.py](experiments/qfft_v93/grading.py)：唯一的 prompt/response 模板与最终段判分。
4. 一键入口：[colab/pipeline.sh](experiments/qfft_v93/colab/pipeline.sh)（按阶段）与 [colab/launch.sh](experiments/qfft_v93/colab/launch.sh)（单次 VM），需要 Linux/WSL2 上的 Colab CLI（按 google-colab-cli 0.7.5 核对过）。

v93 数据在 `data/qfft1000_v93/`（`r1/`、`ds/`、可选 `ds_verified/` 各 1000 条，含 `manifest.json` 与 `teacher_report.md`），评测集在 `data/benchmarks_v93/`（固定 revision 与 SHA-256），均由本地脚本生成、git 忽略。

v26（LoRA、8K 窗口、静态 YaRN、并发 4 评测）保留为历史：[configs/project1_aimo_sft_v26.json](configs/project1_aimo_sft_v26.json)、[experiments/aimo_sft_v26/](experiments/aimo_sft_v26/)、[tools/aimo/start_aimo_sft_v28.py](tools/aimo/start_aimo_sft_v28.py)，数据在 `data/aimo_sft_v26/`。

## 目标目录结构（到二级）

```text
项目一_7B数学后训练/
├── README.md  AGENTS.md  .gitignore  pyproject.toml
├── configs/                # v26 工程配置（历史；v93 配置在 experiments/qfft_v93/configs/）
├── experiments/qfft_v93/   # 现行：1000 题双教师 × SFT/QFFT 全参 × LoRA-GRPO × 评测 × Colab 一键调度
├── experiments/aimo_sft_v26/   # 历史：AIMO LoRA SFT 实验包
├── scripts/
│   ├── training/           # train_project1_sft.py / train_project1_rl.py
│   └── evaluation/         # evaluate_approved_math.py + 两个 Math-Verify worker
├── src/
│   ├── data/  evaluation/  losses/  training/   # 16 个现行库模块
├── tests/
│   ├── data/  evaluation/  losses/  training/  qfft/   # 25 个现行测试
├── tools/
│   ├── aimo/  evaluation/  qfft/  review/  verification/   # 23 个现行工具
├── data/                   # 本地数据，git 忽略（仅 README 入库）
│   ├── aimo_sft_v26/  math/  portable_release/  tokenizers/  math_manifest.json
├── artifacts/              # 造数/评测原始证据，git 忽略（仅 README 入库）
├── reports/
│   ├── current/  runs/     # 现行 7 个报告目录，git 忽略（仅 README 入库）
├── docs/
│   ├── status/  blueprints/  handover/  migration/  metadata/  operations/
├── archive/                # 只读历史快照：configs/ data/ experiments/ mapping/ reports/ scripts/ src/ tests/ tools/
└── .pilot_deps/            # 数学验证器纯 Python 依赖，git 忽略
```

pytest 配置在根目录 `pyproject.toml` 的 `[tool.pytest.ini_options]`（`testpaths = ["tests"]`、`--strict-markers`）；原 `pytest.ini` 已并入并删除。

## 目录职责

| 路径 | 职责 | 备注 |
|---|---|---|
| `configs/` | v26 训练/评测配置 | 只有 `project1_aimo_sft_v26.json`；历史 20 个配置在 `archive/configs/` |
| `experiments/qfft_v93/` | 现行主线 | 数据构建与 API 补齐、全参 SFT/QFFT、生成/判分/图表、LoRA-GRPO、Colab 调度；见其 README |
| `experiments/aimo_sft_v26/` | 历史 AIMO SFT 实验代码 | 被 v93 取代，保留可追溯 |
| `scripts/training/`、`scripts/evaluation/` | 6 个命令行入口与评分 worker | 早期 project1 入口，配套 smoke 配置已归档 |
| `src/data/`、`src/evaluation/`、`src/losses/`、`src/training/` | 16 个现行库模块 | 批准检查、答案验证、损失参考、HF SFT/GRPO 入口 |
| `tests/` | 25 个现行测试 | 分组：`data` 5、`evaluation` 4、`losses` 1、`training` 3、`qfft` 12 |
| `tools/aimo/`、`tools/evaluation/`、`tools/qfft/`、`tools/review/`、`tools/verification/` | 23 个现行工具 | AIMO 启动、MATH-500 评测、QFFT 造数与只读监控、审查包、迁移校验 |
| `data/` | 本地数据（原始/发布/tokenizer） | 大体积，不进 git |
| `artifacts/` | 造数与评测的请求/响应/候选/复核证据（约 1.7 GB） | 不进 git |
| `reports/` | 现行报告：`current/`（审查材料）、`runs/`（造数/复现运行） | 不进 git（仅索引 README） |
| `docs/` | 状态、方案、交接、迁移、元数据、运维手册 | 见 [docs/README.md](docs/README.md) |
| `archive/` | 被取代的历史脚本/测试/配置/报告（只读，不保证可运行） | 见 [archive/README.md](archive/README.md) |
| `.pilot_deps/` | `math_verify 0.9.0` 等纯 Python 运行时依赖 | git 忽略 |

## 数据与仓库边界

`.gitignore` 约定（2026-11-08）：**大体积数据、证据与报告只在本地，不入库**。

- 不入 git：`data/`（`data/README.md` 例外）、`artifacts/`（`artifacts/README.md` 例外）、`reports/`（`reports/README.md` 例外）、`archive/data/`、`archive/reports/`、`.pilot_deps/`、`*.zip`、`.pytest_tmp/`。
- 入库的是：代码（`src/ experiments/ scripts/ tools/ tests/`）、配置、文档、`archive/` 中的代码/测试/配置快照、各目录 README。
- 本地数据规模：`data/aimo_sft_v26` 约 31 MB、`data/math` 约 27 MB、`data/portable_release` 约 22 MB、`data/tokenizers` 约 51 MB；`artifacts/` 约 1.7 GB、`reports/` 约 168 MB、`archive/data/` 与 `archive/reports/` 为历史数据/报告。
- 数据准入以对应 manifest 的 `approved_training_files` / `approved_evaluation_files` 和逐行授权为准；候选、隔离、存疑数据不能靠手改批准字段或哈希进入训练。
- 训练/评测/最终测试用途不同：最终测试不能变成生成种子或选模依据；dev 用于调参与选 checkpoint。

## 测试

在仓库根目录用 `genai-test` 解释器运行：

```bash
D:/anaconda3/envs/genai-test/python.exe -m pytest -q
```

- 配置在根目录 `pyproject.toml` 的 `[tool.pytest.ini_options]`（`testpaths = ["tests"]`，`--strict-markers`）；原 `pytest.ini` 已并入并删除。
- 重构前基线（含已归档的旧测试）：`12 failed, 583 passed, 2 skipped`，原始输出见 [docs/migration/pytest_baseline_before_restructure.txt](docs/migration/pytest_baseline_before_restructure.txt)；12 个失败全部属于已归档范围。
- 重构后只保留 25 个现行测试文件（172 个用例），2026-11-08 重构收尾实测：**172 passed / 0 failed**（约 23s），输出见 [docs/migration/pytest_after_restructure.txt](docs/migration/pytest_after_restructure.txt)。

## archive 与历史

`archive/` 是**只读历史快照**：v1…v92 被取代的脚本/测试/配置/实验/报告不删除，统一移入此处；内部相对路径可能已变，**不保证可直接运行**。

- 找回原路径：`archive/mapping/restructure_v2_mapping.json`（570 条 `from` / `to` / `tracked`）。
- 归档结构、检索示例与「不删除任何文件」原则：[archive/README.md](archive/README.md)。
- 重构动机、决策规则、统计与 pytest 基线：[docs/migration/重构记录_20261108_v2.md](docs/migration/重构记录_20261108_v2.md)。

## 文档维护

- 新状态写 `docs/status/current_status.md`；新历史条目写 `docs/status/history.md`；`AGENTS.md` 只保留长期约定。
- 移动/重命名文件时同步 `docs/metadata/migration_manifest.json`，并跑 `python tools/verification/verify_migration.py`。
- 文档中引用的路径必须真实存在；引用归档文件时写明 `archive/` 前缀。
