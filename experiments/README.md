# experiments/ —— 实验包

现行实验包是 **`qfft_v93/`**（1000 题双教师 × 全参 SFT/QFFT × LoRA-GRPO × 评测 × Colab 一键调度），说明见 [qfft_v93/README.md](qfft_v93/README.md)。`aimo_sft_v26/`（AIMO 直接 LoRA SFT）被 v93 取代，保留为历史；`aimo_sft_v25/` 与 `projection_pilot_v21/v23/v24/` 已移入 `archive/experiments/`。

## aimo_sft_v26/（历史）

| 文件 | 职责 |
|---|---|
| `train.py` | 单卡 A100 BF16 LoRA SFT；assistant-only loss，保留完整 R1 思考；按 epoch 保存 |
| `evaluate.py` | 可恢复的 pass@1 估计：32768 生成 token、AIME 每题 64 次、MATH-500 单次辅助 |
| `export_checkpoint.py` | 在 CPU 上合并某个 epoch 的 adapter；子进程在 vLLM 启动前退出 |
| `run.py` | 从 Drive 恢复模型与数据并启动 SFT 或可恢复评测（Colab 执行入口） |
| `common.py` | 固定身份与「不破坏 loss」的 R1 格式化等公共逻辑 |
| `telemetry.py` | 必需的云端 SwanLab 遥测；凭证不进入配置或指标 |
| `install_missing.py` | 保留已安装核心版本，只补缺失的训练/评测包 |
| `start_colab.sh` | Colab 侧启动脚本 |

## 与其它目录的关系

- 配置：`configs/project1_aimo_sft_v26.json`（模型 revision、数据路径、LoRA/优化器、评测口径）。
- 数据：`data/aimo_sft_v26/`（`train.jsonl` 31,991,127 B，2275 条 / 2263 题）。
- 云端启动入口：`tools/aimo/start_aimo_sft_v28.py`（自动读 SwanLab Key → 还原包 → 应用 v27 补丁 → 启动本实验包）；`package_aimo_sft_v26.py`、`prepare_aimo_sft_v26.py` 负责打包与数据准备。
- 测试：`tests/training/test_aimo_sft_v26.py`、`test_colab_compat_v27.py`、`test_swanlab_autoauth_v28.py`。

## 边界

- 训练窗口 8192 与评测生成 32768 分开；`max_model_len 40960` 必须容纳 prompt + 生成。
- 只有导入闭包内、被配置或文档明确引用的代码才是现行主线；旧实验包（v25、projection pilot 等）在 `archive/experiments/`，只作追溯，不保证可直接运行。
- 不在未获授权时启动 GPU 任务；CPU 单测通过不等于 GPU 验收。
