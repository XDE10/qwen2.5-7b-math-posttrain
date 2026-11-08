# scripts/ —— 命令行入口

**6 个现行入口**（早期 project1 主线，调用 `src/training/approved_hf.py` 与评分 worker）；其余 134 个历史脚本已移入 `archive/scripts/`。

## training/

| 文件 | 职责 |
|---|---|
| `train_project1_sft.py` | 两行封装：`from src.training.approved_hf import cli` → `cli('sft', False)`，manifest 门禁的 SFT 入口 |
| `train_project1_rl.py` | 同上，`cli('rl', False)`，GRPO 入口 |

## evaluation/

| 文件 | 职责 |
|---|---|
| `evaluate_approved_math.py` | 固定 greedy 数学评测；每次运行把封版 JSONL 与模型身份写进结果 |
| `math_verify_pilot_worker.py` | 隔离的有界 Math-Verify worker；stdout 只输出 JSON，绝不输出 API 凭证 |
| `math_verify_v3_strict_worker.py` | 保守的类型化准入补充，与冻结 pilot 验证器分离；关系型答案单独挑出做任务感知复核 |
| `verifier_stream_worker_v4.py` | 持久子进程评分 transport（`python -m scripts.evaluation.verifier_stream_worker_v4`），由 `src/training/frozen_verifier_v4.py` 启动；stdout 只输出 JSON |

## 边界

- 这 5 个入口属于**早期 project1 主线**，与现行 AIMO 配置 `configs/project1_aimo_sft_v26.json` **不直接配对**：其配套 smoke 配置与旧数据已归档。现行 AIMO SFT 走 `experiments/aimo_sft_v26/` + `tools/aimo/start_aimo_sft_v28.py`。
- 参数与数据集以对应 manifest 的批准列表为准；不要为了让旧脚本跑通而改回旧路径或旧键名。
- 历史脚本在 `archive/scripts/`（134 个：各代 AIMO/project1/2/3 训练、评测、造数、审查入口），只作追溯，**不保证可直接运行**；原路径可用 `archive/mapping/restructure_v2_mapping.json` 找回。
- 任何 GPU/付费 API 调用都需当次明确授权；脚本默认不自动启动远端任务。
