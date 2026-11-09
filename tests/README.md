# tests/ —— 现行测试

当前保留 **25 个测试文件**（`conftest.py` 之外，共 172 个用例），分五组，覆盖从入口出发的导入闭包内的库模块与工具行为。运行方式：

```bash
D:/anaconda3/envs/genai-test/python.exe -m pytest -q
```

配置来自仓库根 `pyproject.toml` 的 `[tool.pytest.ini_options]`（`testpaths = ["tests"]`、`--strict-markers`）；原 `pytest.ini` 已并入并删除。

## 目录与文件

| 分组 | 文件 | 覆盖内容 |
|---|---|---|
| 根 | `conftest.py` | 公共 fixture（路径、临时目录、环境隔离） |
| `data/`（5） | `test_approved_training_inputs.py` | `src/data/approved_training_inputs.py`：批准文件校验、哈希/行号准入 |
| | `test_geometry_model_serialization.py` | 几何模型序列化 |
| | `test_perception_v4.py` | `src/data/perception_v4.py` |
| | `test_project3_perception_prototype.py` | project3 感知原型 |
| | `test_project3_perception_v2.py` | project3 感知 v2 |
| `evaluation/`（4） | `test_math_verifier.py` | `src/evaluation/math_verifier.py`：答案抽取与等价判定 |
| | `test_math_verify_v3_type_gate.py` | 冻结 strict-v3 类型门禁 |
| | `test_math500_answer_v41.py` | `tools/evaluation/evaluate_math500_final_answer_v41.py` 的收尾/预算逻辑 |
| | `test_native_ds_v29.py` | `tools/evaluation/evaluate_sft_ds_v29.py` 的原生分与 DS 复核分路径 |
| `losses/`（1） | `test_losses_and_masks.py` | `src/losses/reference.py`：assistant-only loss 与 mask |
| `training/`（3） | `test_aimo_sft_v26.py` | v26 配方与数据/长度分组 |
| | `test_colab_compat_v27.py` | v27 兼容补丁（torchao / warmup_steps / 采样策略） |
| | `test_swanlab_autoauth_v28.py` | v28 自动读取 SwanLab Key 并注入环境 |
| `qfft/`（12） | `test_qfft_ds_dataset_v63.py`、`test_qfft_ds_dataset_v66.py` | 造数入口、题池/预算、默认不请求 API |
| | `test_qfft_resume_v73.py`、`test_qfft_server_errors_v74.py`、`test_qfft_transport_v77.py`、`test_qfft_process_v78.py`、`test_qfft_stream_v79.py`、`test_qfft_circuit_review_v80.py` | 断点恢复、5xx 策略、传输分类、进程隔离、SSE 流式、熔断复核 |
| | `test_qfft_length_retry_v86.py`、`test_qfft_retry_resume_v87.py`、`test_qfft_atomic_save_v88.py` | 截断重采、冻结 slot 恢复、Windows 原子保存重试 |
| | `test_qfft_supplement_v91.py` | 不完整题一次性补跑入口 |

## 边界

- 重构前基线（含已归档的旧测试）为 `12 failed, 583 passed, 2 skipped`，12 个失败全部属于已归档范围，原始输出见 `docs/migration/pytest_baseline_before_restructure.txt`。
- 重构后只跑本目录的 25 个测试文件。2026-11-08 重构收尾实测：**172 passed / 0 failed**（约 23s），输出见 `docs/migration/pytest_after_restructure.txt`；早前 `1 failed, 171 passed` 快照中的失败（`data/aimo_sft_v26/manifest.json` 的批准路径旧写法）已随数据清单改写修复。
- 唯一例外：`qfft/test_qfft_ds_dataset_v63.py` 从 `archive/tools/` 只读加载 v63 造数模块（v63 未进入现行主线，`tools/qfft/` 只有 v66+），仅为保留该历史协议的行为验证；其余测试只加载 `src/`、`scripts/`、`tools/` 下的现行代码。
- 已归档的 70 个历史测试在 `archive/tests/`（例如 v1…v92 早期 smoke、评测与 QFFT 阶段测试），它们依赖旧路径与旧数据布局，**不保证可直接运行**，不要把它们重新加回 `testpaths`。
- 新增测试必须落在当前分组内，并确保不依赖本地大数据集或网络；涉及 GPU/API 的路径用离线分支覆盖。
- 例外：[v93_harness/](v93_harness/README.md) 不是 pytest 用例（文件名不以 `test_` 开头，pytest 不会收集），而是 v93 的 bash launcher、VM 阶段、rclone/GitHub/DeepSeek 客户端的模拟测试，用 `bash tests/v93_harness/run_all.sh` 运行（8 组，共 214 项）。
