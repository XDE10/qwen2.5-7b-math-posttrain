# tools/ —— 现行工具

**23 个现行工具**（另有本 README），按用途分组。历史工具已移入 `archive/tools/`（映射计划 42 个，实际 40 个：`start_aimo_sft_v26.py` 与 `start_aimo_sft_v26_notify.py` 在重构后移回本目录）。

## aimo/（6）

| 文件 | 职责 |
|---|---|
| `start_aimo_sft_v28.py` | **最新启动入口**：自动读私密 SwanLab Key → 准备依赖 → 启动固定 v26 配方 |
| `start_aimo_sft_v27.py` | v27 前置兼容补丁（torchao / warmup_steps / 在线认证）后训练 v26 |
| `start_aimo_sft_v26.py` | 早期入口：重建 v26 包并自动开始训练 |
| `start_aimo_sft_v26_notify.py` | 本地 VS Code 通知扩展接收 Colab 完成信号 |
| `package_aimo_sft_v26.py` | 打包全部合格原始轨迹与独立训练/评测入口 |
| `prepare_aimo_sft_v26.py` | 冻结所有适配 Qwen 8192 窗口的完整 AIMO2 轨迹 |

## evaluation/（3）

| 文件 | 职责 |
|---|---|
| `evaluate_sft_ds_v29.py` | MATH-500 主评测：原生 Math-Verify 打分 + DeepSeek 失败复核 |
| `evaluate_math500_final_answer_v41.py` | step429 MATH-500：7936 推理 + 可选 256 答案 token 的收尾评测 |
| `review_math500_answer_v42.py` | 本地 DS 复核 v41 原生失败样本，完整保留原始与最终输出 |

## qfft/（12）

| 文件 | 职责 |
|---|---|
| `build_qfft_ds_dataset_v66.py` | 造数入口：默认只准备，`--run` 才在非高峰发付费/订阅请求 |
| `qfft_ds_common_v66.py` | 冻结题目与盲评教师契约；导入本模块绝不调用 API |
| `qfft_stream_transport_v79.py` | 持久化 OpenAI 风格 SSE 传输；不重试、不保存凭证 |
| `continue_qfft_all_v71.py` | 监督已授权的 1000 题完成，保留 v66 既有输出 |
| `archive_qfft_followups_v70.py` | 归档既有 QFFT 回答供后续处理；不调用 API/GPU |
| `start_qfft_supervisor_v73.pyw` | 计划任务入口：独立进程宿主、持久 stdout/stderr、不重试 |
| `retry_length_max_v86.pyw` | 用户授权的有界 max → high 重采；原 1000 题只读 |
| `supplement_incomplete_high_v91.pyw` | 明确授权的一次性 high 补跑（有效 62 道不完整题） |
| `watch_qfft_progress_v72.pyw` | 只读原生进度窗口；关闭窗口不停止造数 |
| `watch_qfft_length_retry_v86.pyw` | 只读监控；关闭窗口不停止两个 worker 的任务 |
| `watch_incomplete_high_v91.pyw` | 62 题 high 补跑的只读状态窗口 |
| `查看造数进度.cmd` | 双击即可打开只读进度窗口（使用 `D:/anaconda3/envs/genai-test/pythonw`） |

## review/（1）与 verification/（1）

| 文件 | 职责 |
|---|---|
| `review/build_external_review_zip_v92.py` | 生成外部深度审查包（索引 + ZIP + SHA） |
| `verification/verify_migration.py` | 离线迁移校验，可在任意工作目录运行 |

## 边界

- 所有 `.pyw` 与 `查看造数进度.cmd` 是 Windows 侧后台/监控入口，只读监控关闭不会杀任务；重跑造数前必须确认没有同任务在跑。
- 造数与付费 API 请求需要当次明确授权；`--run` 之外不发起请求。
- 历史工具在 `archive/tools/`（40 个：各代启动器、评测器、造数/审查/监控脚本），只作追溯，不保证可直接运行；旧路径可用 `archive/mapping/restructure_v2_mapping.json` 找回。
