# artifacts/ —— 造数与评测原始证据（不进 git）

保存 QFFT 造数、数学评测、依赖探针、研究对照的**原始请求/响应/候选/复核**证据。**整个目录被 `.gitignore` 忽略**，只有本 README 入库；体积约 1.7 GB。

## 目录

| 目录 | 内容 |
|---|---|
| `qfft_ds_dataset_v66/` | 首批 50 题及后续重采的候选、请求、原始思考 + 最终回答、审查分片、ZIP 校验 |
| `qfft_ds_dataset_v63/` | v63 造数入口的旧输出（被 v66 取代，保留作对照） |
| `qfft_length_retry_v86/` | 截断题有界重采（max 一次 → high 一次）的请求/响应、恢复验证、人工复核 |
| `qfft_incomplete_high_v91/` | 不完整题一次性 high 补跑的证据 |
| `qfft_all_supervisor_v71/` | 全量 1000 题的监督器状态、`status.json`、`run.log`、`plan.json`、熔断复核记录、恢复验证 |
| `final_eval_local_snapshot/` | 历史评测的本地快照：MATH-500 / dev256 / RomEscape / P-ALIGN 等 `evaluations/*` 与 `native_ds_*` 复核 |
| `dependency_probe_v46/` | v46 容量/速度短测的依赖与加载探针 |
| `research_s1_7b/` | s1K-1.1 作者训练记录证据（TensorBoard 标量、training_args 反汇编等） |
| `research_finetuning_sampling/` | 微调与采样策略研究留档 |

## 使用原则

- 证据目录是**只读历史快照**：不要覆盖旧的 `request` / `error` / `candidate` / `partial` 文件；新增尝试放在独立子目录（如 `manual_retry_v84/`、`resume_v89_validation.json`）。
- 溯源优先看目录内的 `*_validation.json`、`circuit_reviews/`、`manual_reviews/` 与 SHA 清单；报告正文在 `reports/runs/<同名报告>/`。
- 原始证据不等于质量批准：候选模型输出全部 `training_approved=false`，能否入训看 manifest 与人工审批记录。
- 归档化重构没有移动本目录；`reports/` 中的旧报告若引用 `artifacts/<旧名>`，以实际目录名为准。
- 大体积证据不纳入 git；如需交付，用 `tools/review/build_external_review_zip_v92.py` 一类脚本生成 UTF-8 索引和 ZIP，并记录 SHA。
