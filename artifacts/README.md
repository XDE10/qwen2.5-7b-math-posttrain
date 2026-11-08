# artifacts 索引

造数与评测的原始证据（请求/响应/候选/复核）。体积大、不入 git，但为本项目结论的可追溯来源；删除前必须确认对应报告已完整固化。

| 目录 | 说明 |
|---|---|
| `dependency_probe_v46/` | v46 依赖安装失败排查的现场证据 |
| `final_eval_local_snapshot/` | 本地评测快照（v32–v58 各模型 MATH500/dev 原始输出与复核） |
| `qfft_all_supervisor_v71/` | v71 监督器状态、熔断复核与恢复验证 |
| `qfft_ds_dataset_v63/` | v63 首轮造数配置与准备状态（后被 v66 取代，仅留档） |
| `qfft_ds_dataset_v66/` | QFFT 1000 题造数主证据（v66/v84/v85 恢复、请求/响应/候选） |
| `qfft_incomplete_high_v91/` | v91 补跑不完整题（high 档）证据 |
| `qfft_length_retry_v86/` | v86 有界重采（41 题 max/high 两档）证据 |
| `research_finetuning_sampling/` | 微调与采样文献研究素材（含 upstream 克隆） |
| `research_s1_7b/` | s1.1-7B 公开训练配置/证据只读研究 |
