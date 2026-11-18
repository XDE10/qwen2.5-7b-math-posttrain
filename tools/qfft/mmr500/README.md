# MMR 500 步与开发集监督入口

本目录保存 2026-11-18 已部署监督脚本的源代码。运行目录为
`/mnt/d/CodexArtifacts/project1-mmr-step500-20261118`，凭证只从用户的
`~/.config/p1/` 读取，未包含在仓库或代码包中。训练数据、完整状态、合并权重、
`READY.json` 和实际冻结代码包仍在本地及 Drive；这些脚本依赖已准备好的部署目录，
不应在另一台机器上直接运行来创建重复任务。

- `guardian500.py` 等待原 150 步完整归档及父会话释放，再在账号 C 续到全局 500；
  仅操作 `p1-mmr500-c-`，检测动态父会话并验证完整恢复状态。
- `resume500.sh` 在申请卡前核对当前代码包和冻结 4000 题的 SHA-256。
- `start_guardian.sh` 幂等启动训练监督器。
- `validation25/controller.py` 等待 150、175…500 的已导出模型，按 FIFO 在账号 A
  运行开发集；已有第 100 步补评已完成。每次验证后确认结果持久化并释放自己的 VM。
- `validation25/run_validation.sh` 校验冻结工作树；推理使用保留的原始 c7bcd9df7b7e
  代码包及现有 infer 环境快照，不依赖另一工作树的未提交文件顺序。
- `validation25/validation_worker.py` 调用
  `experiments/qfft_v93/validation.py`，基线缓存按原协议核验后复用。
- `validation25/preserve_fallback.py` 是原 150 步训练上传受限时部署的侧进程：
  保护本地完整断点，校验后保存到备用盘，不改训练参数或挂载。

已部署进程仍使用 D 盘的冻结副本；修改仓库里的副本不会热更新它们。提交、合并后
必须重新核对启动包身份，保留旧包，并只重载仍在等待中的本地监督器。
实际 GPU 上正在运行的代码和原 150 步身份不得重写。

状态和恢复命令：

```bash
cat /mnt/d/CodexArtifacts/project1-mmr-step500-20261118/guardian_state.json
cat /mnt/d/CodexArtifacts/project1-mmr-step500-20261118/validation25/controller_state.json
bash /mnt/d/CodexArtifacts/project1-mmr-step500-20261118/start_guardian.sh
bash /mnt/d/CodexArtifacts/project1-mmr-step500-20261118/validation25/start_controller.sh
```

本轮授权、参数和真实评测结果见
[MMR 进展记录](../../../docs/status/mmr_dr500_validation_20261118.md)。
