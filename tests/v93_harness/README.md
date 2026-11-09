# tests/v93_harness —— v93 调度、教师与 Drive 的模拟测试

这些检查原来只在云端沙箱里（`/root/p1work/harness`），2026-11-09 迁到仓库。它们覆盖 pytest 测不到的部分：bash 写的 launcher、VM 上的阶段运行器、rclone 传输、GitHub 报告推送、DeepSeek 流式客户端。外部服务全部用替身：

- `mockbin/colab`：按 google-colab-cli 0.7.5 的行为写的 Colab CLI（`new`、`exec`、`upload`、`stop`、`drivemount`），每台"VM"是临时目录，`exec` 在独立进程里跑代码，断线不影响任务。
- `mockbin/rclone`、`mountpoint`、`apt-get`：`mydrive:` 映射到本地目录，挂载就是一个符号链接；`MOCK_RCLONE_UPLOAD_LIMIT=1` 模拟 Drive 每日上传上限。
- GitHub 用本地 bare 仓库代替；DeepSeek API 用 `teacher_mock.py` 里的本地 SSE 服务。
- `setupbin*/`、`localbin/`：`setup_wsl.sh`、`check_local.sh` 用的 colab/uv/rclone 替身。

## 运行

```zsh
bash tests/v93_harness/run_all.sh                        # 全部 8 组（scenarios 约 5–25 分钟，视机器而定）
bash tests/v93_harness/run_all.sh drive teacher data     # 指定几组
P1_SCENARIOS="A R2 RP SD" bash tests/v93_harness/run_all.sh scenarios
```

`run_all.sh` 先把工作区（git 跟踪的和未被忽略的新文件）复制成临时快照，在快照上运行，结束删除，所以不会碰仓库里的 `data/`、`reports/`。`scenarios.sh` 和 `check_local_check.sh` 发现自己不在快照里会拒绝运行（场景 G 会删除并重建 `data/qfft1000_v93`）。

依赖：bash、git、tar、python3 ≥ 3.10（只用标准库）。`teacher`、`pinned`、`data` 三组还要一个装了 `tokenizers` 和 `math-verify` 的解释器（`P1_PYTHON`，默认 `python3`）和 Qwen2.5-7B-Instruct 的 `tokenizer.json`（`local_data.sh` 跑过后在 `data/` 下自动找到，或设 `P1_HARNESS_TOKENIZER`）；缺了就跳过这三组，不算失败。

## 各组内容

| 组 | 文件 | 项数 | 内容 |
|---|---|---|---|
| scenarios | `scenarios.sh` | 96 | launch.sh / pipeline.sh / remote.py / stages.py：正常流程、GPU 回退与 high-RAM 降级、未知 GPU 名、断线后轮询、两台 VM 抢同一阶段、脱离与 `--attach`、大文件分片上传、Drive 未挂载、失败阶段、主机内存检查、`--data-root`、冒烟总结、rclone 挂载（缺配置、成功、失败）、报告推送到 bare 仓库（token 不外泄）、sync-data 拒收预览构建与接受成对剔除 |
| drive | `drive_units.py` | 28 | drive_io 的上传/下载/永久删除，检查点镜像只留最新与新 VM 恢复，Drive 每日上传上限（立即报错、训练继续），JSONL 镜像，挂载拒绝改名时 sync_file 直接覆盖，仓库地址规范化与脱敏 |
| smoke | `smoke_report_check.py` | 9 | 冒烟总结的每一行，包括失败阶段的报错行 |
| teacher | `teacher_scenarios.py` + `teacher_mock.py` | 28 | 缺 Key 不发请求、401 立即停、5xx 熔断、SSE 收尾、额度耗尽退出码 3、配额回退、无 Go Key 时用官方 API、low effort 请求的字段 |
| pinned | `pinned_inputs_check.py` | 5 | tokenizer / s1K 本地副本不对时改用固定 revision 的下载副本 |
| data | `data_build_check.py` | 15 | build 拒绝缺口、预览构建、成对剔除（所有变体题目相同）、DS 来源统计与教师报告 |
| setup | `setup_check.sh` | 18 | setup_wsl.sh：保留已装 CLI、缺配置报错、token 提示输入且不回显、权限 600 |
| local | `check_local_check.sh` | 15 | check_local.sh：各项检查、权限与代理提示、密钥不出现在输出里、`--no-network` |

`make_fake.py` 生成 teacher/data 两组用的假仓库根（1000 题计划，997 题有可用的 v66 轨迹，第 999 题只有别的题留下的尝试）。
