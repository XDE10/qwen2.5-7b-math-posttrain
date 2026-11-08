# experiments/qfft_v93：1000 题双教师 × SFT/QFFT 全参 × LoRA-GRPO 管线

v93 取代 `aimo_sft_v26`（LoRA、8K 窗口、静态 YaRN、并发 4 的评测）。一条命令在一台全新的 Colab VM 上跑一个或多个阶段，所有状态都在 Google Drive 上：VM 随时可以被回收，重跑同一条命令就从最近的检查点或已保存的生成继续。

方案和面试材料见 [docs/blueprints/三周冲刺方案_v93.md](../../docs/blueprints/三周冲刺方案_v93.md) 和 [docs/interview/面试追问与参考答案.md](../../docs/interview/面试追问与参考答案.md)。

## 实验矩阵

| 名称 | 教师轨迹 | 格式 | 初始化 | 说明 |
|---|---|---|---|---|
| `qfft_ds_v93` | DeepSeek-V4.1-Flash（自建） | QFFT | Qwen2.5-7B-Instruct | 主线 |
| `sft_ds_v93` | 同上，response 逐字相同 | SFT | 同上 | 与 QFFT 只差"有没有题目" |
| `qfft_r1_v93` / `sft_r1_v93` | s1K-1.1 的 R1 轨迹 | QFFT / SFT | 同上 | 受控的教师对比 |
| `grpo_qfft_ds_v93` | 920 + 637 题 RL 池 | LoRA-GRPO | `qfft_ds_v93` 合并权重 | 主线 RL |
| `grpo_sft_ds_v93` | 同上 | LoRA-GRPO | `sft_ds_v93` | 可选：同预算比较 RL 起点 |
| 锚点 | — | — | E0、QFFT 作者权重、公开 SFT 孪生、Qwen2.5-Math-7B-Instruct | 只评测 |

固定配方（[configs/train/_defaults.json](configs/train/_defaults.json)）：全参、LR 1e-5、cosine、warmup 10%、6 epoch、wd 1e-4、AdamW(0.9, 0.999)，与 QFFT 官方 yaml 相同；只改两处：有效 batch 16（官方 8，取 s1.1-7B 的值，CPU-Adam 步数减半）和 32K 窗口不截断（官方 16K 截断）。系统实现为 DeepSpeed ZeRO-2 + CPU AdamW offload、梯度检查点、Liger。

## 一、本地（Windows 工作站，CPU）：把 1000 题数据做完

在仓库根目录，用 `D:/anaconda3/envs/genai-test/python.exe`（需要 pandas、pyarrow、tokenizers；Math-Verify 用 `.pilot_deps`）：

```bash
python -m experiments.qfft_v93.fetch_benchmarks --sources   # 固定 revision 下载评测集，核对 MATH-500 封存摘要
python -m experiments.qfft_v93.data_build plan              # 1000 题计划：替换与 MATH-500 同题的 s1k11/row/0918，去污
python -m experiments.qfft_v93.data_build status            # 哪些题还没有可用的 DeepSeek 轨迹
python -m experiments.qfft_v93.teacher fill --fallback-official   # 调 API 补齐，可反复运行
python -m experiments.qfft_v93.data_build build --ds-verified     # 生成 r1/、ds/、ds_verified/ 与教师报告
```

- 准入规则（两位教师、1000 题一律相同）：`finish_reason == "stop"`、思考与最终回答都非空、SFT 编码后 ≤ 32768 token。按时间顺序取第一次可用的尝试（v66 → v86 → v91 → v93），不按答案对错挑选，不截断。
- `teacher fill` 只为还没有可用轨迹的题请求 API，每题最多 8 次（`--max-attempts`），effort 依次为 `high,high,max,...`，提示词默认用 v91 的防重复版本。每次尝试单独落盘到 `artifacts/qfft_teacher_v93/attempts/<index>/aNN/`，中断后重跑会接着做。
- 可选的 `teacher verify` 会为"可用但答案可核对且错误"的题继续采样，`build --ds-verified` 再用第一条答对的可用尝试替换；主矩阵仍用 `ds`。
- `data/qfft1000_v93/teacher_report.md` 给出两位教师在同一批可核对题上的正确率、对错矩阵、长度和反思词频率。参照值：R1 在 s1K-1.1 的 742 道可核对题上答对 62.7%。

## 二、Colab：一键运行

需要 Linux、macOS 或 WSL2（Colab CLI 不支持原生 Windows）：`uv tool install google-colab-cli`，然后 `colab auth`/登录一次。

```bash
experiments/qfft_v93/colab/pipeline.sh 0              # CPU VM：上传数据，下载固定 revision 的模型到 Drive
experiments/qfft_v93/colab/pipeline.sh 1 --gpu G4     # 冒烟：2 步全参、24 条生成、2 步 GRPO
experiments/qfft_v93/colab/pipeline.sh 2 --gpu G4     # 锚点评测
experiments/qfft_v93/colab/pipeline.sh 3 --gpu H100   # DeepSeek 对：QFFT、SFT，各自训完立即评测
experiments/qfft_v93/colab/pipeline.sh 4 --gpu H100   # LoRA-GRPO + 合并 + 评测
experiments/qfft_v93/colab/pipeline.sh 5 --gpu H100   # R1 对
experiments/qfft_v93/colab/pipeline.sh 7              # CPU VM：重新判分、results.md、图
```

单个阶段可以直接用 `launch.sh`，例如 `launch.sh --gpu G4 eval:run:qfft_ds_v93:full`。`launch.sh --gpu cpu status` 打印所有阶段、训练进度和评测完成度。

每次调用的过程：`colab new --gpu ... --high-mem` → `colab drivemount` → 上传代码包 → `colab exec` 执行 `colab/remote.py` → 阶段全部成功后释放 VM（`--keep` 保留，便于 `colab console` 进去排查）。本地日志在 `reports/colab_logs/<session>.log`，VM 端日志在 Drive `runs/project1/v93/logs/`。

## 三、断点与续跑

| 对象 | 本地（VM SSD） | Drive（持久） | 新 VM 上怎么继续 |
|---|---|---|---|
| 全参训练 | `checkpoint-N`（约 122 GB：bf16 权重 + fp32 主权重 + Adam 状态） | `ckpt_mirror/checkpoint-N` + `MIRROR_COMPLETE.json`，后台限速复制 | 自动从最新完整镜像拷回本地，`resume_from_checkpoint` 续跑，LR 调度与数据顺序不变 |
| GRPO | adapter 检查点（约 1 GB），每 10 步 | 同上，保留 3 个 | 同上 |
| 评测生成 | 每条请求结束即追加 `raw.jsonl` | 每分钟同步一次 | 合并本地与 Drive 记录，只补缺失的 (题, 第几次采样) |
| 判分 | — | `scored.jsonl`（带判分器版本） | 增量判分，`--rescore` 全部重判 |
| 环境 | `/content/p1/venvs/{infer,rl}` | `envcache/venvs/*.tar` 快照 + 冻结版本清单 | 约 1 分钟恢复 |

- 训练镜像在后台复制，本地剩余空间低于 24 GB 时暂停复制，等 Drive 客户端上传腾出缓存；每 63 步（约 1 epoch）一次。镜像失败只记警告，训练继续，上一份完整镜像仍有效。
- run 目录里的 `run_identity.json` 记录配方与数据 SHA；配方或数据变了会拒绝在同名 run 上续跑。
- 训练完成后导出 bf16 safetensors 到 `runs/<run>/final/`，写 `TRAIN_COMPLETE.json`，删除镜像，并把权重留在本地 SSD 上供同一 VM 的评测直接使用。

## 四、评测口径（[configs/pipeline.json](configs/pipeline.json)）

- 所有模型用同一组 prompt token：system `Please reason step by step, and put your final answer within \boxed{}.` + user 题目，用基座 tokenizer 渲染。
- vLLM，bf16，原生 RoPE，`max_model_len 32768`（Qwen2.5-Math 按其配置用 4096），`max_tokens = max_model_len − prompt`，temperature 0.6，top_p 0.95，每个 (题, 第几次) 有固定种子。
- 采样数：MATH-500 4、AIME24 16、AIME25 16、AMC23 8、GSM8K 1。
- 判分（`grading.py`）：只看最后一个 `<|im_start|>answer`（或 `</think>`）之后的最终段，取最后一个 `\boxed{}`，用 Math-Verify 0.9.0 比较；判分进程有硬超时，超时和异常单独计数，不算答错。主指标不用 LLM 复核。
- 指标（`metrics.py`）：mean@n ± 按题 bootstrap 95% CI、无偏 pass@k、maj@n、平均 token（分对错）、截断率、尾部重复率、反思词触发率、按 MATH 难度分层、不加强制收尾的"思考预算曲线"；`score.py --compare A:B` 做按题配对 bootstrap。

## 五、文件

| 文件 | 作用 |
|---|---|
| `formats.py` | prompt 与 response 布局、SFT/QFFT 编码、最终段切分（唯一来源） |
| `grading.py` | 最终段判分、带超时的进程池、多数投票聚类 |
| `metrics.py` | pass@k、bootstrap、配对检验、长度与行为统计 |
| `decontam.py` + [configs/decontam_review_v93.json](configs/decontam_review_v93.json) | 13-gram + 相似度去污，17 对已人工复核 |
| `fetch_benchmarks.py` | 固定 revision 与 SHA-256 下载评测集 |
| `data_build.py` / `teacher.py` | 1000 题计划、准入、API 补齐、两份训练集与教师报告 |
| `train_sft.py` | 全参 SFT/QFFT（DeepSpeed ZeRO-2 offload、Liger、Drive 镜像续跑） |
| `generate.py` / `score.py` / `analyze.py` | 生成、判分汇总、图表 |
| `grpo.py` / `merge_lora.py` | TRL GRPO + LoRA + 同卡 vLLM、合并导出 |
| `checkpointing.py` / `telemetry.py` / `models.py` / `common.py` | 镜像与恢复、SwanLab、模型准备、公共工具 |
| `colab/launch.sh` / `colab/pipeline.sh` / `colab/remote.py` / `colab/stages.py` | 一键调度 |
| `envs/*.txt` | 训练附加包、推理与 RL 虚拟环境（2026-11-08 在 PyPI 核对过版本约束） |

## 六、已验证与未验证

- CPU 上已实跑：模板与官方 S1_QFFT 数据逐字一致（1000/1000）；SFT/QFFT 编码不变量；判分器 10 个边界用例；微型 Qwen2 上的全参训练、模拟 VM 丢失后从 Drive 镜像恢复并续跑到结束、SFT 格式、身份校验；假引擎生成的按采样粒度续跑与判分汇总；TRL 1.17.1 GRPO + LoRA + 奖励 + 镜像 + 恢复 + 合并；合成数据上的 plan/status/fill/build 全流程；launch.sh 与 remote.py 的调度流程（模拟 CLI）。
- 只能在 GPU 上验证：DeepSpeed CPU-Adam 的速度与主机内存、Liger 在 32K 下的显存、vLLM 0.32/0.33 在 G4（Blackwell）与 H100 上的吞吐、122 GB 检查点镜像到 Drive 的实际耗时。阶段 1 的冒烟就是为这些准备的，跑完先看日志里的 `s/step`、`peakGPU`、`host/rss_gib`。
