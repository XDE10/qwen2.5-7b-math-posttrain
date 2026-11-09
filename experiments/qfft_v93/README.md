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

造数要在有本地数据的那份仓库里跑（`D:\Users\Mr.Z\Desktop\项目一_7B数学后训练`：`reports/`、`artifacts/`、`data/` 都只在这里，git 不管）。如果平时在 WSL 里另有一份克隆用来跑 Colab，Windows 这份也要先 `git pull`。

一键：双击 `experiments\qfft_v93\windows\v93_local_data.cmd`，它按顺序跑下面五步，出错即停，重跑会续做（解释器默认 `D:\anaconda3\envs\genai-test\python.exe`，可用环境变量 `P1_PYTHON` 改；第 4 步的额外参数用 `P1_FILL_ARGS`，例如 `set P1_FILL_ARGS=--max-attempts 12`）。

手动：在仓库根目录，用 `D:/anaconda3/envs/genai-test/python.exe`（需要 pandas、pyarrow、tokenizers；Math-Verify 用 `.pilot_deps`）：

```bash
python -m experiments.qfft_v93.fetch_benchmarks --sources   # 固定 revision 下载评测集，核对 MATH-500 封存摘要
python -m experiments.qfft_v93.data_build plan              # 1000 题计划：替换与 MATH-500 同题的 s1k11/row/0918，去污
python -m experiments.qfft_v93.data_build status            # 哪些题还没有可用的 DeepSeek 轨迹
python -m experiments.qfft_v93.teacher fill --fallback-official   # 调 API 补齐，可反复运行
python -m experiments.qfft_v93.data_build build --ds-verified     # 生成 r1/、ds/、ds_verified/ 与教师报告
```

- 准入规则（两位教师、1000 题一律相同）：`finish_reason == "stop"`、思考与最终回答都非空、SFT 编码后 ≤ 32768 token。按时间顺序取第一次可用的尝试（v66 → v86 → v91 → v93），不按答案对错挑选，不截断。
- `teacher fill` 只为还没有可用轨迹的题请求 API，每题最多 8 次（`--max-attempts`），effort 依次为 `high,high,max,...`，提示词默认用 v91 的防重复版本。每次尝试单独落盘到 `artifacts/qfft_teacher_v93/attempts/<index>/aNN/`，中断后重跑会接着做。
  - 开始前先读 Key（Go 的 opencode.db 或 `OPENCODE_GO_API_KEY`；官方 Key 文件或 `DEEPSEEK_API_KEY`），找不到就直接退出，一个请求都不发、一次额度都不占。
  - 额度只算"拿到或可能拿到了输出"的尝试（正常返回、流中途断开、进程被打断）；服务端在生成前就拒绝的请求（配额、限流、HTTP 5xx）占一个编号，不占额度。
  - 鉴权或请求格式错误（HTTP 400/401/403/404/413/422）立即停止整个运行；连续 5 次或单次运行累计 20 次失败也会停（服务端故障时不会把每道题的额度都耗光）。退出码 2，修好原因后重跑即可。
  - 还有题用完额度仍没有可用轨迹时退出码为 3，并列出这些题；加大 `--max-attempts` 重跑。
- s1K-1.1 parquet 与 tokenizer 先用本地旧副本；本地副本缺失或哈希不对时，自动改用第 1 步按固定 revision 下载并核对过的 `data/benchmarks_v93/_sources/` 副本（同一 revision、同样的字节）。
- `data_build status` 的预期量级（按 history 里 v66/v86/v91 的记录推算）：v66 最多约 911 题可用（747 + 164 道完整轨迹，按 Qwen tokenizer 超过 32,768 的会被拒），v86 最多 25 题（补全的 27 题里有 2 题超过 32K），v91 最多 62 题；缺口至少 1 题（第 918 题已换成新题，旧轨迹判为 `different_question`）。如果大量出现 `different_question` 或 `missing_reasoning`，说明文件对不上，先不要跑第 4 步。
- 可选的 `teacher verify` 会为"可用但答案可核对且错误"的题继续采样，`build --ds-verified` 再用第一条答对的可用尝试替换；主矩阵仍用 `ds`。
- `data/qfft1000_v93/teacher_report.md` 给出两位教师在同一批可核对题上的正确率、对错矩阵、长度和反思词频率。参照值：R1 在 s1K-1.1 的 742 道可核对题上答对 62.7%。

## 二、Colab：一键运行

需要 Linux 或 WSL2（Colab CLI 不支持原生 Windows）。以下内容按 google-colab-cli 0.7.5 的源码与 `--help` 核对过（2026-11-08）。

Windows 上的一键方式：管理员 PowerShell 里 `wsl --install` 并重启一次；之后都用 `experiments\qfft_v93\windows\v93_colab.cmd`：`setup`（在 WSL 里装 uv 与 Colab CLI、检查仓库和换行符、登录）→ `ping` → `0`、`1`、`2`、`d`、`3`……（参数原样传给 pipeline.sh，例如 `v93_colab.cmd 3 --gpu G4`），另有 `status`、`attach SESSION`。双击不带参数时会提示输入。仓库根目录的 `.gitattributes` 让 `*.sh` 在任何 `core.autocrlf` 设置下都检出为 LF；在 `/mnt/d` 下若提示 Permission denied，用 `bash experiments/qfft_v93/colab/pipeline.sh ...` 运行。

1. 安装：`uv tool install google-colab-cli`（或 `bash experiments/qfft_v93/colab/setup_wsl.sh`）。CLI 要求 Python ≥ 3.12，uv 会自动下载。
2. 登录：第一次运行任何 `colab` 命令会打印一个 Google 授权链接，在浏览器里同意后把页面显示的授权码粘贴回终端，令牌缓存在 `~/.config/colab-cli/token.json`。`colab auth` 是给 VM 内代码授权 GCP 的，不是 CLI 登录。
3. Drive：第一次 `colab drivemount` 会再打印一个"允许 Colab 访问 Drive"的链接，同意后按回车。之后新 VM 自动挂载，不再提示。
4. 先花一分钟 CPU 做端到端检查：`launch.sh --gpu cpu ping:30`（开 VM、挂 Drive、上传、执行、写日志、释放）。

```bash
# 阶段 0–2 不依赖本地数据，可以和第一节的造数并行
experiments/qfft_v93/colab/pipeline.sh 0   # CPU VM：在 VM 上按固定 revision 下载评测集和模型到 Drive
experiments/qfft_v93/colab/pipeline.sh 1   # 冒烟（最坏长度，H100 优先，其次 G4），见下文
experiments/qfft_v93/colab/pipeline.sh 2   # 锚点评测（G4 优先，其次 H100）
# 本地 data_build build 完成后
experiments/qfft_v93/colab/pipeline.sh d   # CPU VM：上传 data/qfft1000_v93（及评测集），核对哈希
experiments/qfft_v93/colab/pipeline.sh 3   # DeepSeek 对：QFFT、SFT，各自训完立即评测
experiments/qfft_v93/colab/pipeline.sh 4   # LoRA-GRPO + 合并 + 评测
experiments/qfft_v93/colab/pipeline.sh 5   # R1 对
experiments/qfft_v93/colab/pipeline.sh 7   # CPU VM：重新判分、results.md、图
```

- 冒烟按最坏情况设计，量出来的峰值就是正式训练的上限：
  - 全参：32 条恰好填满 32,768 token 的序列（`smoke_data.py`，由公开的 s1K-1.1 轨迹按训练格式拼接截取，只用于测量），真实配方跑 2 步并本地存一次 122 GB 检查点。
  - GRPO：真实批形状（8 题 × 8 采样），用 vLLM 的 `min_tokens` 把每条 completion 强制生成到 8192 token，跑 2 步并存检查点。
  - 训练类阶段开始前检查主机内存：低于 128 GiB 直接报错（ZeRO-2 offload 的优化器状态约 122 GB），低于 150 GiB 给出警告。
- 后面可以接 launch.sh 的参数，例如 `pipeline.sh 3 --gpu G4 --wait-min 180`：只用 G4，分不到卡时每 5 分钟重试，最多 3 小时。`--gpu` 支持逗号列表（按顺序尝试）；不认识的卡名会直接报错，因为 CLI 会把未知名称静默当成 A100。
- 单个阶段直接用 `launch.sh`，例如 `launch.sh --gpu G4 eval:run:qfft_ds_v93:full`；`launch.sh --gpu cpu status` 打印所有阶段（运行中的会显示所在 VM 和心跳）、训练进度和评测完成度。
- 每次调用：`colab new --high-mem [--gpu ...]` → `colab drivemount` 并检查 `/content/drive/MyDrive` → 上传代码包（大文件按 8 MiB 分片，VM 端重组并核对 SHA-256）→ `colab exec` 执行 `colab/remote.py` → 看到 `P1_RESULT` 后释放 VM（`--keep` 保留，便于 `colab console` 排查）。本地日志在 `reports/colab_logs/<session>.log`，VM 端日志在 Drive `runs/project1/v93/logs/`。

**断线与重复启动**：任务跑在 VM 的 kernel 里，不依赖本地终端。

- 网络断开时，launch.sh 改为每分钟用 `colab download` 读 VM 上的 `session_tail.json`，直到拿到结果。
- 关掉终端或按 Ctrl+C 只是脱离（退出码 75，pipeline.sh 不会接着开下一台 VM）：任务继续，结束后 VM 自己释放。重新跟踪用 `launch.sh --attach <session>`，中止用 `colab stop -s <session>`。
- 每个阶段在 Drive 上有带心跳的标记文件。另一台 VM 想跑同一阶段时，若心跳在 15 分钟内就拒绝，避免两台 VM 同时写一个 run 目录；确认原 VM 已不在时可加 `--force`。

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
| `fetch_benchmarks.py` | 固定 revision 与 SHA-256 下载评测集（本地或 VM 上的 `benchmarks` 阶段） |
| `smoke_data.py` | 冒烟用的最坏长度序列（每条恰好填满训练窗口，只用于测量） |
| `data_build.py` / `teacher.py` | 1000 题计划、准入、API 补齐、两份训练集与教师报告 |
| `train_sft.py` | 全参 SFT/QFFT（DeepSpeed ZeRO-2 offload、Liger、Drive 镜像续跑） |
| `generate.py` / `score.py` / `analyze.py` | 生成、判分汇总、图表 |
| `grpo.py` / `merge_lora.py` | TRL GRPO + LoRA + 同卡 vLLM、合并导出 |
| `checkpointing.py` / `telemetry.py` / `models.py` / `common.py` | 镜像与恢复、SwanLab、模型准备、公共工具 |
| `colab/launch.sh` / `colab/pipeline.sh` / `colab/remote.py` / `colab/stages.py` | 一键调度 |
| `colab/setup_wsl.sh`、`windows/v93_colab.cmd`、`windows/v93_local_data.cmd` | WSL 一次性安装与登录；Windows 下的 Colab 阶段入口与本地造数入口 |
| `envs/*.txt` | 训练附加包、推理与 RL 虚拟环境（2026-11-08 在 PyPI 核对过版本约束） |

## 六、已验证与未验证

- CPU 上已实跑：模板与官方 S1_QFFT 数据逐字一致（1000/1000）；SFT/QFFT 编码不变量；判分器 10 个边界用例；微型 Qwen2 上的全参训练、模拟 VM 丢失后从 Drive 镜像恢复并续跑到结束、SFT 格式、身份校验；假引擎生成的按采样粒度续跑与判分汇总；TRL 1.17.1 GRPO + LoRA + 奖励 + 镜像 + 恢复 + 合并；合成数据上的 plan/status/fill/build 全流程；launch.sh 与 remote.py 的调度流程（按 google-colab-cli 0.7.5 行为写的模拟 CLI，11 个场景）；冒烟序列用真实 Qwen tokenizer 生成，QFFT 32 条、SFT 4 条都恰好 32,768 token。
- 只能在 GPU 上验证：DeepSpeed CPU-Adam 的速度与主机内存、Liger 在 32K 下的显存、vLLM 0.32/0.33 在 G4（Blackwell）与 H100 上的吞吐、122 GB 检查点镜像到 Drive 的实际耗时。阶段 1 的冒烟按最坏长度测这些，跑完先看日志里的 `s/step`、`peakGPU`、`host/rss_gib`。
