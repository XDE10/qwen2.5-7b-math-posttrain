# 项目一 v93：迁到本机 Kiro IDE 的交接（2026-11-09）

> 写给在本机 Kiro IDE（WSL2）里继续这个项目的人和 Claude。长期约定在根目录 [AGENTS.md](../../AGENTS.md)（Kiro 自动加载），本机执行约束在 [.kiro/steering/v93-local-execution.md](../../.kiro/steering/v93-local-execution.md)，命令细节在 [experiments/qfft_v93/README.md](../../experiments/qfft_v93/README.md)。本文只写此刻的状态、23 道缺口题的处理方案和下一步命令。

## 1. 现在的状态

| 项 | 状态 | 依据 |
|---|---|---|
| 阶段 0：5 个评测集 + 4 个模型（base、qwen_math、qfft_official、sft_repro）到 Drive | 完成（CPU VM，约 11 分钟） | `colab-reports` 分支 `v93/sessions/p1-1109-174349/` |
| 阶段 1：最坏长度 GPU 冒烟（全参 2 步、评测 24 条、GRPO 2 步） | 完成（G4，41 分钟） | `v93/latest_smoke_summary.txt`，commit `d6a9608` |
| DeepSeek-V4.1-Flash 轨迹 | 977/1000 可用，23 题缺口 | 用户本机 `data_build status` |
| 阶段 2、d、3–7 | 未运行 | |

阶段 1 实测（G4 = RTX PRO 6000 Blackwell Server Edition 96 GB，主机内存 177 GiB，48 CPU，本地盘 236 GiB）：

| 项 | 实测 | 含义 |
|---|---|---|
| 全参一步（16 条 × 32,768 token） | 148 / 142 s（约 3,705 tok/s） | 最坏长度；真实数据每步的 token 少得多，见第 5 节 |
| 显存峰值 | 已分配 28.9 GiB，保留 42.6 GiB | 96 GB 余量很大，H100 80 GB 也够 |
| 主机内存 | RSS 最高 145.3 GiB，最少剩余 27.9 GiB | 177 GiB 够用；低于 128 GiB 的规格会被训练阶段拒绝 |
| 本地存一次全参检查点 | 113.5 GiB，386 s | 每次保存 GPU 空闲约 6.4 分钟 |
| vLLM 启动 | 217 s | 冒烟只有 24 条并发，测得的 772 tok/s 不代表正式评测吞吐 |
| GRPO 一步（64 条输出都强制到 8192 token） | 476 / 474 s，显存 66.6 GiB | 训练和同进程 vLLM（45% 显存）合计；150 步的上限约 20 h，真实输出更短 |
| 检查点上传 Drive | 0.9 GiB 用 32 s（约 29 MiB/s） | 小样本；按这个速度一份 113.5 GiB 的全参镜像约 67 分钟 |
| SwanLab | 在线（run `tv6azdee`、`3l7wkpow`） | Drive 上的 Key 可用 |

日志里有两条警告。一是阶段日志复制到 Drive 时 rclone 挂载拒绝了改名（ENOENT），本次已修：改名失败时直接覆盖。二是 FlashInfer 没有预编译内核，每台新 VM 启动 vLLM 时要现场编译几分钟，这次没处理。

## 2. 23 道缺口题怎么处理

### 事实

- 题号（都是 `s1k11/row/NNNN`）：0021 0024 0049 0057 0268 0339 0459 0576 0693 0704 0731 0740 0745 0769 0808 0829 0840 0857 0870 0920 0943 0974 0993。v66、v86、v91 和 v93 的 8 次尝试（high/max）都没有得到可用轨迹：要么在 32,768 个输出 token 处被截断（`not_stop:length`），要么正常结束但 SFT 编码有 33K–54K（`over_window`）。
- 我用 s1K-1.1 原始 parquet（固定 revision，SHA-256 已核对）看了这 23 题。来源：16 题 AoPS 论坛（NuminaMath-CoT），3 题 Omni-MATH，OlympiadBench 证明、OlympiadBench 物理、TheoremQA 各 1 题，还有 1 题是填字游戏（0870，xword1）。至少 10 题有可核对的短答案。R1 轨迹的 SFT 长度 3,693–19,663（中位 13,727），全在窗口内。
- s1K-1.1 自带的 `deepseek_grade` 列（s1 作者对 R1 答案的判分）在这 23 题上只有 4 个 Yes（17%），全集是 630/1000（63%）。R1 能放进窗口，主要是因为它较早给出结论，而这些结论多数不对。Flash 在 high/max 下没有收尾，一直探索到窗口外。
- [DeepSeek 官方文档](https://api-docs.deepseek.com/guides/thinking_mode)：`reasoning_effort` 有 low、high、max 三档，medium 映射为 high。v91 时只考虑过 medium（等于 high），用户当时选了 high；low 是真正更短的一档，还没用过。

### 几种做法的比较

| 做法 | 教师 | 两位教师题目相同 | 1000 题 | 问题 |
|---|---|---|---|---|
| A 同模型、同提示词，effort 降到 low 再采样 | 不变（effort 本来就是混合的 high/max） | 是 | 大概率 | 这几题的思考更短；每行记录 effort，报告单列条数 |
| B 在提示词里加长度预算 | 模型不变，提示词变 | 是 | 大概率 | 在最难的证明题上强行限长，容易得到压缩、跳步的论证；多一个提示词变体 |
| C 用 R1 轨迹补 DS 臂 | DS 臂混入 R1 | 是 | 是 | 正好在两位教师差别最大的题上混入另一位教师，直接污染 RQ2 |
| D 训练端允许缺口（DS 977、R1 1000） | 不变 | 否 | 否 | 教师效应和题目难度混在一起，两臂数据量也不同 |
| E 成对剔除（两位教师都去掉这些题） | 不变 | 是 | 否（1000 − k） | 四个训练臂同时少最难的 k 题 |

### 结论

先做 A。A 之后仍没有可用轨迹的题按 E 成对剔除。这个顺序在看到结果之前写定，之后不按结果改。不用 C，也不用 D。只有"恰好 1000 题"不能让步时才改用 B，并在报告里单列。

A 不改准入规则、不改提示词、不换教师，只是沿用已经在用的 effort 参数，所以 RQ2 比较的仍是"同一批题、同一准入规则下的两位教师"。R1 在这批题上也是"在预算内给出结论（多数不对）"，low 让 Flash 也在预算内收尾，两位教师在这批题上的行为更可比。E 让四个训练臂的题目完全相同，只是少几道题，引入的偏差比 C、D 小。

### 执行（本机运行，花 API 费用，由你决定何时启动）

```zsh
cd ~/projects/qwen2.5-7b-math-posttrain && source ~/.proxy_env.sh
# 1. 不发请求，列出每题已用的 v93 额度（"budget used 8/8" 之类）
"${P1_PYTHON:-python3}" -m experiments.qfft_v93.teacher fill --dry-run
# 2. 每题追加 8 次 low：上一步都是 used 8 时用 16，否则用"最大已用数 + 8"
"${P1_PYTHON:-python3}" -m experiments.qfft_v93.teacher fill --provider official --efforts low --max-attempts 16
# 3. 还剩几题
"${P1_PYTHON:-python3}" -m experiments.qfft_v93.data_build status
# 4a. 剩 0 题时
"${P1_PYTHON:-python3}" -m experiments.qfft_v93.data_build build --ds-verified
# 4b. 还有剩余时：成对剔除
"${P1_PYTHON:-python3}" -m experiments.qfft_v93.data_build build --ds-verified --paired-exclusion
```

- 用官方 API（`--provider official`），因为 low 档的映射写在官方文档里；OpenCode Go 是否原样转发 low 没有核实。官方 Key 来自 `DEEPSEEK_API_KEY`、`~/.config/p1/deepseek.json` 或 Windows 上的旧位置（`check_local.sh` 会显示找到的来源，不显示内容）。
- 第 2 步以退出码 3 结束是正常的（还有题用完了额度），接着做第 3 步。中途断了重跑同一条命令即可续上。
- 预计 23 题 × 最多 8 次，4 个并发，一两个小时。
- 构建后读 `data/qfft1000_v93/teacher_report.md`。"DeepSeek 轨迹来源"一节列出各轮次、各 effort、各提示词的条数；成对剔除时还会列出被剔除的题。
- 如果以前用 `--allow-gaps` 生成过预览，现在的 `data/qfft1000_v93/` 就是预览构建：阶段 `d` 会拒收它，Drive 上的数据不动。第 4 步会覆盖它。

### 2.1 2026-11-10 19 题替换执行记录（客观数据）

- **背景与操作**：2026-11-10 用户指示不采用成对剔除，要求保留 1,000 题。对历史 19 道超出 32K 窗口的死锁题目（`0021, 0024, 0057, 0268, 0339, 0459, 0576, 0693, 0704, 0731, 0740, 0745, 0769, 0808, 0840, 0870, 0920, 0943, 0974`），从 `data/portable_release/project1_raw_paired.jsonl` 中筛选了同科目、R1 词数 $\ge 1400$ 且经 decontam 校验合格的候选，使用 OpenCode Go 接口（DeepSeek-V4.1-Flash，参数 `effort=high`, `prompt=v91_anti_redundancy`）进行了流式生成与验证。
- **19 题生成实测**：
  - 19 道替换题经 Math-Verify 答案判定状态均为 `correct`；
  - 19 道题的 Flash 思考词数分布：极小值 544 词，极大值 1,789 词，均值 976.9 词，中位数 891 词；
  - 19 道题的对应 R1 思考词数分布：极小值 1,644 词，极大值 2,346 词，均值 1,913.4 词；
  - 详细逐题映射、原始题号、题目文本与双模型词数记录保存在 `scratch/final_hard_19_replacements.json`。
- **1,000 题构建产物**：
  - 本机执行 `data_build build --ds-verified`，在 `data/qfft1000_v93/` 下生成了 `r1/train.jsonl`（1000条）、`ds/train.jsonl`（1000条）、`ds_verified/train.jsonl`（1000条）、`manifest.json` 与 `teacher_report.md`；
  - `data_build status` 刷新 `data/qfft1000_v93/ds_status.json`，记录 `gaps: 0`，`DS usable: 1000/1000 by round {'v66': 897, 'v93': 56, 'v91': 22, 'v86': 25}`。
- **`teacher_report.md` 记录的客观数据**：
  - 可核对题共 750 题：Flash 正确率 67.3%（505/750），R1 正确率 64.3%（482/750）；
  - 750 题对错分布：两者都对 428 题，仅 R1 对 54 题，仅 Flash 对 77 题，两者都错 191 题；
  - Response Token 统计：R1 均值 9636.0 / 中位 9332 / p90 16176 / 最大 26835；Flash 均值 10020.8 / 中位 6719 / p90 25141 / 最大 32508。
- **Token 长度分布客观统计（Flash）**：
  - `< 1,000 Token`：13 题（1.3%）；
  - `1,000 ~ 2,000 Token`：93 题（9.3%）；
  - `2,000 ~ 5,000 Token`：301 题（30.1%）；
  - `5,000 ~ 20,000 Token`：425 题（42.5%）；
  - `$\ge 20,000$ Token`：168 题（16.8%）；
  - 全部 106 道 $< 2,000$ Token 的双模型完整推导文本已导出至 `scratch/under_2k_audit_all.json`，供接手方审查其逻辑链是否完整。
- **状态与下一步命令**：
  - 本地环境自检 `check_local.sh` 运行输出为 `1 warnings (git status), no failures`；
  - 数据上传阶段为 `bash experiments/qfft_v93/colab/pipeline.sh d`；
  - 锚点评测阶段为 `bash experiments/qfft_v93/colab/pipeline.sh 2`；
  - 接手方可根据上述数据与文件自行评估，决定何时启动后续阶段。

## 3. 本机 Kiro IDE 初始化清单（zsh）

```zsh
source ~/.proxy_env.sh
cd ~/projects/qwen2.5-7b-math-posttrain
git checkout main && git pull --ff-only
git fetch origin colab-reports
bash experiments/qfft_v93/colab/setup_wsl.sh   # 已装的 colab CLI 保留；检查 rclone 配置和 GitHub token 文件
bash experiments/qfft_v93/check_local.sh        # 只读自检，内容见下表；加 --no-data 跳过 data_build status
```

`check_local.sh` 不安装、不发 API 请求；只有 `git fetch` 会更新 `origin/main`、`origin/colab-reports`，`data_build status` 会刷新 `data/qfft1000_v93/ds_status.json`。

| 检查 | 期望 |
|---|---|
| 代理 | 新终端里已 `source ~/.proxy_env.sh`，GitHub 可达 |
| colab CLI | 版本 `>=0.7.5,<0.8`，已登录（有 `~/.config/colab-cli/token.json`） |
| `~/.config/p1/gh_credentials` | 一行 `https://XDE10:TOKEN@github.com`，权限 600 |
| `~/.config/rclone/rclone.conf` | 有 `[mydrive]` 段，权限 600 |
| Drive（本机装了 rclone 时） | `.project1_credentials/swanlab.json`、4 个模型目录、GRPO 题池文件 |
| `P1_PYTHON`（默认 `python3`） | 有 pandas、pyarrow、tokenizers、math-verify |
| `data_build status` | DS 可用数和缺口数 |
| `data/qfft1000_v93/manifest.json` | 不是预览构建 |
| `origin/colab-reports` | 最近一次 VM 会话及其结果 |

`P1_PYTHON` 指定本地数据用的解释器，例如 `export P1_PYTHON=$HOME/venvs/p1/bin/python`（写进 `~/.zshrc` 长期生效）。`local_data.sh`、第 2 节的命令、`check_local.sh` 都用它。凭据文件的格式和位置见 [templates/README.md](templates/README.md)，真实凭据不进 git。

## 4. 下一步（按顺序）

1. **现在就能跑、不依赖数据：阶段 2 锚点评测**，顺便测出正式评测的吞吐：`bash experiments/qfft_v93/colab/pipeline.sh 2`（G4 优先，分不到用 H100）。
2. 23 题：第 2 节的命令（本机，API）。
3. 上传数据：`bash experiments/qfft_v93/colab/pipeline.sh d`（CPU VM，核对每个 `train.jsonl` 的 SHA-256；预览构建会被拒收）。
4. 阶段 3：`bash experiments/qfft_v93/colab/pipeline.sh 3`。它依次开两台 VM，先 `train:qfft_ds` 加 `eval:run:qfft_ds_v93:full`，再 `train:sft_ds` 加 `eval:run:sft_ds_v93:full`。想一次只跑一个：

   ```zsh
   bash experiments/qfft_v93/colab/launch.sh --gpu G4,H100 train:qfft_ds eval:run:qfft_ds_v93:full
   bash experiments/qfft_v93/colab/launch.sh --gpu G4,H100 train:sft_ds eval:run:sft_ds_v93:full
   ```

   Ctrl+C 或关终端只是脱离，任务继续；`bash experiments/qfft_v93/colab/launch.sh --attach 会话名` 重新跟上；`bash experiments/qfft_v93/colab/launch.sh --gpu cpu status` 看所有阶段的进度。
5. 第一个训练开跑后，在 SwanLab 上实时看 `time/seconds_per_step`，或等约 30 分钟后的第一次推送：`git fetch origin colab-reports && git show origin/colab-reports:v93/latest/status.md`。按第 5 节的公式重算耗时和计算单元，再定后面阶段的顺序。

## 5. 耗时与计算单元（估算，以第一个训练的 s/step 为准）

训练耗时 ≈ 378 步 × s/step + 约 35 分钟（3 次本地保存 19 分钟，导出并上传 14 GiB 约 10 分钟，启动约 5 分钟）+ 等镜像的时间（两次保存的间隔短于一份镜像的上传时间时，训练会在下一次保存前等它传完；按 29 MiB/s 一份约 67 分钟）。1000 题 ÷ 16 = 63 步/epoch，6 epoch 共 378 步，成对剔除几道题不改变步数。

| run | 每步 token | s/step | 训练 | 同一 VM 上的评测 | G4 计算单元（8.71/h） |
|---|---|---|---|---|---|
| `qfft_ds`、`sft_ds` | 约 7 万（16 × ~4.4K） | 20–70 | 2.7–7.9 h | 1–2.5 h | 每个约 33–90 |
| `qfft_r1`、`sft_r1` | 约 16 万（16 × ~10K） | 35–100 | 4.3–11 h | 1–2.5 h | 每个约 46–120 |

- 下限来自冒烟：真实数据每步的计算量只有冒烟的约 13%（DS）到 30%（R1）。每个微批的计算变短以后，ZeRO-2 offload 每个微批把梯度搬到 CPU 的开销会露出来；冒烟里这部分被 32K 的长计算盖住了，测不出来。上限参考 RomEscape 在同型号卡上的同配方（16K、batch 8）：35,174 s / 750 步 ≈ 47 s/步。
- 阶段 3 两个 run 合计约 7–21 h、65–180 计算单元。阶段 2 约 3–6 h（base 和 Qwen2.5-Math 输出短，两个长链模型各 1–2.5 h）。
- 每次开跑前用 `colab usage` 看余额。如果第一个 run 的 s/step 接近上限，先保证阶段 3、4（主结果：SFT/QFFT → GRPO），阶段 5（R1 对照，RQ2）放到额度允许的时候。
- 如果每步慢的主因是每个微批的梯度卸载，可以考虑按长度打包减少微批数。它改变损失在样本之间的加权，属于配方改动，要单独决定。

## 6. 执行约束与已知坑

- 交互 shell 是 zsh，仓库脚本是 bash，一律 `bash 脚本`。多步、带嵌套引号或 heredoc 的操作写成 `.sh` 再执行，不在 PowerShell/cmd 里拼长命令。
- 新终端先 `source ~/.proxy_env.sh`。
- `launch.sh` 的挂载步骤原来用 `MOUNT_OUT="$(colab exec ... --timeout 1500 ...)"` 整段捕获输出，网络慢时最多 25 分钟没有任何输出。你本机已改成 `tee` 实时输出、300 s 上限，提交后以仓库里的 `launch.sh` 为准；这次的 PR 没有动 `launch.sh`，避免冲突。注意 VM 端挂载脚本自己的时限还是 apt-get install 600 s、apt-get update 300 s、重试 install 600 s、rclone mount 120 s，最后再等挂载点 60 s：新 VM 上 apt 慢时客户端的 300 s 先到期，launch.sh 判定挂载失败并释放 VM，结果是多开一台 VM，不会卡住。建议在同一个提交里把这些时限加起来压到 300 s 以内（例如 90、60、90、30、20 s）。
- Google Drive 每个账号每 24 小时约 750 GiB 上传上限（[rclone 文档](https://rclone.org/drive/)，`--drive-stop-on-upload-limit` 一节）。本次起全参每 2 个 epoch 存一次（第 126、252、378 步），每个 run 上传约 241 GiB（两份 113.5 GiB 镜像加 14 GiB 导出），所以 24 小时内最多 2 个全参 run：阶段 3 和阶段 5 不要连着跑。撞上限时 rclone 立即报错，镜像失败只记警告、训练继续；如果撞在最后的导出上传上，训练阶段会失败，第二天重跑从第 252 步的镜像续训再导出。
- `--keep` 时 `/content/gh_credentials` 留在 VM 上，直到 `colab stop -s 会话名`。
- 另一台 VM 的心跳在 15 分钟内时，同一阶段会被拒绝；确认原 VM 已经不在了才用 `--force`。
- Claude 不启动 GPU 任务、不发付费 API 请求；它准备命令、读 `colab-reports`，代码改动按阶段建分支、小步提交、PR 合入 main。

## 7. 不在 git 里、只在本机的东西

- 凭据：`~/.config/p1/gh_credentials`、`~/.config/rclone/rclone.conf`、`~/.config/p1/deepseek.json`（或 Windows 上的旧位置）、`~/.proxy_env.sh`，以及 Drive 上的 `.project1_credentials/swanlab.json`。格式见 [templates/](templates/README.md)。
- 本地数据：`data/`（评测集、计划、训练集）、`artifacts/qfft_teacher_v93/attempts/`（DeepSeek 的全部原始返回，是数据来源的证据）、`reports/colab_logs/`。阶段 `d` 只上传 `data/qfft1000_v93` 和 `data/benchmarks_v93`。建议把 `artifacts/qfft_teacher_v93/` 也备份到 Drive；本机装了 rclone 时：`rclone copy artifacts/qfft_teacher_v93 mydrive:projects/project1/v93/local_backup/qfft_teacher_v93`。

## 8. 测试

`bash tests/v93_harness/run_all.sh` 用模拟的 Colab CLI、rclone、GitHub 和 DeepSeek API 跑 8 组检查：launcher 场景 96、Drive 传输 28、冒烟总结 9、teacher 28、固定输入 5、data_build 15、setup 18、check_local 15，共 214 项。它在工作区的临时快照上运行，不碰 `data/` 和 `reports/`。只跑一部分：`bash tests/v93_harness/run_all.sh drive teacher data`。说明见 [tests/v93_harness/README.md](../../tests/v93_harness/README.md)。`pytest -q` 是另一套（`tests/` 下 25 个文件），缺本地数据时个别用例会失败或报错。
