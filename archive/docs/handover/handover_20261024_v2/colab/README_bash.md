# Colab / Linux 接手入口 v2

状态：**Colab、Linux CUDA 安装、A100 40GB/80GB、真实7B均未验证**。本地 genai-test 的CPU验收不替代这些检查。旧根README/配置保留；本文件是新增 bash 版入口。

## 1. 文件与环境

在Colab界面选择A100；具体显存以 `nvidia-smi` 为准，不保证获得40GB或80GB。项目代码复制到 `/content/project1`；携带现有tokenizer、批准release、math留出集、src/scripts/tests和.pilot_deps。不要解压大审核缓存，不引用旧Windows路径。

先在 **notebook Python单元** 挂载Drive，再调用shell。挂载需要Colab前端授权，不能保证在独立shell进程里交互成功，所以提供了 `bootstrap_colab.ipynb`，没有假装sh本身能完成登录：

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['PROJECT_DRIVE_ROOT'] = '/content/drive/MyDrive/math7b_v2'
os.environ['HF_HOME'] = '/content/drive/MyDrive/math7b_v2/hf_cache'
os.environ['PATH'] = '/content/project1-env/bin:' + os.environ['PATH']
```

```bash
cd /content/project1
bash reports/handover_20261024_v2/colab/colab_bootstrap.sh
```

安装完成后**重启运行时**，重新运行Drive挂载与环境变量单元，再运行下方验收。训练依赖安装在 `/content/project1-env` 独立venv（可用PROJECT_VENV_DIR配置），避免混入Colab预装训练栈；Drive挂载仍在notebook原Python内完成。notebook的Python内核不会自动变成venv，模型/测试命令应使用下方激活后的shell或PATH中的venv Python；若改变venv路径也要同步PATH。安装脚本不下载模型、不启动训练。PyTorch候选为官方cu128索引可见的2.11.0+cu128，其他主要版本与本地快照相同；本轮没有安装或验证这套Linux组合。官方索引未列出快照的2.14.1+cu128，不能捏造同版本CUDA锁。依赖清单只固定主要版本，实际安装后的pip-freeze-linux才记录传递依赖；第一次成功验收后另存完整Linux锁。

```bash
cd /content/project1
source /content/project1-env/bin/activate
python -m pip check
python -m torch.utils.collect_env
python tools/verify_migration.py
python -m pytest tests/test_approved_training_inputs.py tests/test_training_entries_cpu.py tests/test_losses_and_masks.py tests/test_hardening_v2.py tests/test_trl_loss_alignment_v2.py -q
python -m scripts.verify_sft_reference_v2 --tiny --output "$PROJECT_DRIVE_ROOT/environment/reference_tiny.json"
```

tiny脚本默认CPU。真实7B权重须在后续明确进入GPU验收时另行准备；本轮没有下载。初次测试通过也不代表真实7B已通过。

## 2. Drive输出与v2配置

v2工程模板在本目录 `configs/`，所有路径用Linux相对路径或 `${PROJECT_DRIVE_ROOT}`。v2运行器会解析环境变量；未设置时拒绝。checkpoint、final、merged、reward_status.jsonl与Trainer日志均在该Drive输出目录。代码建议放本地 `/content`，持久产物留Drive。

以下命令只用于后续已授权的GPU验收，本轮未执行：

```bash
mkdir -p "$PROJECT_DRIVE_ROOT/configs"
python -m scripts.materialize_colab_config_v2 --template reports/handover_20261024_v2/colab/configs/project1_raw_sft_smoke_v2.json --output "$PROJECT_DRIVE_ROOT/configs/raw_sft.json"
python -m scripts.train_project1_sft_v2 --config "$PROJECT_DRIVE_ROOT/configs/raw_sft.json"
```

SFT导出成功后才可固定RL起点。materialize脚本只读取实际merged身份并写新配置，不改release或历史配置：

```bash
python -m scripts.materialize_colab_config_v2 --template reports/handover_20261024_v2/colab/configs/project1_raw_grpo_smoke_v2.json --output "$PROJECT_DRIVE_ROOT/configs/raw_grpo.json"
python -m scripts.train_project1_rl_v2 --config "$PROJECT_DRIVE_ROOT/configs/raw_grpo.json"
python -m scripts.verify_sft_reference_v2 --config "$PROJECT_DRIVE_ROOT/configs/raw_grpo.json" --rl-adapter "$PROJECT_DRIVE_ROOT/runs/v2_raw_grpo_smoke/final" --device cuda --output "$PROJECT_DRIVE_ROOT/environment/reference_raw_7b.json"
```

compact和Dr.GRPO使用相应模板，不能错用另一臂merged。真实reference脚本读取已有RL更新后的adapter，核对run身份；脚本本身不执行7B优化，也不额外在GPU克隆整份冻结基座。tiny路径才做合成CPU更新。正式draft被执行器拒绝，不能把 `draft` 改为false就认为已完成冻结与数据审批。RL smoke仍复用已批准配对文件；4810候选及筛选结果均未获新增训练批准。

## 3. 首次GPU通过标准

- longest SFT batch覆盖原生3876、精炼1443 tokens，打印实际input/label/mask；无静默截断、NaN或OOM，只有LoRA更新。
- logits reference脚本对固定prefix比对SFT基座、禁用新RL LoRA、更新RL LoRA后再禁用；浮点容差与差值保存在结果。tiny通过不替代真实BF16容差验收。
- RL每题4条completion的有效长度、EOS、截断mask、奖励状态均留日志；至少存在有效非全同奖励组，所有completion被mask直接拒绝。
- 同一max_steps与scheduler horizon，step1中断后从完整checkpoint恢复到step2；核对模型、optimizer、scheduler、RNG、数据游标和step。不得用只加载adapter替代恢复。

## 4. 会话中断与版本绑定

新Colab会话先挂载Drive，按同一Linux固定清单重新安装并重启，读取旧run_identity.json的versions逐项比对，再用 `pip-freeze-linux.txt` 还原传递依赖。本地CPU版本与Colab CUDA版本可以不同；同一run恢复前后必须一致。镜像更新后若库版本不符，v2会拒绝恢复；不要编辑身份JSON“修好”它。无法恢复旧版本时，新开run并声明它是新环境的继续研究，不能称严格续跑。

Drive可能在断开时只同步了部分文件。选择已有 `run_identity.json`、`optimizer.pt`、`scheduler.pt`、`rng_state.pth`、`trainer_state.json`及adapter权重均完整的最近checkpoint；v2先检查缺项，框架再反序列化。缺项选更早完整检查点，不能造空状态文件。

```bash
python -m scripts.materialize_colab_config_v2 --template "$PROJECT_DRIVE_ROOT/configs/raw_sft.json" --output "$PROJECT_DRIVE_ROOT/configs/raw_sft_resume.json" --resume "$PROJECT_DRIVE_ROOT/runs/v2_raw_sft_smoke/checkpoint-1"
python -m scripts.train_project1_sft_v2 --config "$PROJECT_DRIVE_ROOT/configs/raw_sft_resume.json"
```

output_dir、max_steps、其他配置和代码必须保持不变，仅resume路径从identity中排除。后续代码升级也会因runner SHA改变而拒绝旧run的严格续跑；请保存本次版本代码。Drive写入不是原子快照，本轮没有实测会话中断或Drive同步。

## 5. 磁盘与A100 40GB估算（均未验证）

BF16 7B merged约15GB；两臂约30GB，初始模型缓存再约15GB。rank16七投影LoRA约40.4M参数；adapter、梯度和Adam状态远小于全量模型，但每个完整checkpoint仍可约0.5～0.8GB。若6个smoke各保留2份checkpoint约6～10GB，再加final、环境、预测及重试，建议Drive至少80～100GB空闲。不要把一份15GB估计误认为全部项目磁盘需求；导出瞬时副本和HF缓存也占空间。

最长SFT L=3876、B=1：基座BF16约14～15GiB，LoRA及优化器约0.5～1GiB；单个 `[1,3876,152064]` logits约1.10GiB BF16 / 2.20GiB FP32，loss临时张量可能再增加；28层残差激活的简单下界约0.72GiB，MLP中间激活、反向、SDPA临时区和allocator会使实际占用高得多。保守规划约20～32GiB，非实测上界。

**40GB建议保持gradient_checkpointing=True、use_reentrant=False、microbatch=1、BF16、SDPA、LoRA rank16，关闭vLLM共置**。当前v2已这样设置，不能据估算关掉checkpointing；80GB也先按同配置完成验收。梯度累积只改变有效batch，不降低单条最长样本峰值。RL四条rollout、logprob/KL和KV cache有额外占用，必须单独验收，不能从SFT估算推出RL一定可跑。若OOM，先查真实allocated/reserved/进程占用，另立版本配置；不截断封存SFT答案，也不悄悄改量化策略。

参考：[PyTorch官方安装版本](https://pytorch.org/get-started/previous-versions/)、[固定revision模型配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/config.json)、[vLLM引擎参数](https://docs.vllm.ai/en/latest/configuration/engine_args.html)。本轮只查索引/文档，未运行Linux安装或性能基准。
