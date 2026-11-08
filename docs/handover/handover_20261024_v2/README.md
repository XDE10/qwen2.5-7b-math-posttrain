# 接手 v2：加固、去重、预登记与 Colab 化

日期：2026-10-24。代码/数据根目录仍是 `D:\Users\Mr.Z\Desktop\项目一_7B数学后训练`。本轮全部新实现用v2命名，旧源码、根README/配置及所有发布清单保留。后续Linux入口见 [Colab bash说明](colab/README_bash.md)。

## 已实际完成与未验证边界

已在用户创建的 **genai-test / Python3.12.15** 按原CPU快照安装；Transformers保持4.57.6，TRL1.16.0，PyTorch2.14.1。`pip check` 输出No broken requirements found，完整freeze见 [pip_freeze.txt](pip_freeze.txt)。原README的CPU检查为 **17 passed**；最终原测试+新增测试合并为 **69 passed、0 failed、3 warnings**，见 [pytest_final_69.log](pytest_final_69.log)。其中新增52项，涉及准入、身份、错误分类、预算、完整编码、reference、真实TRL损失/梯度、配置落盘/拒绝覆盖及CPU入口防加载。3项warning来自tiny GPT2的PEFT Conv1D方向适配和无远程config的本地adapter fixture，不是GPU验收记录。

原 `src/training/approved_hf.py:42` 的encode_sft本轮在4.57.6环境对两臂各1500条直接成功，未借Transformers5包装器才通过；真实tokenizer返回list。最长原生3876、精炼1443，prefix完整、EOS在监督区，最长prompt1248。证据：[token_audit.json](token_audit.json) 的entry_encode_sft与tokenizer返回类型；新增测试也直接调用原入口。

**没有下载任何模型权重、付费API调用、GPU训练或实际7B推理。** tiny CPU测试有合成小模型的优化器更新；Colab/Linux CUDA安装、Drive交互、A100显存与吞吐、真实7B reference和恢复均为 **未验证**。embedding近邻也未验证：本地MiniLM/bge-small缓存有配置和tokenizer，但没有权重，已在去重报告列出；没有联网补模型。

## 最小加固的实际位置与行为

1. **文件批准后仍检查split/dev/配对。** `src/data/approved_training_inputs_v2.py:10,17,33`：继承旧白名单/文件SHA/逐行批准检查，要求split=train，校验保护表SHA并拒绝sample_id交集；raw/compact的pair_id必须唯一、集合一致，题面/答案/题型/split/sample_id和arm一致。配对顺序不要求完全相同，训练时按ID配对，当前数据顺序本来也一致。校验失败立即拒绝，不重写批准。
2. **RL长度显式落地。** `src/training/approved_hf_v2.py:114-120` 把prompt=1752、completion=2048、total=3800显式传入v2配置。实际TRL1.16的GRPOConfig没有max_prompt_length，所以 `src/training/trl_bridge_v2.py:11` 派生配置保存并校验这两个项目字段，`21` 的trainer适配层在生成前拒绝超长prompt，在生成后/策略loss前打印真实prompt/completion mask有效长度、检查后端长度和全空mask。它不截断题面，也没有重写TRL loss。旧RL配置的max_length原本完全没接入；v2明确将它解释为prompt+completion总预算，不能与SFT的单序列max_length混用。
3. **奖励日志与资源错误。** `src/training/math_reward_v2.py:13,26,40` 区分equivalent、mismatch、unsupported_relation、multiple_boxes、unclosed_box、parse_failure、timeout、verifier_error。JSONL逐批保留精确计数/worker结果/step；Trainer指标同步分类计数（其日志聚合可能是多批均值，累计审计以JSONL为准）。进程失败、坏JSON、超时先记日志再中止，不变成0分答错。当前unsupported/格式失败仍是保守0奖励，**只加分类不代表关系式评分覆盖已修复**；它们不计入mismatch类别，正式RL池筛选必须使用冻结后的支持范围统计。
4. **恢复前拒绝缺项。** `src/training/guards_v2.py:39` 要求optimizer.pt、scheduler.pt、rng_state.pth、trainer_state.json和run_identity.json均存在且非空、global_step>0、身份一致，然后框架才加载状态。单卡范围明确；未声称支持ZeRO/FSDP的其他状态布局。检查存在性不替代反序列化与同horizon恢复对照，真实7B恢复未验证。
5. **绑定实际身份。** `guards_v2.py:51,86` 检查merged属于声明arm、SFT数据SHA、SFT输出路径和v2 schema，并固定merged_sft_identity与weight_manifest摘要。清单逐文件校验且拒绝路径逃逸。run_identity新增tokenizer文件、chat_template、worker及.pilot_deps评分源码SHA、merged清单摘要和更多库版本；与resume配置比较时仅忽略resume路径。Drive materialize脚本只写新配置，把实际SFT导出摘要填入RL配置，不能造占位身份绕过检查。
6. **reference脚本。** `scripts/verify_sft_reference_v2.py:12` 比较固定prefix上(a)基座、(b)新LoRA禁用、(c)已更新adapter再禁用的logits，同时检查冻结基座不变和启用adapter时有变化。tiny实测两次禁用最大差均0、激活adapter变化约2.66e-4，见 [reference_tiny.json](reference_tiny.json)。真实路径只加载已有本地RL adapter并核对run身份，不做7B优化；使用CPU分块摘要避免GPU复制整份基座。BF16真实容差/行为未验证，tiny的零差不能替代它。

v2训练命令是 `python -m scripts.train_project1_sft_v2 --config ...` / `python -m scripts.train_project1_rl_v2 --config ...`，本轮未执行。新schema不追认旧v1 SFT导出；旧产物保留，后续需按新版本另行验收。

## TRL loss 对照得到的具体结论

`tests/test_trl_loss_alignment_v2.py` 真正调用安装的TRL `_generate_and_score_completions` 和 `_compute_loss`，使用tiny GPT2 logits而非只复述参考公式。覆盖GRPO/Dr.GRPO × 截断开关 × beta=0/0.04，共8种情况，核对advantage、逐位置mask、scalar loss和参数梯度，并使用旧logp偏移触发ratio裁剪。比较使用FP32、atol1e-6/rtol1e-5；真实GPU fused kernel仍未验证。

- TRL组内标准差为样本标准差，缩放epsilon是 **1e-4**；历史reference.py默认1e-8仍保留，测试显式传1e-4对齐。
- `mask_truncated_completions=True` 会清空截断行的attention/loss mask；这行仍占batch归一化分母，奖励仍参与组均值/标准差。不能说mask后它对所有统计完全没有影响。
- GRPO逐回答有效token平均，Dr.GRPO固定除以max_completion_length；完整截断行是零贡献而非从batch删除。`reference_v2.py` 调用旧policy_loss后按实际batch比例补齐这个语义。
- TRL默认开启 **use_bias_correction_kl=True**，KL项还乘重要性比率；v2入口显式固定它。初轮4项失败正是独立参考漏掉该项，已修复；没有放宽容差掩盖差异。梯度累积的loss分母也纳入比较。

## 去重结果与解释

完整命中清单：[decontamination_20261024.json](decontamination_20261024.json)，并按要求提供根reports同名副本。输入包括SFT1500、RL4810、dev256、MATH-500和GSM8K-test1319，均本地存在，没有补下载对照集。

用训练侧50对真实raw/compact同题、各50条受控改写和数值替换家族、50条无关对做校准，选择字符5gram Jaccard阈值0.50：原题/改写召回100%，数值替换家族94%，合计98%，无关小样本误报0%。这些不是独立标注的自然改写召回，也不保证全量误报率为0；数值替换可能是不同题，但属于应复核的题族信号。阈值在扫描留出集前由训练侧校准固定。

- SFT→dev/MATH-500/GSM8K：规范化精确与阈值Jaccard均0命中。
- RL→dev：67对近似命中；RL→MATH-500：131对；RL→GSM8K：0。前两项共涉及172个唯一候选ID；规范化精确命中均为0。它们是人工复核清单，不能把198对全称已证实泄漏。
- SFT→RL：8对近似命中，其中 **2对规范化精确重叠**。训练集之间是否允许重叠须预登记，不能混称留出污染。
- 本轮没有删题或改配额，六类仍为代数525、几何300、数论300、组合225、不等式100、逻辑谜题50。当前SFT没有留出命中，因此本轮报告的假设留出排除影响为0对；若另行决定禁止所有SFT/RL近似重叠，需复核涉及的代数3、几何4、组合1题，成对处理raw/compact共16行，另立版本审批。

embedding未运行意味着本轮不能认证语义去重完成；低Jaccard也不证明无同构或预训练污染。

## 两套正式草稿与候选筛选计划

见 [对照预登记](对照预登记.md)、[预算模拟](formal_v1/budget_totals.json) 和 `formal_v1/equal_exposure/`、`formal_v1/equal_supervised_tokens/`。两套都列出E0/E1/E2/E3/E4、raw/compact共享关系、起点、数据、评分器、seed和生成参数；正式RL数据均明确待另行批准。等监督token的计数器/采样停止运行时实现尚未完成，草稿不能直接训练。

等曝光：各1 epoch、计划94更新；等token：原生94更新，精炼按声明顺序模拟917更新/9.7573 epoch，超额831 token。后者增加题目重复与更新次数，不能单独解释成长度因果。E3/E4必须共享同臂同一正式E1，E2从E0直接RL，不能用当前SFT后smoke替代。

[RL候选筛选预案](RL候选筛选预案.md) 包含4810×8=38,480次采样、k∈2～6的经验通过率门槛、最多3000题、无支持/资源错误如何排除统计、逐题seed与分片续跑。以平均1024 output tokens、假设400/800/1600 tokens/s估算纯生成约27.36/13.68/6.84小时，均为参数化估算，非实测。候选筛选、vLLM和正式RL池本轮未运行或批准。

## Colab化与下一步

[Colab README](colab/README_bash.md)、[bootstrap脚本](colab/colab_bootstrap.sh)、[notebook](colab/bootstrap_colab.ipynb) 和 [Linux CUDA固定主要版本候选](colab/requirements-linux-cu128.txt) 已生成。bash语法检查通过；Drive挂载须通过notebook Python前端交互，sh检测未挂载时给出明确提示，不假装挂载成功。安装后要求重启，保存pip freeze/check和torch collect_env。

官方cu128索引本轮可见最高2.11.0+cu128，未列出本地快照2.14.1+cu128，因此CUDA清单选2.11.0+cu128，**组合未验证，不能称CPU快照原样转成CUDA锁**。Colab初次安装验收后应再冻结完整Linux传递依赖。run_identity严格绑定库版本；镜像更新后先恢复同一固定版本、重启、比对旧identity，再恢复checkpoint，不能改JSON绕过拒绝。

Drive建议预留80～100GB；两臂merged约30GB、初始缓存约15GB，checkpoint/日志/临时导出另计。A10040GB建议保留BF16+SDPA、microbatch1、gradient checkpointing、LoRA rank16，raw最长3876的约20～32GiB规划值未实测，RL另行验收。详细公式与恢复命令均在Colab README。

下一阶段优先：复核172个RL留出近似命中；补齐任务感知评分并冻结支持域；实现/验证等token停止机制；在目标Colab环境完成最长batch→reference→有效组奖励→同horizon恢复。之后才能冻结正式E0～E4，筛选结果仍须另行授权成为训练池。

文件变化、初轮失败原因与未验证事项另见 [CHANGELOG](CHANGELOG.md)。
