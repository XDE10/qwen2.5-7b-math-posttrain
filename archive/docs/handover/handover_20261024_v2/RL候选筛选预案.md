# E0 4810题 × 8次筛选预案

状态：**设计草稿，未冻结；未生成、未筛选、未批准训练**。唯一输入仍是 `project1_rl_priority_not_rollout_selected.jsonl`，不能将计划输出加入训练白名单。

## 1. 评分先于采样冻结

先固定 scorer源码/依赖/测试、答案抽取、题型与定义域支持表、EOS/截断规则，再冻结采样协议。现有strict-v3会拒绝关系式，不能把该拒绝作为0分参与经验通过率；仅调低拒绝率或跳过难题也不能称无偏。

需在train/dev侧建立关系式评分支持：在题目定义域明确时比较精确解集，处理变量、端点、开闭、整数/实数域；函数族、多参数、未知域继续标unsupported，不能用“两个关系式都为真”判等价。不使用最终测试错题来修改规则。完成覆盖检查后才冻结新评分器版本。当前v2只增加分类审计，没有声称扩展了数学覆盖。

每题记录8个独立trial的 seed、原文、finish_reason、EOS、token数、scorer类别。只有 **8/8可评分** 才使用 `k/8`，其中 k∈{2,3,4,5,6} 满足25%～75%。解析失败/超时/关系式不支持不偷偷归为不等价；infra错误重试同一trial seed并保留原事件，不能重抽直到答对。多框/格式失败是否作为任务失败须先冻结，不能运行后临时选择。截断但含正确答案如何处置也须冻结，并单列截断比例。

若某题有不可评分trial，标记“通过率未验证”，不进入阈值筛选，单列 coverage。报告全4810题及分题型的 unsupported_relation / parse_failure / timeout / verifier_error 比例；同时报告每题完整可评分率、eligible率、scorable子集通过率和全池未知比例。只报告支持域内的条件通过率，不能宣称全池通过率无偏。若关系式存在系统拒绝，先完善任务感知评分再跑正式筛选，或明确缩小研究域并另行审批；不能称“原池难度筛选完成”。

最多保留3000题。超过时按预先冻结的分层配额与seed42抽取，记录未入选原因；不足3000则报告实数。输入SHA、去重报告、支持范围和sampling/scorer版本写入新的候选结果身份文件。**筛选产物仍需用户另行授权及独立数据版本审批，不能靠脚本写training_approved=true。**

## 2. 建议脚本接口与数据流

将来另行实现 `sample_rl_candidates_v1.py --protocol <冻结JSON> --local-model <本地E0快照> --output <Drive目录>`。本轮没有执行或下载vLLM；不在bootstrap训练环境安装它，避免其torch依赖改变训练锁。使用独立Colab会话/环境，固定vLLM、torch、CUDA，验收后锁完整pip freeze。

数据流：校验候选文件与保护ID/去重隔离清单 → 固定顺序渲染problem-only prompt → 分片（例如每50题） → 每题8次采样 → 原始JSONL持久化 → 独立有超时的CPU评分 → 只读eligibility清单。resume键为 `(candidate_sha,model_revision,prompt_sha,scorer_sha,sample_id,trial_index,seed)`，不按输出文件行数猜续跑位置。

vLLM草稿参数：BF16、tensor_parallel_size=1、max_model_len=3800、gpu_memory_utilization=0.80、max_num_seqs=32起步、enable_prefix_caching=True；采样 temperature=0.8/top_p=0.95/max_tokens=2048/n=8。先用训练侧32题诊断实际吞吐、上下文/EOS与显存，参数改动另立协议；不把这个小诊断混入冻结后的8次统计，除非其身份完全一致且预先声明复用。

每个trial显式分配seed（如通过SHA256(seed42,ID,trial)导出32位整数），独立trial请求能保证续跑身份；若以n=8批请求，则必须记录引擎seed映射和稳定性，不能保证更换batch顺序/硬件后逐token相同。提示模板与revision和训练主线相同，标准答案只交评分器。不要把参考答案、审核意见放入vLLM prompt。

## 3. 时间和显存：参数化估算，未验证

生成条数固定4810×8=38,480。若平均回答512/1024/2048 tokens，则分别约19.70M/39.40M/78.81M output tokens。令实测批量总吞吐为 R output tokens/s，生成时间约 `38480 × 平均回答长度 / R`，再加prefill、启动、I/O、评分及会话重启。

以平均1024为例：R=400/800/1600时，纯decode约27.36/13.68/6.84小时；这三个R只是敏感性假设，**不是A100实测吞吐**。平均2048时间再翻倍。以平均prompt300计，未复用时prefill约11.54M tokens；每题共享prefix能减少重复prefill，但不能把prefix caching当零成本保证。评分单次0.5～2秒、4路并行的假设另约1.34～5.34小时；实际长表达式、20秒超时与进程启动会显著改变它。必须先测32题，将tokens/s、评分成本和失败率代回公式。

显存估算：BF16 E0权重约15GB；Qwen28层、4个KV head、head_dim128，BF16 KV约 `2×28×4×128×2=57,344 bytes/token`。32条并发、每条3800 token最坏约6.49GiB KV，64条约12.99GiB，另加激活、CUDA graph和allocator。40GB上32并发+0.80显存比例是待验收起点；80GB可另测更高并发，不能沿用未实测吞吐。八条回答会占多个序列槽位，不是只占一个prompt的空间。

Colab资源与会话长度不保证，38,480次生成可能超过一次会话；按题/分片持续写Drive、完成后验证校验和再标记done，避免仅靠内存队列。模型推理与训练分开会话，避免vLLM与训练器共占显存或改动训练依赖。磁盘除模型缓存外，38,480条原文与token日志可能数GB，先估算每片实际大小。

参考：[vLLM官方引擎参数](https://docs.vllm.ai/en/latest/configuration/engine_args/)、[固定revision模型配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/config.json)、[Colab官方资源说明](https://research.google.com/colaboratory/faq.html)。估算由配置维度和显式假设推导，不是官方性能承诺。
