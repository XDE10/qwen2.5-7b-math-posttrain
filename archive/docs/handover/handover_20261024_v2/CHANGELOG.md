# CHANGELOG — handover v2 — 2026-10-24

## 保留边界

没有修改任何旧版本训练器、根配置、README、发布数据、manifest、批准字段或已有SHA。新增代码均使用 `_v2` 文件名；旧v1与历史报告保留。临时合成测试fixture的批准/摘要只用于拒绝路径测试，不接触真实发布文件。

主要输出在本目录；另按用户指定路径新增 `reports/decontamination_20261024.json`，与本目录同名完整报告逐字节相同。此兼容副本不进入训练批准或迁移原清单。

## 环境与实际执行

- 在现有 `D:/anaconda3/envs/genai-test/python.exe`（Python3.12.15）按未改写的requirements-cpu-snapshot安装；Transformers固定4.57.6，无5.x升级。保存pip_install.log、pip_freeze.txt、pip_check.log。
- 原README指定CPU三文件本轮实测17 passed，见pytest_baseline.log。
- 原encode_sft对两臂全量实际编码均通过，见token_audit.json；此环境返回list，不需要v1交接诊断中的API兼容包装。
- v2初轮36 passed/4 failed：4项定位为独立KL参考漏掉TRL默认bias correction；已在reference_v2补齐，保留初轮日志，修正后40 passed。随后增加Drive、身份、全量编码、预算、进程失败、空mask及已有adapter测试；最终结果见pytest_final.log。
- 最后增加配置落盘/禁止覆盖、formal materialize拒绝与CPU入口不加载模型检查，最终合并 **69 passed / 0 failed / 3 warnings**，见pytest_final_69.log；共新增52项。pytest_final.log保留前一版66项通过记录。
- 未下载任何模型权重、调用付费API、启动GPU训练或执行候选采样。tiny CPU测试生成临时小模型并执行必要的优化器检查，不代表7B训练。

## 新增代码

- `src/data/approved_training_inputs_v2.py`：继承v1文件白名单/SHA检查，新增train split、保护表SHA与ID交集、raw/compact唯一pair和题目身份检查。
- `src/training/guards_v2.py`：Drive变量解析、完整checkpoint状态拒绝、SFT臂/数据/输出/精确身份校验、merged清单固定、tokenizer/template/评分器及依赖源码身份。
- `src/training/math_reward_v2.py`：8类状态计数，JSONL/Trainer日志；超时/基础设施错误先记日志再中止，未归为答错。
- `src/training/trl_bridge_v2.py`：TRL1.16没有原生max_prompt_length，新增派生配置显式保存prompt/completion/总预算并在生成前后执行检查；loss仍使用TRL。
- `src/training/approved_hf_v2.py` 与两个 `scripts/train_project1_*_v2.py`：保留旧数学SFT结构，接入上述最小防线。新配置需要显式arm、预算与对应SFT来源；本轮未运行7B入口。
- `src/losses/reference_v2.py`：复用reference.py，适配全截断行留在batch分母、累积和KL bias correction，不改历史参考算子。
- `scripts/verify_sft_reference_v2.py`：固定prefix比对基座/新adapter禁用/更新adapter禁用；tiny路径合成CPU更新，真实路径只加载已有RL adapter，不执行7B优化；不在GPU额外克隆整份基座。
- `src/data/decontamination_v2.py`、`scripts/decontaminate_project1_v2.py`：规范化精确、字符5gram Jaccard与无漏候选prefix筛选；只读命中清单。embedding缓存只有metadata/tokenizer，无权重，未运行第三层。
- `scripts/build_project1_drafts_v2.py`：真实编码计数、两种预算模拟、E0～E4矩阵、Drive smoke模板；正式draft拒绝执行。
- `scripts/materialize_colab_config_v2.py`：只写新配置、解析Drive变量并从真实SFT导出固定身份和权重清单；拒绝formal draft。
- `tests/test_hardening_v2.py`、`tests/test_trl_loss_alignment_v2.py`：新增针对性拒绝、数值、梯度、全量数据、预算与reference测试。

## 文档与配置

- `formal_v1/` 两套草稿、精确预算模拟和当前输入身份，状态统一为“草稿，未冻结”。
- `对照预登记.md`：等曝光/等监督token的解释边界、完整矩阵与冻结条件。
- `RL候选筛选预案.md`：38,480次生成设计、支持范围与拒绝比例统计、参数化时间/显存估算、另行授权门槛。
- `colab/requirements-linux-cu128.txt`、`colab_bootstrap.sh`、notebook、README_bash.md及6个模板。Colab训练栈放独立venv，Drive仍在notebook原内核挂载，CLI通过venv Python执行。Linux/CUDA组合、Drive挂载、A100显存、真实7B reference均未验证；bash只做语法检查。

## 保留问题与未验证

关系式评分覆盖没有被本轮日志加固自动修复；RL池仍未批准。等监督token的真实停止计数器与声明sampler尚未实现。Colab依赖是固定主要版本候选，不是完整已验收Linux锁；本地torch2.14.1与CUDA候选torch2.11.0+cu128分别记录，不能混作同一环境。去重相似命中待人工/语义复核，不能直接删题或保证零污染。
