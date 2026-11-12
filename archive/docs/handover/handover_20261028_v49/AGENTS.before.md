# 项目一入口与持续协作约定

适用范围：本项目根目录及子目录。2026-10-26更新，整合前两次对话的实际进展。

## 本次对话后续更新（优先于下方历史状态）

- 2026-10-28 v46第三次安装报错已定位：新日志确认实际NumPy2.1.3可导入，上一轮NumPy修复已生效；现在是SwanLab依赖pwinput>=1.0.3仅发布sdist，被入口--only-binary=:all:误拦，并非训练/OOM。已仅给pwinput增加--no-binary=pwinput例外，其余包仍wheel-only，DeepSpeed仍独立安装。核实PyPI1.0.3只有4.4KB源码、无运行依赖；本地禁缓存真实构建出py3-none-any wheel成功，9项CPU测试通过，含pip真实FormatControl语义确认NumPy/SciPy仍拒绝源码。Drive同名入口读回字节一致，SHA256 d1f9f6a44359796cae68fd78028b21d13af7e56da18c0a6d7cfe887942f184ae。用户同命令重试，不删server、不重装核心栈、不改Key。远端SwanLab/DeepSpeed安装和8步GPU仍待用户实跑验证；交接总结仍待继续。

- 2026-10-28 用户附完整日志确认v46第二次安装失败：新wheel-only阶段被旧约束numpy==1.26.4拦下，Python3.13无匹配wheel；仍未进入训练。最新同名Drive入口再修复：NumPy先在新子进程实际import读取__version__/__file__，兼容且可导入则保留，以运行时版本覆盖约束；仅缺失/损坏/在Python3.13上<2.1时，以--only-binary=:all: --no-deps定向重装numpy2.2.6(已核实cp313 manylinux x86_64 wheel)，不改torch/Transformers。其他核心版本改用metadata.version逐包查询，不再让distributions遍历中的最后一个重复dist-info覆盖版本。旧metadata是否重复尚需现场打印确认，不宣称已实测。8项CPU测试通过，含1.26/Python3.13拒绝与定向修复分支，Drive读回SHA一致。用户仍同命令重试；先看V46_NUMPY_RUNTIME，再看V46_DEPENDENCIES_READY。远端安装/8步GPU仍未验收。交接总结任务暂被用户要求先处理此错误，完成后继续。

- 2026-10-28 v46第一次用户实跑在ensure_dependencies安装阶段失败，pip BackendUnavailable Cannot import mesonpy；尚未进入模型训练，不能称OOM或训练失败。原入口混合DeepSpeed及SwanLab等依赖用--no-build-isolation，并冻结全环境，可能回退到源码构建。已同名入口定向修复：只锁核心torch/Transformers/tokenizers/NumPy等；运行依赖和SwanLab用--only-binary=:all:正常隔离方式补齐；DeepSpeed0.19.8本体单独--no-deps --no-build-isolation --no-binary=deepspeed、DS_BUILD_OPS=0，避免其依赖重新解析/编译；清除继承pip约束/二进制偏好并在模型加载前新进程验证deepspeed/swanlab/ninja/accelerate导入。6项CPU测试通过，含安装顺序/参数模拟，不是远端pip/GPU实测。Drive同名文件已更新读回SHA一致；用户同命令重试，不必安装mesonpy、删除server或重装核心环境。当前报错尾部不能确定具体哪个传递依赖触发源码构建，不宣称已证实是NumPy或SciPy。

- 2026-10-28 P-ALIGN v47 rp1.05已完成500题GPU生成并下载本地，raw/reasoning哈希与完成标记核验通过。原生首段328/500=65.6%，收尾后362/500=72.4%(300带boxed)；195首段length(39%)、全部触发收尾，193收尾stop/2length，最终134无boxed。v48本地DS全读138条原生失败，新增10正确、128错误、0uncertain；助手逐条对照10条原gold和最终输出，核对3向量tuple、1矩阵、其余纯文本公式/解集等价，00347的r2按题目Ar²+Br+C及上下文确认为纯文本平方缩写。辅助复核372/500=74.4%，非人类推导金标；原生通过362未再独立审核。已知2条训练重合均未判对，排除后诊断372/498=74.699%，不能替代完整500分。结果 `artifacts/final_eval_local_snapshot/evaluations/public_palign_7b_math500_v47_rp105/`，`final_summary_v48.json`综合，`native_ds_v48/recovered_answer_checks.json`全部新增判对原文/gold核对证据。用户决策为接近80则LoRA、否则先全参短测；本次74.4不达标，下一步执行既有v46容量/速度probe，不自动6epochs。GPU已可释放；当前聊天未附加远端终端，不宣称v46已启动。

- 2026-10-28 用户明确要求P-ALIGN评测加重复惩罚，并称1.0已运行且有较多重复。已告知先Ctrl+C退出旧任务、保留GPU环境；不自动杀进程。v47同名Drive入口已更新，首段与256token收尾均 repetition_penalty=1.05（共用常量），其余预算/温度/top_p不变。新默认输出独立 `evaluations/public_palign_7b_math500_v47_rp105`，旧1.0输出保留不混用；本地v48复核路径同步更新。启动命令不变，开头明确打印PALIGN_PROTOCOL repetition_penalty=1.05。6项CPU测试通过(包含两段SamplingParams均使用该常量)，Drive字节读回SHA核验一致。需旧进程结束后再执行同命令；不保证1.05消除循环，不改写已有1.0结果。

- 2026-10-28 用户要求实测P-ALIGN作者成品。入口tools/evaluate_public_palign_math500_v47.py已上传Drive同名并读回一致，6项CPU测试通过；当前无附加远端终端，GPU未启动。固定qizheyanger/P-ALIGN revision dd6f8b1b85c95658d573c46103a3501d74b013a5子目录checkpoint-qwen2.5-7b，实际r4/alpha8七类线性层adapter；复用原始Qwen2.5-7B-Instruct固定底座，通过vLLM LoRARequest两段均加载，无需重训或合并大模型。作者仅给底座本地路径无精确commit；本项目固定同名官方底座为明确假设。作者tokenizer词表与底座语义一致。原封存MATH500，每题一次temp0.6/top_p0.95，7936+256总8192、context16384、原生RoPE；提示为处理后训练数据instruction+换行+原题，不插XML think/s1 answer角色。自动缺依赖准备/底座缓存复用/每16题持久化/拒绝已有GPU任务，结果evaluations/public_palign_7b_math500_v47，PALIGN_7B_MATH500_V47_ALL_GENERATION_COMPLETE后可停GPU再本地CPU评分。公开数据有2道MATH500精确重合00080/00103，分别报完整500及排除后498诊断。不是论文4096/n3/MathVerify OR OAT协议精确复现。CPU/DS入口tools/review_public_palign_math500_v48.py已预备，未调用API。操作说明reports/s1_full_sft_probe_20261027_v46/PALIGN_EVAL_v47.md。

- 2026-10-27 用户批准尝试第一个s1K-1.1复现的单A100全参SFT，当前只准备容量/速度短跑v46。Drive入口 `projects/project1/probe_s1_full_sft_v46.py` 及同目录 `s1_full_sft_probe_v46.json`、`.jsonl.gz` 已上传并读回字节一致。原始7B-Instruct固定revision、BF16/SDPA/gradient checkpointing、ZeRO2 CPU AdamW offload、16384窗口、batch1×累积8；64条按长度分位冻结真实样本(最长第一)、8次更新后退出，不自动6epochs、不启动评测、不保存短跑权重。现有dev/MATH500精确排除6题得到994候选，其中103条按参考16K cutoff会截断；新manifest只批准64条probe，不把候选全池当正式训练批准。入口复用v28 prepare-only/原模型/私密SwanLab，检查已有GPU进程，不杀任务、不重装核心栈，只补缺失依赖。5项CPU测试、源码编译、local-check通过；当前聊天无附加远端终端，GPU未启动。结果目录 `runs/project1/s1_full_sft_probe_v46`，结束标记 `S1_FULL_SFT_V46_PROBE_COMPLETE`，看峰值/稳定步速后决定正式训练。报告 `reports/s1_full_sft_probe_20261027_v46/README.md`。历史E0 MATH500 76.4%(382/500)、dev78.125%(200/256)仍是旧2048口径；P-ALIGN69.8/75.6/75.8来自其论文自身协议，改进法较普通SFT仅+0.2pp，不能用作80分保证。

- 进一步核实作者s1.1-7B实际训练记录：下载固定模型revision随附92KB TensorBoard事件并只读解析args/model_config/scalars；training_args.bin仅pickletools反汇编未反序列化执行。实际7B参数为5epochs、315步、LR1e-5、max_seq_length23000、单进程batch1/累积1、16 FSDP进程（由bin num_processes16直接确认，故有效batch16）、full_shard/CPU offload/gradient checkpointing、bf16混合精度、AdamW beta(.9,.95)/weightdecay1e-4/warmup5%/cosine；日志runtime1182.41秒、trainloss0.7434。GPU型号未确认，不能再沿用8×A40说法；公开训练源码无LoRA配置，按全参SFT路线理解，保存F32与BF16混合计算不矛盾。证据artifacts/research_s1_7b/{s11_7b_training_args.json,s11_7b_training_evidence.json,tensorboard_scalars.json,training_args_disassembly.txt}。
- 当前公开s1K-1.1 revision96c411f1fe4c49d20f0e2a1565f61e1a28b0b84d下载只读检查：1000题，math877/science108/crossword15；deepseek_grade Yes630/No370为自动标签非人工金标。按发布格式和本地Qwen tokenizer重建完整chat平均10038、median9725、p9517263、max26969，614条>8192、3条>23000；不是确切历史7B tokenized训练快照，不能称绝对无截断。与现MATH500有3条忽略空白后精确重合(00080,00103,00285)，这3题本次均未判对，当前全量406/500=81.2%不改；排除后仅为诊断406/497，不替换500题分母。与dev也3题精确重合(02148,04620,07294)。后续自己训练须排除现评测题；不能仅据当前数据认定作者历史训练逐题身份。未批准新训练数据、未启动新GPU任务。

- v43/v44分数纠正（优先于下方407/81.4%）：用户追问复核原因后，逐条对照23新增判对与原gold，发现math500/test/00490模型明确答16但gold14，DS误判correct。原生384/500=76.8%不变；保留原DS407和原始输出，独立corrected_summary_v45.json修正为406/500=81.2%，94incorrect/0uncertain。其余22：19条在Final Answer提示后首行明确给正确值，但Math-Verify整段优先抽到后续公式，孤立最终答案重验19条全等价；1条整数解集合顺序不同，2条向量tuple/list对列向量，按题意等价。assistant_recovered_answer_checks.json记录全部23，不能把原始DS81.4继续当最终分。之前只核证据出自原文未逐条比gold不足，需明确承认修正。

- 2026-10-27 作者s1.1-7B成品v43已实际完成MATH500并下载500题原始/最终回答，SHA和唯一ID核验通过；固定revision ff0270ab653ee96811d034acba1a880e48612cc1，BF16、原生RoPE、8192=7936+256/context16384。原生首段391/500=78.2%；收尾后384/500=76.8%（383有框）。118首段length触发收尾，106收尾length/12stop；最终67无框，自动收尾新增9/失去16等价（首段可解析中间量，不能称真实正确率降低）。本地tools/review_public_s11_math500_v44.py复核116自动失败：DS新增23，总407/500=81.4%，93incorrect/0uncertain；不是人工金标，自动通过384未再送DS。新增23里20为收尾开头给出目标答案后继续解释截断，已查看20条首行并记录recovered_answer_presence.json，不是仅搜索正确数字。对比自训v41同题/预算/温度/评分：362自动、364复核(72.8%)；作者高22自动题/+4.4pp，高43复核题/+8.6pp。两模型prompt细节/原生RoPE对YaRN/收尾格式不同，不单独归因于训练或LoRA。结论：成品有价值、72.8不是普通7B能力上限，但不能保证单A100 LoRA复现81.4；未启动新训练/dev/GRPO。结果artifacts/final_eval_local_snapshot/evaluations/public_s11_7b_math500_v43，summary.json原生、native_ds_v44/summary.json复核；GPU可停。

- v43首次实跑在environment摘要json.dumps(prompts)处因Transformers返回BatchEncoding失败，Drive结果仅identity.json，没有保存回答；模型已加载后正常清理，不是OOM。已修复prompt_token_ids显式return_dict=False并兼容Mapping/input_ids与单批嵌套列表，校验整数序列、JSON摘要移到引擎加载前。同名Drive入口已更新读回一致，5项CPU测试通过（含真实BatchEncoding）。仅已知失败旧脚本SHA 5aa6721545fcf7d1350a437869b10a02ab8143a59d9e233ba0cc21bea604b1fa、其他身份相同且无raw/reasoning/environment/完成标记时，自动归档旧identity并记录初始化修复，不能迁移已有输出或任意哈希。用户可同命令重试，复用已下载权重和环境；修复后GPU尚未验证。

- 用户要求直接评测作者成品s1.1-7B的MATH-500。已准备tools/evaluate_public_s11_math500_v43.py并上传Drive同名。固定HF revision ff0270ab653ee96811d034acba1a880e48612cc1，公开权重约30.5GB(F32)，首次下载到MyDrive/models/s1.1-7B--ff0270ab653e，再拷到/content/models，同目录复用；推理BF16不训练。原生RoPE/context16384，同MATH500/seed/temp0.6/top_p0.95/7936+256总8192，与v41比较预算和评分但模型格式及RoPE不同，不称完全受控训练对照。s1原生<|im_start|>think/answer角色格式保留，收尾必要时加answer角色+直接给答案，不插XML think标签、不做Wait延长。每16题持久化，结果evaluations/public_s11_7b_math500_v43，S11_7B_MATH500_V43_ALL_GENERATION_COMPLETE后可停GPU，本地原生两阶段评分已内置。3项CPU测试通过；本聊天无附加远程终端，尚未实际启动GPU。

- 2026-10-27 v41 MATH-500已完成并下载本地，500行原推理/最终输出及完成SHA校验通过，确认8192=7936+256/context16384。原生Math-Verify首段335/500=67.0%，收尾后362/500=72.4%，带boxed且等价361；160首段length触发收尾，156收尾stop/4仍length，最终13无boxed；收尾新增判对30且原先等价3条在最终答案判错，净+27。tools/review_math500_answer_v42.py已CPU/API复核138条自动未通过：DS额外判对2，最终364/500=72.8%，133incorrect/3uncertain，非人类金标。DS完整读取原题/gold/完整原推理/干预提示/新增答案，强制收尾样本只据新增答案判最终对错，引用验证来自最终答案；原生分与原始输出未覆盖。所有结果在artifacts/final_eval_local_snapshot/evaluations/final_math500_answer_v41，summary.json为原生，native_ds_v42/summary.json为复核；GPU可停，未启动其他任务。旧E0 76.4%不同提示/预算/判分，仅历史参考，本轮未证实SFT提分。

- 2026-10-27 用户紧接着纠正v41预算：总生成8192=7936首段+256直接要答案，总上下文16384；其余最后429模型/MATH500/YaRN4/temp0.6/top_p0.95/保留原始与收尾分数不变。已同步更新同名Drive脚本与CPU测试；运行命令不变。此条优先于下方v41旧32768描述。

- 2026-10-27 用户最新决定：跳过前两轮pilot与完整dev，直接测最后429模型MATH-500，要求最后直接要答案。新入口tools/evaluate_math500_final_answer_v41.py已上传Drive同名并读回核验。总生成32768=32512首段+最多256强制收尾；首段length或无boxed时追加直接给单一boxed答案提示，若有未闭合think才补关闭标签，使用原prompt和精确原生成token_ids续写，不是CPU补字符串。YaRN4/temp0.6/top_p0.95/context40960/并发16，题目500行身份校验；每16题保存原始reasoning.jsonl和最终raw.jsonl，原始与续写分别评分，续写评分只看新增答案段，原记录不覆盖、不发送标准答案、不调用DS。新server自动prepare-only恢复与仅补评测依赖，复用Drive merged。结果evaluations/final_math500_answer_v41；MATH500_V41_ALL_GENERATION_COMPLETE后可停GPU再本地评分。3项CPU测试通过，尚未在GPU实跑，未自动启动任何评测。

- 2026-10-27 v39已GPU生成64条并在本地CPU评分，raw/选择SHA核验通过。四组每组16题，原生等价数/截断数/精确尾部重复样本数：native_greedy 7/9/8、native_sample 9/8/7、yarn4_greedy 9/6/6、yarn4_sample 10/5/3。等价且带boxed数6/6/8/9；截断但自动解析等价数1/3/1/1，不视作已完成解题。YaRN4+采样本次症状最少，不支持YaRN是已证实根因；16题一次采样不足外推全量/32k表现。结果在artifacts/final_eval_local_snapshot/evaluations/pilot16_rope_decode_v39，详见对应报告README。GPU可停；未启动新dev/MATH500/637。

- 用户已批准16题推理配置控制对照。v39入口 `tools/run_pilot16_rope_decode_v39.py` 和冻结选择 `reports/rope_decode_pilot_20261026_v39/pilot16_rope_decode_v39.json` 已上传Drive同名文件；从旧dev输出按seed42分层随机选8条length+8条stop，不依新结果挑题。原生/YaRN4×greedy/(temp0.6,top_p0.95)，同最后429模型、同SFT prompt、同8192生成上限、同16384总上下文、并发16、重复惩罚1.0，不加budget forcing。共64回答，GPU端只生成，`PILOT16_ALL_GENERATION_COMPLETE`后可停GPU，再本地原生评分/重复/耗时分析。脚本和选择读回核验、3项CPU测试通过，尚未实际启动GPU；当前聊天无可执行远端终端。
- 用户随后授权DS全读122条截断原回答做四类诊断，已本地CPU/API完成。`truncation_ds_v38/summary.json`：46条较早明确给出正确目标答案后继续重复，3条仅正确中间结果，73条推导/理解/算术错误后循环；这些是DS行为分类，不是新增基准正确数或因果金标。两条引用未逐字匹配，另存assistant_evidence_checks.json核对原文（k=36只是中间量，n=23明确目标）。实际API日志估算空闲价4.3683元含记录重试，不是账单回读。原始输出与旧评分均未覆盖，GPU未启动。需要v39控制对照才能判断YaRN/解码影响，不能据文本断定根因。
- v32 dev已实测完成，本地下载256回答并验证predictions SHA256 `6d3fc96b064660fc04eaf92df7dd47a752fdfbd3876e5d5d9da136f435ad7c73`。旧strict125/256=48.828%，stop134/length122；正确交叉分组是stop/equivalent124、length/equivalent1、length/parse_failure121。本地原生Math-Verify143/256=55.859%；DS复核113条未通过后159/256=62.109%，59判错、38 uncertain，DS额外判对16；不是人类金标。结果在 `artifacts/final_eval_local_snapshot/evaluations/final_dev256_32k_v32/native_ds_v33/`。
- 截断症状已CPU检查：121条parse_failure都length且无boxed；36有<think>开头、85无该标签；118条tail8192中32-token精确窗口重复>=4次；前2000/6000 tokens原生解析等价候选分别12/16，不证明已完成解题。121条全输出复核31判对/52判错/38不确定。122条截断消耗总生成tokens的92.737%。不要把仅正常结束样本的条件正确率宣传为总体能力，不能断言YaRN是已证明原因或“模型根本没训废”。
- 用户提供Gemini建议后要求CPU分析，不启动GPU；目前MATH-500/637未启动，GRPO未启动。Qwen官方说明静态YaRN可能影响短文本，DeepSeek-R1建议温度0.5~0.7，但本模型是Qwen普通Instruct+本地SFT；因果尚需小规模控制对照。32768为后续用户明确恢复的生成预算，8192是训练窗口，二者别混。若未来去掉YaRN，总上下文32768必须容纳prompt+output，不可保留40960却只删rope override。Budget forcing要再次模型生成，不是CPU文本修补，不能保证救回半数。Drive已核实checkpoint143/286/429及前两轮adapter/optimizer/trainer_state存在。
- 用户要求GPU不能空等DS：复核改在本地CPU+DS API做，`tools/review_final_local_v34.py`，解释器D:/anaconda3/envs/genai-test/python.exe；本地凭证C:/Users/Mr.Z/.codex/private/project1/deepseek.json，已保存，不回显。原始输出从Drive同步到artifacts/final_eval_local_snapshot，不下载权重。v33新增 `--generate-only`：Colab可只生成MATH-500/637，不调用DS或等待复核；可与 `--stage next` 组合仅依次生成math500、hard637。当前dev仍在生成，尚无predictions.jsonl，不能宣称本地复核已开始。
- v32运行期间不修改其脚本或终止任务。后续入口 `tools/continue_final_eval_v33.py`（Drive同名）已补齐：`--stage review-dev`仅CPU/DS复核已完成v32的原始输出；`--stage math500`测最后模型500题并复核；`--stage hard637`测原637题×4并复核；`--stage next`按此顺序串行。默认仅review-dev，任何阶段均要求v32 COMPLETE存在。后续统一AIMO SFT prompt_ids、32768生成、40960上下文/YaRN4、动态并发16。MATH-500一次采样temp0.6/top_p0.95；637复用原题池和trial seeds/temperature/top_p，但新提示/预算不同于旧E0探针，不能宣称旧协议精确对照。只用最后模型，不测前三轮、不重测E0。原生自动分+DS失败复核分分别保存；无GPU模型的CPU复核不需要保留GPU实例。
- 最新提示词纠正：v32必须直接调用 `experiments/aimo_sft_v26/common.py::prompt_ids`，与本轮AIMO SFT一致（单user原题+固定后缀），不能沿用旧E0的system消息格式。dev256/最后模型/32768优先级不变，旧E0分数仅参考。
- 最后一次用户修正优先级：先只测最后一轮的dev256，tokens预算恢复32768，不自动跑MATH-500/637或前两轮。入口 `tools/evaluate_final_dev256_v32.py`（Drive同名），使用AIMO SFT用户提示格式及旧strict辅助评分，每题一次greedy，40960上下文+YaRN4，动态调度上限16（不按4题阻塞）。结果独立 `evaluations/final_dev256_32k_v32`，结束标记FINAL_DEV256_COMPLETE。旧E0 dev78.125%是2048预算，非同预算对照。待此分数出来再讨论后续；v30/v31仍保留，不自行启动。
- 用户要求顺便测637道E0四次全错的条件探针。新总入口 `tools/evaluate_last_with_hard637_v31.py`（Drive同名）先跑v30 MATH-500，再用同一个最后模型跑637题×4，沿用Drive `rl_screening_v8_n4` 的原始prompt tokens、温度/top_p、2048预算与每ID seeds；结果独立放 `evaluations/final_hard637_e0_protocol_v31`，不称独立测试或训练批准。v30漏依赖rollout_pipeline/runner_support已去掉，旧协议内置并验证便携版500题与旧E0逐行同题同答案；实际底包解压测试已通过。
- 最新选择：用户嫌三个dev及32k评测太慢，跳过dev选择和前两个checkpoint，只测最后一轮；先按旧E0同口径跑MATH-500。使用 `tools/evaluate_last_e0_protocol_v30.py`（Drive同名），2048/greedy/旧system提示/并发上限128/旧冻结strict-v3；本次不启动v29或DS。独立结果 `evaluations/final_math500_e0_protocol_v30`，不覆盖长预算结果。说明见 `reports/evaluation_ds_20261026_v29/旧E0口径快速评测_v30.md`。
- 用户已贴出v26正式训练及merged导出完成日志：429步、3epochs、train_loss 0.5705、训练循环约4小时03分，SwanLab run v9l5kmap。新评测分数仍待实测。
- 用户将MATH-500改为主测，暂不跑AIME，本轮不重新生成E0。dev256选择epoch后测所选SFT的MATH-500，保留32768生成预算；旧E0 76.4%仅作旧预算/旧判分口径历史参考。
- 用户批准评测侧改用原生Math-Verify 0.9.0，取消strict-v3额外拒绝规则；未通过样本用DeepSeek复核，必须发送原题、原始标准答案与完整原输出。保留自动分、DS复核分与uncertain，不能覆写旧记录；训练/RL原有strict-v3不受此评测变更影响。
- 新入口 `tools/evaluate_sft_ds_v29.py`（Drive同名文件）负责顺序dev、复核、选checkpoint、MATH-500；说明见 `reports/evaluation_ds_20261026_v29/README.md`。沿用已生成raw，不重复生成；旧生成器strict分仅为辅助记录。
- DS凭证已独立保存并真实合成题API验证成功，私密Drive路径 `.project1_credentials/deepseek.json`，字段DEEPSEEK_API_KEY；不要求用户重复输入，不打印密钥。该凭证与SwanLab分开。

## 阅读入口

1. 先读[第三次对话完整接手提示词](reports/handover_20261026_v4/第三次对话_完整接手提示词.md)，尤其当前状态、最新启动命令、已知错误与修复。
2. 再看[README.md](README.md)的最新入口。正文中旧“GPU尚未验收”等表述是历史快照，不能覆盖后续实测。
3. 当前任务读[训练配置](configs/project1_aimo_sft_v26.json)、[新数据manifest](reports/aimo_sft_20261026_v26/manifest.json)与源码；研究旧raw/compact时再读[data/portable_release/manifest.json](data/portable_release/manifest.json)。
4. 不需要先遍历原始数据、缓存和全部历史报告。旧project_blueprints/配置可能含旧绝对路径。

## 当前主线

- 用户要做Qwen2.5-7B-Instruct → SFT取得实测提升 → GRPO继续提升，与Qwen2.5-Math-7B-Instruct对标，形成可写简历的结果链。不能预先保证涨分。
- 固定模型 `Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`。不要切回R1-Distill，也不自行更换数据集。
- 当前数据是原始AIMO2 sft-4k的8192完整窗口子集：**2275条、2263道题**；全4033条扫描，不压缩、不截断、不再限1000条；有/无think标签的原始推导都保留。
- 训练配方称v26，**最新启动入口为v28**，叠加v27兼容补丁。LoRA r32/alpha64/dropout0.05，LR2e-5，3epochs，有效batch16；默认batch2×累积8，可选4×4；逐epoch保存，在线详细SwanLab。
- 用户日志已确认本轮LoRA注入成功（80,740,352可训练参数），但先后遇到torchao、warmup_ratio、SwanLab认证错误。**本轮完整训练和提分尚未确认**。
- 用户明确说过有任务在跑、不能重载VS Code。先看最新状态，不重启窗口、杀进程、终止运行时或重复启动GPU任务。
- **弹窗排查已由用户暂停**。插件虽然安装，实际未弹出，不能宣称可用；不继续排查或把弹窗作为训练前置条件。v28默认跳过通知包装层。

## 路径、启动与恢复

- 本地根目录 `D:/Users/Mr.Z/Desktop/项目一_7B数学后训练`，CPU解释器 `D:/anaconda3/envs/genai-test/python.exe`。独立于旧modern_genai，不能引用旧`.venv`。
- 运行采用VS Code **Colab远程终端**；notebook仅用于连接和挂载Drive。不能把Python挂载代码贴到Bash。
- 使用Drive账号 `padaxingbakamaka@gmail.com`，单卡A100 80GB。保留预装Python/PyTorch/Transformers，只补缺失依赖和已证实的兼容修复。
- Drive持久目录为`projects/project1`、`models`、`runs/project1`。工作副本 `/content/workspaces/aimo-sft-v26`；默认结果 `/content/drive/MyDrive/runs/project1/aimo_sft_v26`。
- Drive挂载后且没有同任务运行时，最新命令：

```bash
python3 /content/drive/MyDrive/projects/project1/start_aimo_sft_v28.py
```

- `--batch-size 4`和`--output-root`可继续传入；恢复沿用原值。入口自动读凭证、还原旧包、应用补丁、处理依赖和启动，不用手工拼分片或重下已有模型。
- 源码：[v28自动认证](tools/start_aimo_sft_v28.py)、[v27兼容入口](tools/start_aimo_sft_v27.py)、[训练](experiments/aimo_sft_v26/train.py)、[环境](experiments/aimo_sft_v26/install_missing.py)。不要再推荐旧v26入口，它会重新解压旧代码。

## 已知兼容项必须落实到入口

- torchao 0.10.0与当前PEFT不兼容，用户早已明确提醒。v27自动将已安装且低于0.16.0的版本定向升级到0.18.0，使用`--no-deps`，不连带升级torch。
- Transformers 5.20.2不接受`warmup_ratio`。v27使用整数`warmup_steps`（本轮22步），在模型加载前验证TrainingArguments；长度分组适配旧`group_by_length`与新`train_sampling_strategy`。
- 历史`pyroaring<1.0.4`曾解决SwanLab安装问题，按具体错误处理，不无条件重装整套依赖。
- **SwanLab必须在线、详细记录、自动认证。**用户已提供Key并授权保存复用。专用私密文件 `/content/drive/MyDrive/.project1_credentials/swanlab.json`，字段`SWANLAB_API_KEY`；v28读取后注入子进程环境，不要求用户每次输入或export，不静默转offline。
- 私密凭证已写入当前Drive且仅所有者可访问，当前Key已通过真实在线登录验证；不等于远端训练成功。不得打印Key、整份读取后回显、写入AGENTS/交接文档/公开运行包。
- v27仅对已知旧源码、相同配置/模型/数据且无checkpoint或训练状态的失败初始化归档并迁移run_identity。不能删训练状态或改批准哈希来绕过检查。

## 数据与评测

- 新数据用独立manifest和逐行授权；旧1500raw/1500compact是同一批题的两臂，不是3000道题。旧数据与批准清单继续保留。
- 训练/评测范围以对应manifest的approved_training_files/approved_evaluation_files及逐行授权为准；候选不等于批准，AI审核或答案等价不等于人工推导金标。不能手改批准字段或哈希绕过准入。
- 根目录4810条RL文件仍是旧候选；第一对话已有另一快照筛到4522，E0每题4次rollout后**920条获批**，保存在Drive `runs/project1/rl_approved_v8_n4`。不能说项目从未筛出RL数据，也不把该E0通过率当新SFT模型通过率。
- 637题是E0四次全错的条件探针，不是独立最终测试，不因后来有正确答案自动获批。
- 用户确认现有去污已完成，不重复全量去污/审计/扩量。验证聚焦当前报错与身份关键项，避免防御性检查阻碍执行。
- 训练窗口8192与评测生成预算32768分开。dev256选checkpoint；AIME2024/2025主测，MATH-500辅助。AIME每题64次的平均单次正确率是pass@1估计，不是pass@64或多数投票。
- 新评测E0/SFT统一32768生成上限、40960上下文、YaRN factor4；dev每题1次greedy，AIME温度0.6/top_p0.95每题64次，MATH-500每题1次；评测并发4。不将推理并发4混同训练batch4。
- 最后一轮merged不自动称最佳；先比较dev，再正式测试。旧2048预算成绩不能直接充当新协议基线。
- 使用math_reward_v2/冻结strict-v3评分路径；答案等价不是逐步证明正确，解析失败等单列。保留合理限制，但不因此拒绝继续实验。

## 历史与工作方式

- 历史7B GPU smoke、raw/compact正式SFT、v18训练和评测均实际完成。旧“只有2步smoke、GPU未验收”的总括结论过时。
- 旧MATH-500（2048）：E0 76.4%、raw 76.2%、compact 75.4%。v18 compact新配方（4096）55.2%，不作为当前GRPO起点。完整预算与证据见交接。
- MCMC v21/v24已实际试过，用户决定暂时放弃；不默认扩量，不把投影数据加入新SFT。v24成本约9倍、正确数不变，且有推导错误案例。
- 优先执行，给完整可复制的重连/挂载/恢复命令，明确结束标记与何时可停GPU。不重复索要已授权的许可。
- 文档与代码保留来源和验证记录，CPU通过不代替GPU结果。v27为15项聚焦CPU测试，v28认证入口2项；不必每次重跑全项目回归。
- 离线迁移入口仍是`python tools/verify_migration.py`，不是每次任务前置条件。E0/E1/E2/E3/E4框架保留，当前先完成新SFT再接GRPO，不为接手重跑旧实验。
