"""Human-readable execution record from sealed pilot and bulk artifacts."""
import json
from pathlib import Path


def run():
    def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
    pilot=read('data/code_v2_pilot/report.json');manifest=read('data/project2_code_v2/manifest.json')
    sandbox=read('reports/code_v2_sandbox_selftest.json')
    tests=Path('reports/code_v2_pytest_full.log').read_text(encoding='utf-8').strip().splitlines()[-1]
    text=['# 项目二代码数据 v2：小样本验收与批量构建','',
      '执行日期：2026-10-23。范围：CPU 数据工程与隔离代码测试；无模型训练、GPU 操作或付费教师 API。','',
      '## 1. 先验收，再扩量','',
      '- AceCode 固定 revision `13216309a9f6cb40b60cb1a9750071efeac414ad`；Magicoder 固定 revision `5f839b1f368a76b161028bb9edff055db34022b2`。',
      '- 各抽查 50 条：seed42 从 Ace 第一 shard 前2000行、Magicoder 前2000条 Python 记录中取样。不是全源随机抽样，不能估计全源正确率。',
      f'- Ace50条最终保留{pilot["stats"]["ace"]["counts"]["static_candidate"]}条，Magicoder50条最终保留{pilot["stats"]["magicoder"]["counts"]["static_candidate"]}条；逐条结果与拒绝原因见 `data/code_v2_pilot/source_audit.jsonl`。',
      '- 初筛后的22条由 Codex agent 逐条审阅题面/签名/可见参考；隔离8条规格冲突，保留Ace6条、Magicoder8条。不是人工审核。',
      '- Ace保留的6条全部通过容器内全套源测试，并拒绝None常量、首个期望值常量两类错误实现；Magicoder8条仅验收OPD prompt接口，不冒充有完整测试。',
      '- 回归检查10项通过：入口与辅助函数区分、显式签名冲突、所有用例参数绑定、危险/复杂接口隔离、改名代码结构和最终测试/旧dev保护。',
      '- `data/code_v2_pilot/report.json` 中的 `bulk_gate_passed=true` 才允许批量入口运行；规则、审核记录和保护文件的SHA同时校验。',
      '- 已发现问题及排除理由：`data/code_v2_sources/pilot_quarantine.json`。隔离不是根据学生模型是否答对而决定。','',
      '## 2. v2实际产物','',
      f'- 训练：{manifest["train_rows"]}题，接口 `{json.dumps(manifest["train_interfaces"],ensure_ascii=False)}`。',
      f'- dev：{manifest["dev_rows"]}题，接口 `{json.dumps(manifest["dev_interfaces"],ensure_ascii=False)}`。',
      f'- 训练来源：`{json.dumps(manifest["train_sources"],ensure_ascii=False)}`；dev来源：`{json.dumps(manifest["dev_sources"],ensure_ascii=False)}`。',
      '- 若训练函数题全由Ace满足，Magicoder作为已审计的备用池保留，不为混来源而强行加入。',
      f'- 扫描范围：{manifest["source_scan"]["scan_scope"]}',
      f'- 结构候选4000条，家族过滤后{manifest["source_scan"]["after_family_filter"]}条；过滤统计 `{json.dumps(manifest["source_scan"]["family_rejections"],ensure_ascii=False)}`。',
      '- Ace仅下载本轮使用的前2个shard；Magicoder下载单个源JSONL。各文件SHA见 `data/code_v2_sources/*_manifest.json`。',
      '- OPD训练文件只含题面、接口、来源和审计元数据，不含解答、测试或奖励标签；Magicoder参考代码只参与静态筛选与去重。',
      '- `data/project2_code_v2/train_prompts.jsonl`：2500条OPD代码训练prompt。',
      '- `data/project2_code_v2/dev_metadata.jsonl`：200条dev题面和接口。',
      '- `data/project2_code_v2/dev_verification.jsonl`：独立保存参考实现、全部测试及执行证据，禁止拼进训练prompt。',
      '- `data/project2_code_v2/manifest.json`：版本、规则、来源、环境、文件SHA。',
      '- 数学训练池继续使用v1的MATH1500＋GSM1000；数学dev仍为原256题，本轮未重新划分。','',
      '## 3. 执行验收和隔离边界','',
      f'- 所选200道dev的参考实现均通过全部已有测试：函数断言{manifest["dev_cases"]["function"]}条，stdin/stdout用例{manifest["dev_cases"]["stdin_stdout"]}条。',
      '- 函数dev至少3组不同输入，并拒绝两种常量错误实现。训练题不要求都有测试，也不把参考实现当学生监督解答。',
      '- stdin/stdout从旧dev及从未训练的v1 prompt中重新封存，按用例数量、seed42哈希排序；每题至少3例，全用例运行，每题总预算20秒。旧dev不足100条通过时，扩大候选池而非放宽测试。此选择偏向低验证成本，属于可见分布偏差。',
      '- stdout采用按空白分词后的精确比较，不支持自定义checker、浮点容差或多合法输出。仅通过当前约定的题进入此版dev。',
      f'- 实际运行Python：`{pilot["sandbox_python"].strip()}`；本机已有镜像 `{manifest["sandbox"]["image_id"]}`。',
      '- Docker禁网、只读根文件系统、非root UID65534、drop全部capabilities、no-new-privileges、限制CPU/内存/进程/文件/墙钟时间；只挂载只读worker脚本，源程序通过stdin送入。',
      '- 隔离自检：写只读根目录失败、外网连接失败、非root检查通过；无限循环被终止。记录 `reports/code_v2_sandbox_selftest.json`。',
      '- 官方Python镜像拉取遇到网络EOF，故使用已有镜像完成本轮兼容性筛选；没有改全局代理。正式训练/评测环境更换Python版本时必须重新运行dev，不能直接继承本次执行结论。',
      '- 本轮Docker启动时发现历史AF_UNIX残留socket，保留并改名socket父目录后引擎恢复；未出厂重置，未删除镜像、容器、卷。恢复记录见 `reports/code_v2_docker_recovery.json`。','',
      '## 4. 去重与开发集隔离','',
      '- 最终基准：本地锁定HumanEvalPlus164题、MbppPlus378题；旧代码dev全部200题继续作为训练排除对象。',
      '- 本轮没有引入MBPP train；已知其中108题与当前MBPP+任务ID重叠。',
      '- 同时检查规范化题面/3gram Jaccard≥0.85、完整参考AST、目标函数去注释/局部改名后的AST。相同入口名仅作为保守复核/家族分组依据，不等同重复证明。',
      '- AceCode含Magicoder派生题，不当作独立来源随机切分；同入口和相同代码结构候选先保守合并，家族不跨train/dev。',
      f'- 新dev与旧dev重叠{manifest["legacy_dev_overlap_rows"]}题，故v1/v2 dev不独立；额外从未训练的v1 prompt重新封存部分dev，这些题全部排除出v2训练。不能比较两版总分来声称进步；v2各实验组须使用同一版dev。',
      '- 词法/结构筛选不能证明语义零污染，也不能排除底座预训练接触。','',
      '## 5. 输入预算与质量边界','',
      f'- 学生Qwen2.5-1.5B实际tokenizer检查：输入≤2560，生成预留1536，总上下文4096。超长训练题排除并同接口补齐{manifest["input_budget"]["rejected_and_refilled"]}条。',
      f'- 实际训练输入最大{manifest["input_budget"]["actual_train_max"]}token，dev最大{manifest["input_budget"]["actual_dev_max"]}token。',
      '- 使用manifest记录的用户消息＋默认system chat template；更换system或模板须重新审计。没有截断题面或源断言。',
      '- 参考实现全通过和常量变异检查只是相容性与最低区分力证据，不证明全部测试符合题意或覆盖充分。',
      '- 仅小样本完成agent规格审阅；全量dev规格/覆盖审计仍待完成，抽检队列 `agent_audit_queue_50.jsonl` 不能写成已经人工通过。',
      '- 不把本轮CPU测试写成模型Pass@1提升，不把源inferences的pass_rate直接当金标。','',
      '## 6. 复现入口','',
      '```powershell',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m src.data.code_v2_sources',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m src.data.audit_code_v2_pilot',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.seal_code_v2_pilot',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m src.data.code_v2_sources --ace-shards 2',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.prepare_code_v2_stdio_extension',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m src.data.build_code_v2 --extend-stdio-dev',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.apply_code_v2_input_budget',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m pytest tests -q',
      '```','',
      '前提：Docker引擎正常；本机固定镜像和tokenizer存在。`seal_code_v2_pilot`里的逐条审核清单只适用于这次固定样本，换样本必须重新审阅，不是通用自动签字器。',
      '已封存产物只允许相同配置复用；规则改变须新版本。容器命令含本机worker绝对路径，工作区迁移需新建环境记录。','',
      '测试结果：`'+tests+'`；完整日志 `reports/code_v2_pytest_full.log`。','',
      '## 7. 上游资料','',
      '- [AceCode数据卡](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K/blob/13216309a9f6cb40b60cb1a9750071efeac414ad/README.md)',
      '- [AceCoder论文：种子来源与合成测试](https://arxiv.org/html/2502.01718v1)',
      '- [Magicoder数据集](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)',
      '- [Docker同类残留socket问题](https://github.com/docker/for-win/issues/15064)']
    Path('reports/code_v2_execution_20261023.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('v2 execution report saved')


if __name__=='__main__':run()
