"""Summarize actual screening outputs; no model/API calls."""
import json
from pathlib import Path
from src.data.screening_io import sha


def run():
    def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
    math=load('data/project1_openr1/manifest.json')
    code=load('data/project2_code/manifest.json')
    official=load('reports/official_code_overlap_audit.json')
    text=['# 2026-10-23 数据筛选与四项文档修正','',
          '本轮 CPU 执行；没有付费 API、没有模型权重下载、没有训练。',
          '原设计文档备份：`reports/backups/blueprints-before-screening-20261023.zip`。','',
          '## 项目一：逐轨迹紧凑 SFT 候选','',
          f'- 来源 revision：`{math["source_revision"]}`，default/train。',
          f'- 扫描 {math["source_rows_scanned"]} 道源题，接受 {math["selected_rows"]} 道唯一题，目标 1500。',
          '- 每条轨迹对齐完整性/数学正确性标记；完整回答 token 范围 500～1600，总 SFT 上下文不超过 4096。',
          '- 同题多条合格轨迹选择其中最短的一条，长度相同按轨迹索引排序；不截断候选。',
          '- 以 seed=42 调整 shard 顺序，达到目标即停止；不是全源随机抽样或全量流行率调查。',
          '- 排除共同数学 dev、MATH-500、GSM8K-test 的当前规则下重叠；词法去重不证明语义零污染。',
          '- 源数据验证标记不证明过程完全正确；已导出 50 条人工审核表，尚未人工审阅，不冒称金标。',
          f'- 来源类别：`{json.dumps(math["source_category_counts"],ensure_ascii=False)}`。',
          f'- 长度分桶：`{json.dumps(math["response_length_buckets"],ensure_ascii=False)}`。',
          '- 候选：`data/project1_openr1/sft_train.jsonl`；审核表：`data/project1_openr1/human_audit_50.jsonl`。',
          '- manifest：`data/project1_openr1/manifest.json`；逐轨迹审核：`data/project1_openr1/trajectory_audit.jsonl`。','',
          '## 项目二：代码接口分类候选','',
          f'- 来源 revision：`{code["identity"]["source_revision"]}`；全源扫描 {code["source_rows"]} 行。',
          f'- 初筛结构候选 {code["candidate_count"]} 条；额外字面量检查隔离 {code["literal_diagnostics"]["excluded_rows"]} 条，剩余 {code["final_candidate_count"]} 条；最终家族/词法去重移除 {code["final_deduplicated_families"]} 条。',
          f'- 最终训练 {code["train_rows"]} 题，dev {code["dev_rows"]} 题。',
          f'- 训练接口：`{json.dumps(code["actual_train_interfaces"],ensure_ascii=False)}`。',
          f'- dev 接口：`{json.dumps(code["actual_dev_interfaces"],ensure_ascii=False)}`。',
          f'- 训练 callable 目标 1250，实际 {sum(v for k,v in code["actual_train_interfaces"].items() if k!="stdin_stdout")}；此来源不能满足 50% 目标，补充来源需单独验证和去重。',
          '- 约 50% callable 只是目标配比；独立函数/类方法不足时保留实际数，不通过重复题凑比例。',
          '- 检查所有测试用例 type/fn_name、JSON/Python 字面量参数与 arity、参考 AST/入口/return；显式命名参数按签名顺序绑定，不执行真实数据集代码。',
          '- 缺参、C 风格数组、树/图对象或其他歧义输入均隔离，不凭 fn_name 非空强行判断可执行；静态可解析不代表运行正确或测试覆盖充分。',
          '- 初筛 4 道函数候选中，另隔离 1 道“str = 101 却直接按字符串遍历”的明显类型冲突；没有自动猜测并改写输入类型。窄范围静态检查不替代沙箱运行。',
          '- 训练只输出 prompt/接口和来源元数据；dev 参考代码/测试保存在独立验证文件，未拼入训练 prompt。',
          '- 隔离沙箱执行参考程序与人工测试覆盖审计仍待完成；当前不是已经执行验收的代码训练集。',
          '- `data/project2_code/train_prompts.jsonl` / `dev_metadata.jsonl` / `dev_verification.jsonl` / `interface_audit.jsonl`。','',
          '## 共用数学划分与基准','',
          '- 原 `data/math/math_dev.jsonl`（256题）及 `math_train.jsonl`（7184候选）未重写，哈希身份相同。',
          f'- 项目二数学池：MATH {code["math_pool"]["math_count"]}＋GSM8K {code["math_pool"]["gsm_count"]}，`data/project2_code/math_train_prompts.jsonl`。',
          '- 当前 dev 的题面、错题、变式均不能进入训练或生成请求；泛化失败只能用于题型级诊断，再从 train 选择种子。',
          '- HumanEvalPlus v0.1.10：164；MbppPlus v0.2.0：378。与旧文档 399 题说法不同，以真实 task ID 清单为准。',
          f'- 官方发布文件补充重叠检查通过：`{official["checked"]}`；详细版本/哈希/ID：`reports/official_code_overlap_audit.json`。','',
          '## 复现命令','',
          '```powershell',
          '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.download_screening_sources',
          '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m src.data.screen_openr1_sft',
          '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m src.data.screen_code_interfaces',
          '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.export_code_function_cache',
          '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.quarantine_code_literal_conflicts',
          '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.audit_official_code_benchmarks',
          '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m pytest tests -v',
          '```','',
          '所有大日志重定向到 reports；source manifests 固定 revision、缓存已下载 parquet。OpenR1 每250题保存恢复点。',
          '环境与脚本哈希：`reports/data_screening_environment.json`。附加字面量诊断使用本次全源扫描导出的 198 行函数源记录缓存 `data/screening_sources/code_function_rows.jsonl`，其哈希记入 manifest。',
          '筛选配置变化时应另建产物版本，不能覆盖已封存训练/dev。','',
          '## 测试证据','',
          '完整输出：`reports/data_screening_pytest_full.log`。','```text',
          Path('reports/data_screening_pytest_full.log').read_text(encoding='utf-8').rstrip(),'```']
    Path('reports/data_screening_20261023.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('Screening report saved')


if __name__=='__main__':run()
