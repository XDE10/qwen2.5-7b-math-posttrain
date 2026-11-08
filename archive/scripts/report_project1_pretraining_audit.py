"""Report measured SFT selection bias and fixed-tokenizer readiness limits."""
import json
from pathlib import Path
from src.data.screening_io import json_write


def all_correct(counter):
    total=0
    for key,count in counter.items():
        if key=='unknown':continue
        numerator,denominator=map(int,key.split('/'))
        if numerator==denominator:total+=count
    return total


def run():
    def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
    m=read('data/project1_sft/manifest.json');d=read('data/project1_sft/distributions.json')
    t=read('reports/tokenizer_alignment_revalidation_20261023.json')
    n=d['scanned_questions'];after=m['selected_rows'];before_full=all_correct(d['before_questions']['teacher_success_proxy'])
    after_full=all_correct(d['after_questions']['teacher_success_proxy'])
    bias={'population':'Original seed42 scanned prefix, raw source rows including repeated UUIDs; not whole-source prevalence.',
          'source_rows':n,'source_uuid_repeated_rows':d['repeated_source_uuid_rows'],
          'before_all_source_trajectories_verified_fraction':before_full/n,
          'after_all_source_trajectories_verified_fraction':after_full/after,
          'before_all_verified_count':before_full,'after_all_verified_count':after_full,
          'difficulty_balance_status':'NOT_ESTABLISHED','true_difficulty_labels_available':False,
          'conclusion':'Strong selection for shorter, source-teacher-solvable answers. Cannot certify balanced difficulty or absence of easy-problem bias.',
          'problem_type_shift':{key:{'before_rows':d['before_questions']['problem_type'].get(key,0),
                 'after_rows':d['after_questions']['problem_type'].get(key,0),
                 'before_fraction':d['before_questions']['problem_type'].get(key,0)/n,
                 'after_fraction':d['after_questions']['problem_type'].get(key,0)/after}
                 for key in sorted(set(d['before_questions']['problem_type'])|set(d['after_questions']['problem_type']))}}
    json_write('reports/project1_sft_bias_assessment.json',bias)
    lines=['# 项目一 SFT 与四模型 tokenizer：训练前复核','',
      '日期：2026-10-23。结论：此前两项已有产物，本轮复用并补齐源回放、正式路径、RL重叠计数和分布审计；没有重复下载权重、重新训练或替换开发集。','',
      '## 1. 项目一正式 SFT 产物','',
      '- 配置为OpenR1-Math-220k的 `default` config、`train` split；不是名为default的split。源revision `'+m['identity']['source_revision']+'`。',
      '- 正式目录：`data/project1_sft/`。保留原1500个题目ID和完整解答，新增 `answer_token_count` 别名、难度缺失标记、RL重叠标记；原 `data/project1_openr1/` 未覆盖。',
      f'- 固定seed42扫描前缀共{n}行，回放其中1500条所选轨迹；源UUID有{d["repeated_source_uuid_rows"]}行重复，故按扫描位置对齐审计行，不能以UUID字典覆盖前一次记录。',
      '- 每题一条：逐索引核对generations[i]、is_reasoning_complete[i]、correctness_math_verify[i]与结束原因；完整原文、源题面、最短合格轨迹规则均逐条复核。',
      f'- Qwen2.5-7B真实tokenizer：完整回答{m["answer_tokens"]["min"]}～{m["answer_tokens"]["max"]}token；题面＋system＋chat template＋回答/EOS最大{m["total_tokens_max"]}token，低于4096。没有截断回答。',
      f'- 长度桶（入选1500条）：`{json.dumps(d["length_after_selected"],ensure_ascii=False)}`。',
      '- `sft_train.jsonl`：训练候选；`manifest.json`：身份、哈希和规则；`trajectory_revalidation.jsonl`：逐条源回放证据。',
      '- `distributions.json`：筛选前后题型、来源、question_type、长度与难度缺失情况；`human_audit_50.jsonl`：覆盖非空题型×长度桶及来源的50条人工待审表，未填human_verdict。','',
      '## 2. 开发/测试与RL训练池重叠','',
      '- 数学dev256题、MATH-500、GSM8K test1319题全部作为只读保护集合；原7184道MATH训练候选池也逐题核查。',
      f'- 精确/规范化匹配＋3gram Jaccard≥0.85下：dev/test命中{m["evaluation_overlap_rows"]}条，RL候选池命中{m["rl_overlap_rows"]}条。`rl_overlap_audit.jsonl`为空对应零命中，不是缺失审计。',
      '- 策略明确为允许记录在案的train-to-train重叠；本轮实际为0。将来更换SFT或RL池须重验并报告命中ID，不能把训练集合共享等同评测泄漏。',
      '- 零命中仅对应当前词法规则，不证明改写/同构题或底座预训练污染为零；封存dev/test与RL文件哈希保持一致。','',
      '## 3. 分布与难度：尚未通过的质量条件','',
      '- 难度字段在该快照中不存在：筛选前28572行、筛选后1500题均为unknown。不能把olympiads来源、题型或回答长度当作真实难度标签。',
      '- 前后题型如下，前者分母是扫描源行（包含重复UUID），后者分母是1500道去重后的入选题：']
    for key,record in bias['problem_type_shift'].items():
        lines.append(f'  - {key}：{record["before_rows"]}（{record["before_fraction"]:.2%}）→{record["after_rows"]}（{record["after_fraction"]:.2%}）。')
    lines += ['',
      f'- 源R1所有已记录轨迹都通过math-verify的比例：扫描前缀{before_full}/{n}（{before_full/n:.2%}），入选后{after_full}/{after}（{after_full/after:.2%}）。这只是源教师可解性代理，不是真实难度或学生Pass@k。',
      '- 代数比例上升、几何与不等式比例下降，可解性代理也大幅上升；有明显选择偏差，不能声称已确认难度均衡或消除易题偏差。',
      f'- 长度筛选前统计的{d["length_before_denominator"]}条轨迹，已经先通过索引正确性/完整性/结束结构检查：`{json.dumps(d["length_before_compact_filter"],ensure_ascii=False)}`。',
      '- 未正确/不完整轨迹在原筛选阶段没有计token，未将它们伪装成全源长度分布；同题多轨迹与每题一条的计数单位也不同。',
      '- 本轮保留紧凑SFT候选作为固定基线，不自动重采样或改长度凑均衡。要验收难度分布，需要额外可信难度标签、分层解题审核，或训练侧统一模型/预算的可解性诊断；随后版本化决定是否补样。',
      '- 人工过程正确性抽检仍待完成，源math-verify通过不是逐步推理正确的独立证明。','',
      '## 4. 四模型 tokenizer 与词表对齐','',
      '- 原报告：`reports/tokenizer_alignment_report.json`；本轮离线复验：`reports/tokenizer_alignment_revalidation_20261023.json`。没有重新解析main分支来漂移既有revision。',
      '- 四模型所有本地tokenizer/config文件SHA、added tokens、special IDs、chat template哈希和完整token→ID映射重新核对。',
      f'- {t["case_count"]}个数学LaTeX、Python缩进、Unicode和特殊符号样本全部裸文本编码一致，各类16例；独立应用各模型chat template时编码不全一致。',
      '- 共同有效ID：0～151664（共151665）；映射差异0。学生输出行数151936，三教师152064，分别有271和399行在本轮共同有效支持集之外。',
      '- 仅对这四个固定快照可用 `logits[..., :151665]`；分别对教师/学生在FP32中做log_softmax重新归一化，再计算KL。它是条件于共同支持集的目标，不完全等同全输出空间KL。',
      '- 生成采样前也须屏蔽其余输出行；不能只在KL计算时裁切。温度统一记录，首版T=1。',
      '- 使用学生冻结模板渲染一次，将同一token-ID前缀传给所有教师。不能分别套Math教师默认模板，再将不同前缀上的logits逐位置比较。',
      '- 屏蔽前的尾部概率质量可用 `1-exp(logsumexp(valid_logits)-logsumexp(all_logits))` 计算。本轮未加载模型，未测真实尾部概率质量。','',
      '## 5. 是否是训练前最后一步','',
      '- 两项属于数据与tokenizer静态基础验收，本轮已完成可执行的索引/文件/长度/映射复核。',
      '- 仍需完成SFT人工过程审核与难度偏差处置、项目二全量dev规格/覆盖审核。',
      '- 7B完整更新的显存和恢复smoke、真实rollout截断、assistant-only mask/EOS/shift、教师前缀一致性及真实logits尾部质量等运行验收，还需要实际模型环境；本轮不能替代。',
      '- 因此不能据这两项报告把项目标为全部训练就绪，更不能宣称模型效果提升。','',
      '## 6. 本轮复现命令','',
      '```powershell',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.finalize_project1_sft',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.revalidate_tokenizer_alignment',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m pytest tests -q',
      '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.report_project1_pretraining_audit',
      '```','',
      '原始筛选入口仍为 `src/data/screen_openr1_sft.py`；本轮从固定源parquet回放已选轨迹，不进行全源重新抽样。',
      '完整测试日志：`reports/project1_pretraining_audit_pytest.log`。']
    test_log=Path('reports/project1_pretraining_audit_pytest.log')
    if test_log.exists():
        summary=test_log.read_text(encoding='utf-8').strip().splitlines()
        if summary and 'passed' in summary[-1]:lines+=['','实际测试结果：`'+summary[-1]+'`。']
    Path('reports/project1_sft_pretraining_audit_20261023.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('SFT/Tokenizer audit report saved; source teacher all-verified ratio:',before_full/n,'->',after_full/after)


if __name__=='__main__':run()
