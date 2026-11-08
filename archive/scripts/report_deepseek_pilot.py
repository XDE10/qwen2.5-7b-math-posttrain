"""Summarize actual paid pilot evidence without changing acceptance decisions."""
import json
import math
import re
from collections import Counter
from pathlib import Path
from scripts.prepare_deepseek_pilot import ROOT,OUT,file_sha
from src.synthesis.deepseek_pilot import dump

R=ROOT/'reports/deepseek_pilot_20261023'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def wilson(k,n):
    if not n:return None
    z=1.96;p=k/n;d=1+z*z/n
    a=(p+z*z/(2*n))/d;b=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [a-b,a+b]


def run():
    summary=load(R/'summary.json');results=load(R/'solution_results.json');ledger=load(R/'api_cache/ledger.json')
    refs={r['sample_id']:r for r in load(OUT/'verifier_references.json')}
    final=[];failures=Counter();attempts=[];patterns=[]
    for r in results:
        if 'attempts' not in r:failures['job_failed']+=1;continue
        attempts+=r['attempts']
        a=next(a for a in r['attempts'] if a['attempt']==r['selected_attempt']);final.append((r,a))
        if not a['math_verify']['correct']:failures['answer_not_verified']+=1
        if a['answer_tokens']<500:failures['below_500_tokens']+=1
        if a['answer_tokens']>1500:failures['above_1500_tokens']+=1
        if a['total_tokens']>3800:failures['context_above_3800']+=1
        if not a['structure_ok']:failures['wrapper_or_final_structure']+=1
        if a['finish_reason']!='stop':failures['non_stop_finish']+=1
        if not a['quality']['automatic_pass']:failures['noise_signal']+=1
        if r['process_review'].get('verdict')!='pass':failures['process_review_not_pass']+=1
        if re.search('translate|translation result|retain the original',refs[r['sample_id']]['problem'],re.I):
            patterns.append({'sample_id':r['sample_id'],'flag':'source_contains_translation_instruction'})
    by_type={}
    for kind in sorted({r['problem_type'] for r,a in final}):
        group=[(r,a) for r,a in final if r['problem_type']==kind]
        by_type[kind]={'n':len(group),'answer_verified':sum(a['math_verify']['correct'] for r,a in group),
            'automatic_pass':sum(r['automatic_pass'] for r,a in group),'silver_pair_eligible':sum(r['silver_pair_eligible'] for r,a in group)}
    balances={}
    before=load(R/'balance_before.json');after=load(R/'balance_after.json')
    if 'balance_infos' in before and 'balance_infos' in after:
        previous={r['currency']:float(r['total_balance']) for r in before['balance_infos']}
        balances={r['currency']:round(previous[r['currency']]-float(r['total_balance']),8) for r in after['balance_infos'] if r['currency'] in previous}
    details={'solution_jobs':len(results),'generation_attempts':len(attempts),
        'any_attempt_answer_verified':sum(any(a['math_verify']['correct'] for a in r.get('attempts',[])) for r in results),
        'first_attempt_answer_verified':sum(r.get('attempts',[{}])[0].get('math_verify',{}).get('correct',False) for r in results),
        'final_attempt_answer_verified':sum(a['math_verify']['correct'] for r,a in final),
        'final_under_500_but_answer_verified':sum(a['answer_tokens']<500 and a['math_verify']['correct'] for r,a in final),
        'automatic_pass':summary['automatic_pass'],'silver_pair_candidates':summary['silver_pair_candidates'],
        'process_verdicts':dict(Counter(r['process_review'].get('verdict','invalid') for r,a in final)),
        'failures_nonexclusive':dict(failures),'per_type':by_type,
        'final_math_verify_status':dict(Counter(a['math_verify']['status'] for r,a in final)),
        'raw_mean_answer_tokens':sum(r['original_answer_tokens'] for r,a in final)/len(final),
        'final_mean_answer_tokens':sum(a['answer_tokens'] for r,a in final)/len(final),
        'account_balance_delta':balances,'balance_note':'Observed account delta, could include other concurrent account activity or delayed billing.',
        'usage_price_range_usd':[sum(x.get('usage_cost_peak_usd',0) for x in ledger)/2,sum(x.get('usage_cost_peak_usd',0) for x in ledger)],
        'returned_models':dict(Counter(x.get('returned_model','unknown') for x in ledger if x['state']=='complete')),
        'source_instruction_flags':patterns,'human_review_completed':False,'training_approved':False}
    dump(R/'analysis.json',details)
    # Paired files contain the identical accepted intersection, explicitly silver/pilot.
    pairs=load(R/'paired_silver_candidates.json')
    for name,field in [('sft_pilot_raw','raw_solution'),('sft_pilot_compact','compact_solution')]:
        with (R/(name+'.jsonl')).open('w',encoding='utf-8') as f:
            for p in pairs:
                f.write(json.dumps({'sample_id':p['sample_id'],'problem':p['problem'],'reference_answer':p['reference_answer'],
                    'solution':p[field],'generation_mode':'source_openr1' if field=='raw_solution' else p['generation_mode'],
                    'status':'pilot_silver_pending_review','training_approved':False},ensure_ascii=False)+'\n')
    dev=load(R/'calibration_dev_metrics.json');audit=load(R/'frozen_acceptance_metrics.json')
    lines=['# DeepSeek 训练侧难度校准与 50 题盲生成 Pilot 实测', '',
        '日期：2026-10-23。真实付费 API 已执行；没有启动 GPU，没有扩展到 1500 题。', '',
        '## 执行边界与划分', '',
        '35 道难度锚点、50 道校准开发题、50 道冻结验收题，均从 math_train 抽取，按科目与 Level 分层；三个集合互斥，当前规则下近重复也排除。两个 Level ? 源记录不进入等级校准。',
        '另取 coverage 方案的 50 道 OpenR1 题，代数15、几何10、数论10、组合8、不等式5、逻辑2；长度桶分别14/19/17。它是分层 pilot，不是1500条的简单随机无偏估计。',
        '生成器只收到原题与统一格式要求；未发送原解答、标准答案、估计难度或判分反馈。难度请求只含已知训练锚点等级，不含待测题等级。',
        '非思考模式下请求可检查的书面解答。<think> 是格式约定，不能证明内部推理真实。', '',
        '## 难度校准结果', '']
    for name,m in [('校准开发集',dev),('冻结验收集',audit)]:
        lines += [f"- {name}：有效 {m['valid']}/{m['n']}，精确一致 {m['exact_agreement']:.1%}，±1 级 {m['within_one']:.1%}，平均绝对误差 {m['mae_valid']:.2f}；低等级误标为高等级 {m['low_to_high_count']}/{m['low_gold_n']}。"]
    high_recall=sum(audit['confusion_gold_rows_pred_columns'][i][j] for i in range(2,5) for j in range(2,5))/30
    lines += ['',f"冻结集真实 Level 3～5 的识别召回为 {high_recall:.1%}。低→高误标比例的 Wilson 95% 区间为 {wilson(audit['low_to_high_count'],audit['low_gold_n'])}，仅作小样本参考；分层抽样不代表源总体比例。",
        f"预登记门槛：精确≥60%、±1级≥90%、低→高误标≤20%、结构合法100%。实际 gate={summary['difficulty_gate_passed']}。",
        '本轮不调整提示后重新使用同一冻结集宣称独立验证；未通过时不批量标注候选，也不据此控制中高难度配比。此结果说明本版模型/锚点/提示组合不足，不代表所有 LLM 难度标注都不可行。', '',
        '## 盲生成与验证', '',
        f"50 题实际生成 {len(attempts)} 次，每题至多3次，无答案反馈。首次最终答案通过 {details['first_attempt_answer_verified']}/50，选定末次/成功次答案通过 {details['final_attempt_answer_verified']}/50。",
        f"至少一次最终答案通过的题共 {details['any_attempt_answer_verified']}/50。失败题按预定规则保留末次输出，不事后挑选最有利结果；因此末次通过数可能低于首次。选定输出的判分状态：{json.dumps(details['final_math_verify_status'],ensure_ascii=False)}；解析失败不直接等于数学错误。",
        f"满足答案、长度、格式和噪声门槛 {summary['automatic_pass']}/50；再加独立模型过程审查及原轨迹答案复验后，共同银标候选交集 {summary['silver_pair_candidates']}/50。",
        f"原轨迹平均 {details['raw_mean_answer_tokens']:.1f} Qwen tokens，选定输出平均 {details['final_mean_answer_tokens']:.1f}。两者包含失败样本，不能把这一差异当有效训练集压缩率或训练收益。",
        f"最终答案正确但短于500 tokens：{details['final_under_500_but_answer_verified']} 条。过程审查结果：{json.dumps(details['process_verdicts'],ensure_ascii=False)}。",
        '答案由隔离子进程中的 Math-Verify 0.9.0 复验，超时/不能解析/多个冲突 boxed 均不放行；原解答也复验。过程由另一模型 deepseek-v4-pro 审查，只见题面和候选解答，不见参考答案或自动判分结论。这不是人工认证，也不保证逐步正确。', '',
        '失败原因可以重叠：']
    lines += [f'- {k}：{v}。' for k,v in failures.items()]
    lines += ['', '按题型结果：']
    lines += [f"- {k}：n={v['n']}，答案通过{v['answer_verified']}，自动门槛通过{v['automatic_pass']}，共同银标候选{v['silver_pair_eligible']}。" for k,v in by_type.items()]
    lines += ['', '## 费用与缓存', '',
        f"API 请求记录 {len(ledger)} 条，预留金额累计 {summary['reserved_usd']:.6f} USD，上限3 USD。按响应 usage 和官方峰/谷单价估算 {details['usage_price_range_usd'][0]:.6f}～{details['usage_price_range_usd'][1]:.6f} USD。",
        f"账户余额前后观测差额：{json.dumps(balances,ensure_ascii=False)}。这可能受并发账号活动或延迟扣费影响；不把估算单价当最终账单。",
        '官方价格：https://api-docs.deepseek.com/quick_start/pricing/ 。返回模型身份、request ID、usage、请求正文和响应已记录；密钥未写入文件。',
        '每次派发前持久化预算预留；收到响应先缓存再判分。若进程在派发期间退出，reserved 状态默认拒绝自动重发，防止不确定重复计费；错误请求也不返还预留预算。', '',
        '## 额外的 Codex 数学抽查', '',
        '本会话对全部4条银标候选和4条失败/不确定案例进行了定向数学复核，见 codex_spot_review.json；这是AI定向抽查，不是人类审核或总体精度估计。',
        '发现独立API过程审核器也会误判：椭圆 x²/16+y²/9=1 第一象限的点满足 x²+y²=9+7cos²θ>9，确实在半径3的圆外；审核器声称点在圆内是错误的。原解答的面积下界27/4可成立。该样本仍有本轮格式/长度问题，本轮没有据抽查修改冻结准入结果。',
        '已通过候选也仍有重复表述：数学正确不等于示范足够精炼；不要把银标自动升级成最终SFT成品。', '',
        '## 数据对照与下一步', '',
        '共同候选输出为 sft_pilot_raw.jsonl 与 sft_pilot_compact.jsonl，题目ID一一对应，training_approved=false；不混入既有正式数据。原轨迹过程质量和双方最终审核仍需完成。',
        '未将超短但正确解答填充至500 tokens，也未机械截断超长回答。若下轮改变长度下限或格式，先版本化规则，再独立评测；不得把本轮失败事后改名为通过。',
        '题面中含翻译请求等元指令的记录已留证。生成器虽然声明题面是数据，仍需源题意审核；不能只靠 system 提示防止源数据中的指令干扰。',
        '下一步应先审核失败分布、修订训练侧源题清洗/输出格式和合理长度准入。难度标注需在校准开发集改进并使用新的独立验收集，不进入1500题批量生产。GPU预算与训练收益尚未验证。', '',
        '## 复现', '',
        '`python -m scripts.prepare_deepseek_pilot` 冻结/核验划分；`python -m scripts.run_deepseek_pilot` 仅 dry-run；设置环境变量 DEEPSEEK_API_KEY 后加 `--live` 才付费；`python -m scripts.report_deepseek_pilot` 重建本报告。',
        '依赖和本轮产物哈希见 verifier_dependencies.json 与 delivery_manifest.json；7项专用测试检查请求隔离、预算、恢复、难度指标与数学判分。']
    (R/'执行报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[p for p in R.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']
    dump(R/'delivery_manifest.json',{'files':{str(p.relative_to(ROOT)):file_sha(p) for p in files},
        'code':{str(p):file_sha(ROOT/p) for p in [Path('src/synthesis/deepseek_pilot.py'),Path('scripts/prepare_deepseek_pilot.py'),
            Path('scripts/run_deepseek_pilot.py'),Path('scripts/math_verify_pilot_worker.py'),Path('scripts/report_deepseek_pilot.py')]},
        'original_data_unchanged':True,'training_approved':False})
    print(json.dumps(details,ensure_ascii=False))

if __name__=='__main__':run()
