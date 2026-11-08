"""Supplement v3 with typed-answer isolation and reproducible readable data report.

No API call, no changes to frozen policy/results, no automatic bulk generation.
"""
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from src.synthesis.deepseek_pilot import dump
from scripts.compact_distillation_v3 import ROOT,DATA,PILOT,REPORT,read,file_sha,validate_manifest


def strict_check(text,reference):
    try:
        p=subprocess.run([sys.executable,'-m','scripts.math_verify_v3_strict_worker'],cwd=ROOT,
            input=json.dumps({'output':text,'reference':reference}),capture_output=True,text=True,encoding='utf-8',timeout=20)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:return {'correct':False,'status':'verifier_error','error_type':type(e).__name__}


def run():
    validate_manifest(DATA);validate_manifest(PILOT)
    results=read(REPORT/'results.json');refs={r['sample_id']:r for r in read(PILOT/'verifier_references.json')}
    strict_path=REPORT/'typed_answer_review.json'
    worker_hash=file_sha(ROOT/'scripts/math_verify_v3_strict_worker.py')
    cache=read(strict_path) if strict_path.exists() else None
    if cache:
        assert cache['worker_sha256']==worker_hash
        assert cache['results_sha256']==file_sha(REPORT/'results.json')
        checked=cache['rows']
    else:
        checked=[]
        def one(r):
            ref=refs[r['sample_id']];text=r.get('admission',{}).get('text','')
            return {'sample_id':r['sample_id'],
                'compact':strict_check(text,ref['reference_answer']),
                'raw':strict_check(ref['solution'],ref['reference_answer']),
                'frozen_common_candidate':r['common_candidate'],
                'admission_accepted':r.get('admission',{}).get('accepted_candidate',False)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(one,r) for r in results]
            for f in as_completed(futures):checked.append(f.result())
        checked.sort(key=lambda r:r['sample_id'])
        dump(strict_path,{'worker_sha256':worker_hash,'results_sha256':file_sha(REPORT/'results.json'),
            'scope':'post-discovery conservative type gate, not rewritten frozen success rate','rows':checked})
    common=read(REPORT/'common_candidates.json');byid={r['sample_id']:r for r in checked}
    review_path=REPORT/'ai_process_review.json'
    ai=read(review_path) if review_path.exists() else None
    ai_index={}
    if ai:
        for r in ai.get('reviews',ai.get('candidate_reviews',[])):
            ai_index[r['sample_id']]=r
    reviewed=[]
    for pair in common:
        c=byid[pair['sample_id']];a=ai_index.get(pair['sample_id'])
        typed=c['compact'].get('correct',False) and c['raw'].get('correct',False)
        verdict=a.get('decision',a.get('verdict','pending')) if a else 'pending'
        reviewed.append({**pair,'typed_answer_review':c,'ai_process_review':a or {'decision':'pending'},
            'candidate_status':'isolated_by_typed_answer_gate' if not typed else
                ('ai_pass_pending_human' if verdict=='pass' else 'process_review_'+verdict),
            'human_review':'pending','training_approved':False})
    dump(REPORT/'reviewed_common_candidates.json',reviewed)
    summary=read(REPORT/'summary.json');pool=read(DATA/'summary.json')
    lengths=[r['admission']['answer_tokens'] for r in results if 'admission' in r]
    correct_short=sum(r.get('admission',{}).get('math_verify',{}).get('correct',False) and
                      r['admission']['answer_tokens']<150 for r in results)
    typed_pairs=sum(c['admission_accepted'] and c['compact'].get('correct',False) and c['raw'].get('correct',False) for c in checked)
    audit={'frozen_pilot_summary':summary,'typed_common_candidates':typed_pairs,
        'typed_compact_statuses':dict(Counter(c['compact']['status'] for c in checked)),
        'typed_raw_statuses':dict(Counter(c['raw']['status'] for c in checked)),
        'correct_answer_but_below_150_qwen_tokens':correct_short,
        'answer_tokens_min':min(lengths),'answer_tokens_max':max(lengths),
        'ai_review_available':bool(ai),'reviewed_candidate_statuses':dict(Counter(r['candidate_status'] for r in reviewed)),
        'human_approval':'pending','bulk_generation_enabled':False,'training_approved':False}
    dump(REPORT/'final_data_status.json',audit)
    warnings=read(REPORT/'source_review_flags.json') if (REPORT/'source_review_flags.json').exists() else None
    lines=['# 项目一 v3：文本候选池净化与30题冻结验收', '',
        '本轮只做数据工作；没有编写GPU训练脚本、启动训练或批量API生成。人类确认前不封板、不扩产。', '',
        '## 1. 基底池净化', '',
        f"从固定OpenR1回答≤3000候选中保留{pool['retained_from_old_1500']}道原池题，隔离{pool['old_1500_rule_isolated']}道并按原六类配额补{pool['replacement_count']}道，最终{pool['actual']}道，配额缺口{pool['quota_shortfalls']}。",
        '完整数据：data/project1_textonly_v3/native_candidates.jsonl；筛除题面及理由、真实源位置二次检查、前后哈希和manifest同目录保存。真实tokenizer重新检查原生回答500～3000及完整上下文≤3800。',
        '正则规则保守隔离figure/diagram/图示/图表/外部媒体等；会误杀部分完全给定的图函数题，也可能漏掉不显式提图的缺条件题。因此只称“通过文本筛查”，不称100%语义自包含。',
        '扫描代表题面和实际选中轨迹源行分别检查，避免同一规范化题目不同源行措辞被混用。清洗后重新做heldout/内部词面匹配，未发现当前阈值命中；不保证语义零污染。', '',
        '## 2. 预登记协议与费用', '',
        '仅deepseek-flash，thinking.type=disabled、temperature=0.3、max_tokens=1400、JSON模式。返回status/solution/final_answer，本地将教师自生成字段序列化为<think>书面解答</think>与boxed；标准答案从不进入生成请求。',
        '这轮验证优化后的单份书面解答提示，而非thinking开关对照，不能据此判断开/关thinking哪种更正确。150-token下限仍是用户指定的实验窗口，不是正确性证据；短证明不填充。',
        '冻结规则包含LaTeX显示定界符兼容、长段落重复/分离12gram重复信号、Math-Verify、150～1500 Qwen tokens、stop和完整上下文≤3800。测试通过后先封存源码/输入/policy哈希，再派发。',
        '30新题与此前50+40生成题及135锚点/校准题均无当前规范化/Jaccard≥0.85命中；固定分层配额9/6/6/5/3/1，不是总体随机抽样。',
        f"每题只生成一次以适应<$0.10要求；全轮预算硬上限$0.095，派发前最坏预留$0.069465，实际累计预留${summary['reserved_usd']:.6f}。完成{summary['generation_requests']}个唯一请求，状态{summary['request_states']}。",
        f"响应usage按当前峰价估算${summary['usage_estimated_peak_usd']:.8f}；谷价为其一半。预留不是账单，估算也不是最终扣费。实际墙钟{summary['wall_seconds']:.2f}秒。", '',
        '## 3. 冻结结果（不事后改写）', '',
        f"结构格式通过{summary['json_format_pass']}/30；答案验证通过{summary['answer_verified']}/30；自动准入{summary['automatic_candidates']}/30；双方答案均通过原验证器的同题候选{summary['common_pairs']}/30。",
        f"预登记自动准入≥70%、结构格式≥90%；engineering_gate_passed={summary['engineering_gate_passed']}。没有扩大到1500题。",
        f"全部输出Qwen长度范围{min(lengths)}～{max(lengths)}；{correct_short}条答案通过但短于150。拒绝信号{summary['failure_signals']}。",
        'JSON字段类型不符、JSON语法错误或缺少完整书面解答均保留失败；本轮不在看过结果后放宽类型/长度再重标为冻结通过。新提示减少重复是观察，不是与v2同题同预算的因果对照。', '',
        '## 4. 逐题过程复核与答案验证器反例', '',
        '逐题AI复核发现：某参考答案的指数与题面不符，Math-Verify 0.9.0仍把不同的关系表达式判为等价。单纯final-answer verify=true不能解释为严格金标。相关原始响应和反例留存在ai_process_review.json/md。',
        '独立补充scripts/math_verify_v3_strict_worker.py，不修改冻结旧worker：关系式/不等式转任务相关人工核查；多个不同boxed可能表示一个正确集合，转集合/元组人工核查，不把不支持误报成数学错误。数字、分数等仍独立验证。',
        f"这项发现后的保守类型重审留下{typed_pairs}对自动共同候选；它是补充隔离检查，不修改上一节冻结统计。关系及多boxed复验状态见typed_answer_review.json。",
        '已逐题复核全部12对（原生与精炼双方），AI通过、隔离和不确定分开；AI不是人类签字，所有training_approved仍false。' if ai else '全部12对正在做逐题AI过程复核；人工审核仍pending。', '',
        '## 5. 人工确认材料与下一步', '',
        '[三项目人工审核总览](../data_human_review_pack.md)含固定规则挑选的10对本轮原生/精炼样本及失败样本证据，不隐去隔离案例。不能用预览样本代表全量质量。',
        '[项目三200题原型](../project3_perception_prototype/prototype_200.md)及[30条双定位原图关联核查单](../project3_perception_review/manual_link_review_30.md)均未签字；visual_candidate不是已证明的image-only事实。',
        '后续先由用户查看样本、处理源题/答案与过程争议，再决定是否降低长度下限或另冻结新版本、是否做新验收和批量生成。本轮不额外派发新实验，不替用户签字。', '',
        '## 6. 本地复现', '',
        '```powershell',
        '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.compact_distillation_v3 prepare',
        '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.compact_distillation_v3 run',
        '# 已执行付费命令：run --live --key-prompt；复现报告不需要再次付费。',
        '.\\tiny_posttrain\\.venv\\Scripts\\python.exe -m scripts.report_compact_v3',
        '```', '',
        '冻结policy/results不能覆盖调参；同一live命令只复用已有缓存。任何新配置需新版本和用户确认。']
    if warnings:
        lines += ['', '## 补充源元数据待审队列', '',
            f"对净化池1500题再查译文声明、空SOLUTION/Markdown标题、参考答案空单位/缺左端等，{warnings['flagged_questions']}题有至少一个待审信号：{warnings['counts']}。",
            '这些信号不表示全部题目数学错误；不静默改参考答案，不覆盖冻结池，不把尚未处置的队列签成合格。详见source_review_flags.json；复现python -m scripts.audit_v3_source_metadata。']
    (REPORT/'执行报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(REPORT/'final_integrity_manifest.json',{'files':{str(p.relative_to(REPORT)):file_sha(p) for p in REPORT.rglob('*') if p.is_file() and p.name!='final_integrity_manifest.json'},
        'bulk_generation_enabled':False,'training_approved':False,'human_approval':'pending'})
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
