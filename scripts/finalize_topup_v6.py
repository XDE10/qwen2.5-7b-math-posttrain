"""Seal exact observed counts and input checks after both API owners finish."""
import subprocess,sys,re,zipfile
from collections import Counter
from scripts.topup_common import *
from src.data.approved_training_inputs import load_approved,model_view
from src.data.perception_v4 import token_key
from src.teachers.opd_contract import render_student_prompt

def run():
    assert (REPORT/'budget_after_reserves.json').exists(),'API work still active'
    from scripts.reparse_topup_geometry import run as reparse
    reparse()
    from scripts.export_topup_v6 import run as export
    if not (DATA/'release/manifest.json').exists():export()
    from scripts.configure_topup_training import run as configs
    configs()
    release=DATA/'release';m=read(release/'manifest.json');s=m['summary'];ledger=read(BUDGET/'ledger.json');before=read(REPORT/'before_ledger.json')
    assert ledger[:len(before)]==before and all(r['state']!='reserved' for r in ledger)
    budget={'cap_total_usd':8,'all_dispatch_attempts':len(ledger),'new_dispatch_attempts':len(ledger)-len(before),'completed_peak_usage_usd':sum(r.get('charged_upper_usd',0) for r in ledger if r['state']=='complete'),'unknown_error_retained_reserve_usd':sum(r['reserved_usd'] for r in ledger if r['state']!='complete'),'committed_upper_usd':sum(r.get('charged_upper_usd',r['reserved_usd']) for r in ledger),'is_actual_account_debit':False,'live_pending_requests':0,'unknown_error_dispatches':sum(r['state']!='complete' for r in ledger)}
    assert budget['committed_upper_usd']<=8;dump(REPORT/'budget_final.json',budget)
    from transformers import AutoTokenizer
    student=AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-1.5B-Instruct',local_files_only=True);math=AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True);checks={}
    for name in m['approved_training_files']:
        rs=load_approved(release/'manifest.json',name);lengths=[]
        for r in rs:
            if name.startswith('project2'):
                view=model_view(r,'opd');ids=render_student_prompt(student,view['input']['prompt'],view['input']['domain']);lengths.append(len(ids))
            elif name.startswith('project1'):
                v=model_view(r,'math_sft');messages=[{'role':'system','content':r'Please reason step by step, and put your final answer within \boxed{}.'},{'role':'user','content':v['input']['problem']}]
                prefix=math.apply_chat_template(messages,tokenize=True,add_generation_prompt=True);full=math.apply_chat_template(messages+[{'role':'assistant','content':v['target']}],tokenize=True,add_generation_prompt=False)
                assert full[:len(prefix)]==prefix and len(prefix)<len(full)<=3800
                if 'compact' in name:assert 150<=len(math.encode(r['solution'],add_special_tokens=False))<=1500
                lengths.append(len(full))
            else:
                v=model_view(r,'perception_sft');assert sha(Path(v['input']['image_path']))==r['image_sha256'];assert r['split']=='train' and r['certified_positive_tokens']
                for t in r['certified_positive_tokens']+r['certified_negative_tokens']:token_key(t)
                assert not re.search(r'source_radius_id|radius_\d',v['target'])
        checks[name]={'rows':len(rs),'max_tokens':max(lengths) if lengths else None,'passed':True}
    p1raw=rows(release/'project1_raw_paired.jsonl');p1compact=rows(release/'project1_compact_paired.jsonl');assert [r['sample_id'] for r in p1raw]==[r['sample_id'] for r in p1compact]
    levels={r['sample_id']:r.get('level','unknown') for r in rows(ROOT/'data/math/math_train.jsonl')+rows(ROOT/'data/math/math_dev.jsonl')}
    mathtrain=rows(release/'project2_math_train.jsonl');mathdev=rows(release/'project2_math_dev.jsonl');codetrain=rows(release/'project2_code_train.jsonl');codedev=rows(release/'project2_code_dev.jsonl')
    for train,dev in [(mathtrain,mathdev),(codetrain,codedev)]:
        assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in dev};assert all(not r['training_approved'] and r['evaluation_approved'] for r in dev)
    p3train=rows(release/'project3_train_fact_pruned.jsonl');p3val=rows(release/'project3_validation_fact_pruned.jsonl');assert not {r['original_id'] for r in p3train}&{r['original_id'] for r in p3val}
    old=read(ROOT/'reports/data_release_preparation_20261024/delivery_manifest.json');expected_changes={'reports/data_release_preparation_20261024/shared_api_cache/ledger.json','src/training/approved_hf.py'};preserved=0
    for name,h in old['files'].items():
        if name in expected_changes:continue
        assert sha(ROOT/name)==h,('Unexpected old artifact change',name);preserved+=1
    dump(REPORT/'approved_input_preflight.json',{'files':checks,'paired_ids_equal':len(p1raw),'old_artifacts_unchanged':preserved,'intentional_changes':sorted(expected_changes),'original_ledger_prefix_unchanged':True,'gpu_runs':0,'image_processor_and_A100_smoke_not_executed':True})
    local=read(REPORT/'p1_local/verification_summary.json');p1=read(REPORT/'p1_summary.json');v=read(REPORT/'p3_reparsed_summary.json');source=read(DATA/'project1/summary.json')
    relation_counts=Counter(f['fact']['relation'] for r in p3train for f in r['certified_fact_evidence'] if f['token'] in r['certified_positive_tokens'])
    incidence_only=sum(all(t.startswith(('PointLiesOnLine(','PointLiesOnCircle(')) for t in r['certified_positive_tokens']) for r in p3train)
    summary={'release':s,'budget':budget,'p1_main':p1,'p1_main_local':local,'rl_official_levels':source['official_level_distribution'],'p2_math_train_levels':dict(Counter(levels.get(r['sample_id'],'not_MATH_levelled') for r in mathtrain)),'p3_final_review':v,'p3_training_relation_counts':dict(relation_counts),'p3_training_incidence_only_cases':incidence_only,'gold_precision':None,'no_gpu_run':True}
    dump(REPORT/'summary.json',summary)
    def link(p,t):return '['+t+'](<'+Path(p).resolve().as_posix()+'>)'
    lines=['# 严标准 Top-up Sampling 实测报告（2026-10-24）','',
        '用户将同一 API 账本硬上限从$5提高至$8；没有清零旧费用或失败请求预留。逐题批准是用户委托AI的工程准入，不是人工逐题签字或100%数学/视觉金标。','',
        '## 精确批准数量','',
        f"项目一同题A/B：**{s['project1_approved_pairs']}对**，旧213对加本轮{s['project1_new_approved']}对。raw/compact文件ID与题面完全一致；仅这些明确行获训练准入。",
        f"项目二数学：**{s['project2_math_train']} train / {s['project2_math_dev']} dev**；代码train **{s['project2_code_train']}**，code dev **{s['project2_code_dev']}**。dev的training_approved仍为false、evaluation_approved为true。",
        f"项目三：**{s['project3_train']} train / {s['project3_validation']} validation**通过题答检查并至少保留1条逐事实AI认证且通过本地接地/蕴含/解析检查的事实。未通过或无剩余事实的题保留在完整审核账本。",
        '目标是否达到：'+json.dumps(s['targets_met'],ensure_ascii=False)+'。','',
        '## 项目一与RL优先池','',
        '原1495题主批次与独立5题代数补充批次分开冻结。实际新源题1500；原生联合候选3000。原六类目标下只有95道新不等式题通过所有条件；用同标准代数题组成独立补充批次，最终配额530/300/300/225/95/50，未伪造100道不等式。原生候选全集没有被整体批准。',
        f"主批次1495道题、1495份已返回生成结果，保守本地共同交集{local['common_pairs']}对；双轨过程审核{p1['pair_review']}。传输失败的额外派发逐次记入总账，不冒充免费或语义重抽。旧验证函数的元数据target_questions=500为复用代码的历史常量，实际分母明确使用1495，未据此伪报通过率。",
        '准入仍为150~1500 Qwen tokens、完整格式/无重复、双方保守Math-Verify与同题两条完整轨迹的过程审核。源最短正确轨迹1200~3200，完整SFT上下文<=3800。源长度、竞赛来源、源多次采样非全对仅是难度代理，不能据此宣称官方中高难度；短于150也不等于数学错误或简单题。',
        '同题A/B同时改变了教师、生成提示词和轨迹长度，属于两种数据管线的比较；没有GPU结果，也不能单独归因于消除overthinking或宣称准确率/速度提升。',
        f"6520道纯文本MATH官方Level分布：{source['official_level_distribution']}；最初L3~L5优先池4830，扣除新增封存dev及其近重复后剩{s['rl_priority_excluding_new_dev']}。这是上卡筛3000题的候选，不是已经完成7B采样筛选。",'',
        '## 项目二补齐与新dev版本','',
        '保留旧逐题pass训练样本；新数学题来自未分配MATH候选，优先原生Level3~5。新dev先封存，既不从旧训练题回收，也不把未通过dev题改作训练。旧dev原始文件与所有被隔离ID均保留，新版本以显式替补ID及来源追踪，不能悄悄改写原dev。',
        '代码新题来自已静态审计候选并回读固定源参考代码核对AST；去重包含题面、family/body/name保护、官方代码测试集和旧dev保护。新函数题剔除原样/常量/纯包装并要求计算或控制逻辑。新code dev需全部源用例通过且常量mutant被拦住，排除多解构造题；旧stdin dev也补做同一门槛。',
        '第一次补抽仍有stdin train缺口，使用未进入任何dev的独立train备用池补审。额外dev备用池也先通过受限Docker、源测试与mutant门槛，再送AI；不对语义isolate/uncertain反复采样一个pass。源测试不是完备正确性证明，AST复杂度不是官方难度。',
        '批准数学训练集实际官方Level分布：'+json.dumps(summary['p2_math_train_levels'],ensure_ascii=False)+'。非MATH数据不伪造Level。','',
        '## 项目三事实修剪与V4奖励','',
        f"完整逐图审核/重解析统计：{v}。validation始终不参与训练或合成种子；test未用于本轮训练扩展。",
        '逐条剔除无可见字母接地、内部圆ID、圆身份歧义、题干显式/直接蕴含、不可解析或未认证事实；不因一个坏事实连坐其余合格事实。题答/方向冲突仍为整题阻断，ID742继续隔离。',
        '审阅输出冗余type=json_object只在有日志的侧车重解析中剥离；必须保留原始响应、实际stop finish、完整索引分区、逐事实证据，其他额外字段仍拒绝。无新API请求，没有重抽失败语义判决。',
        '独立像素/题干抽查发现ID265对角线交点事实已由题干推出，ID1053的已命名弧端点属于已命名圆也可从文字推出；确定性文字蕴含守卫已对全部缓存逐事实补剔除。AI审阅会错，因此报告不使用100%纯净、完整图像金标或人工精度等说法。',
        f"V4按认证正事实TP和认证常量属性位可证伪的FP计算有限范围Precision，Recall分母为认证正事实；未认证预测mask、去重、不当作FP。训练样本中{s['project3_cases_with_explicit_negatives']}题有可交叉支持的显式负事实。另可由认证Length(A,B)=3直接判同位数值4为FP；未知实体或未证不等的符号表达式仍mask。仅凭AI说没有标记不作否定；不能把有限Precision解释成全图防幻觉能力。格式/128事实上限及日志记录mask数量。",
        '已接入train_project3对应的sparse_certified_v4奖励，默认答案奖励权重1、感知F1权重0.2；私有认证/答案元数据不会拼入模型输入。仅批准新的train逐事实子集；完整类别、全图Precision或未认证范围的惩罚仍不启用。VLM SFT监督感知事实，没有虚构完整reasoning金标。','',
        f"训练事实关系分布：{dict(relation_counts)}；仅有点在线/点在圆关联事实的题有{incidence_only}道。关联事实确属部分视觉监督，但不冒充数值角长标注、官方难度或完整几何推理轨迹。",'',
        '## 费用与验证','',
        f"账本累计{budget['all_dispatch_attempts']}次派发，本轮新增{budget['new_dispatch_attempts']}次；已完成用量按峰值价格估算${budget['completed_peak_usage_usd']:.8f}，错误/未知费用保留${budget['unknown_error_retained_reserve_usd']:.8f}，合计上界**${budget['committed_upper_usd']:.8f} <= $8**，不是账户实扣差额。无仍在途请求；错误费用预留不代表后台仍在生成。",
        link(REPORT/'approved_input_preflight.json','实际tokenizer/掩码/分割/SHA预检')+'；'+link(REPORT/'pytest.log','CPU测试日志')+'；'+link(DATA/'release/manifest.json','唯一批准范围')+'。',
        '本轮没有GPU权重下载/加载、A100训练、GPU端到端Smoke或模型准确率/速度实测。CPU及接口检查不替代真实VL processor、最长样本显存、rollout与恢复验收。',
        link(DATA/'training_configs','新批准子集的训练配置目录')+'；'+link(ROOT/'project_blueprints/Topup_v6_感知奖励与运行说明.md','训练与奖励说明')+'。','']
    (REPORT/'执行报告.md').write_text(re.sub(r'\$(\d+(?:\.\d+)?)',r'USD \1','\n'.join(lines)),encoding='utf-8')
    print(json.dumps({'release':s,'budget':budget},ensure_ascii=False),flush=True)
if __name__=='__main__':run()
