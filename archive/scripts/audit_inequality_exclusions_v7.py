"""Read-only reason audit. No API, no quota or acceptance-rule changes."""
import json,re,statistics
from collections import Counter
from scripts.finalfill_common import *
from scripts.refill_project1_native_v5 import source_filter
from scripts.compact_distillation_v2 import replay
from src.synthesis.compact_v3 import screen_problem
from src.data.screening_io import ScreeningSimilarityIndex

def run():
    out=REPORT/'inequality_diagnosis';out.mkdir(exist_ok=True)
    exclusions=Counter();all_flags=Counter();allq=[];proposals=[];lengths=[];eligible_short=[];prior_ids=set()
    paths=['data/project1_native_v5/native_candidates.jsonl','data/topup_v6_20261024/release/project1_native_fresh1500_candidates_not_blanket_approved.jsonl','data/project1_batch500_20261024/verifier_references.jsonl']
    paths += [str(p.relative_to(ROOT)) for p in (ROOT/'data').glob('compact_distillation_v*/verifier_references.json')]
    paths += ['data/deepseek_pilot_20261023/'+n+'.json' for n in ['verifier_references','anchors','calibration_dev','frozen_acceptance']]
    for path in paths:
        rs=rows(ROOT/path) if path.endswith('.jsonl') else read(ROOT/path)
        prior_ids.update(r['sample_id'] for r in rs)
    with (ROOT/'reports/openr1_full_audit_final_20261023/questions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            q=json.loads(line)
            if q['problem_type']!='Inequalities':continue
            allq.append(q);ts=[t for t in q['trajectories'] if t.get('verified') and t.get('complete') and t.get('answer_tokens') is not None]
            reason=None;minimum=min((t['answer_tokens'] for t in ts),default=None)
            if minimum is not None:lengths.append(minimum)
            if not ts:reason='no_structurally_complete_source_correct_tokenized_trajectory'
            elif minimum<1200:reason='source_shortest_under1200_proxy_window_not_proven_easy'
            elif minimum>3200:reason='source_shortest_over3200_budget_window'
            else:
                valid=[t for t in ts if t['answer_tokens']==minimum and t.get('quality',{}).get('automatic_pass') and t.get('total_tokens',99999)<=3800]
                if not valid:reason='shortest_source_noise_flags_or_full_context_over3800'
                elif q.get('flag_conflict') or q.get('evaluation_overlap') or q.get('internal_near_duplicate'):reason='flag_conflict_or_preexisting_overlap'
                elif q['question_hash'] in prior_ids:reason='already_in_prior_native_or_API_pools_not_fresh'
                elif not screen_problem(q['problem'])['screen_pass']:reason='source_problem_keyword_scope_screen'
                else:
                    proposals.append({'question_hash':q['question_hash'],'problem_type':q['problem_type'],'source':q['source'],'trajectory':valid[0],'linked_train_level':q.get('linked_train_level')});reason='remaining_source_replay_candidates'
            exclusions[reason]+=1
            eligible_short.append({'sample_id':q['question_hash'],'source':q['source'],'problem':q['problem'],'shortest_correct_tokens':minimum,'first_exclusion_or_stage':reason})
    print('Inequality corpus stages',len(allq),dict(exclusions),'replay',len(proposals),flush=True)
    rr={r['sample_id']:r for r in replay(proposals)};detail=[];post=Counter()
    for p in proposals:
        r=rr[p['question_hash']];row={'sample_id':r['sample_id'],'problem':r['problem'],'solution':r['raw_solution'],'reference_answer':r['reference_answer']};a=source_filter(row)
        relational=bool(re.search(r'[<>≤≥≠=]|\\(?:leq?|geq?|neq?|lt|gt)\b',str(r['reference_answer'])))
        reasons=a['reasons'][:]
        if relational:reasons.append('relational_answer_outside_frozen_generic_MathVerify_admission_scope')
        tri=a['v4_triage'];all_flags.update(reasons);all_flags.update('quality:'+x for x in tri['quality_isolation_reasons']);all_flags.update('scope:'+x for x in tri['experiment_scope_exclusions']);all_flags.update('manual:'+x for x in tri['manual_review_flags'])
        label='would_pass_source_and_answer_type' if not reasons else 'requires_typed_inequality_verifier' if a['accepted'] and relational else 'other_source_or_structure_or_scope_rejection'
        post[label]+=1;detail.append({**row,'source':p['source'],'source_position':p['trajectory'],'shortest_correct_tokens':p['trajectory']['answer_tokens'],'source_filter':a,'relational_reference':relational,'combined_reasons':reasons,'category':label})
    # Actual preceding V6 paid funnel, not guessed from source lengths.
    refs={r['sample_id']:r for r in rows(ROOT/'data/topup_v6_20261024/project1/native_new1500.jsonl') if r['problem_type']=='Inequalities'}
    ver=[r for r in read(ROOT/'reports/topup_v6_20261024/p1_local/verification_results.json') if r['sample_id'] in refs]
    verdicts={r['sample_id']:r for r in rows(ROOT/'reports/topup_v6_20261024/p1_pair_review.jsonl')}
    paid={'source_questions':len(refs),'generation_complete':sum(r['status']=='complete' for r in ver),'compact_answer_verified':sum(r.get('admission',{}).get('math_verify',{}).get('correct',False) for r in ver),'local_common':sum(r.get('common_candidate',False) for r in ver),'process_review':dict(Counter(verdicts[r['sample_id']]['verdict'] for r in ver if r['sample_id'] in verdicts)),'overlapping_failure_signals':dict(Counter(e for r in ver for e in r.get('admission',{}).get('errors',[])))}
    # Length-only counts are not process-correctness or official difficulty.
    summary={'all_unique_inequality_questions':len(allq),'source_first_stage_exclusive_counts':dict(exclusions),'remaining_replay_candidates':len(proposals),'post_replay_categories':dict(post),'overlapping_source_signals':dict(all_flags),'source_shortest_length_median':statistics.median(lengths) if lengths else None,'previous_v6_paid_inequality_funnel':paid,'api_calls':0,'current_rule_or_approval_changed':False,'warning':'Short source or compact trajectory is not proof of an easy question. Relational-answer unsupported means verifier-scope limitation, not wrong mathematics.'}
    dump(out/'summary.json',summary);write(out/'source_stage_audit.jsonl',eligible_short);write(out/'remaining_candidate_details.jsonl',detail);write(out/'previous_paid_funnel_details.jsonl',ver)
    (out/'原因审查.md').write_text('# 不等式不足：实际逐层筛查原因\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n\n所有记录只读；没有API调用、没有放宽门槛或改写旧结论。详见同目录逐题原因与原题/答案/轨迹。\n',encoding='utf-8');print(summary,flush=True)
if __name__=='__main__':run()
