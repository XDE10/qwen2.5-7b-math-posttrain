"""Export only explicitly delegated AI-pass subsets; never certify whole pools."""
import json,re
from pathlib import Path
from collections import Counter
from scripts.compact_distillation_v2 import ROOT,read,write_jsonl,file_sha
from scripts.review_project2_deepseek import jsonl
from src.synthesis.deepseek_pilot import dump

OUT=ROOT/'data/delegated_ai_releases_20261024';REPORT=ROOT/'reports/data_release_preparation_20261024'


def run():
    if (OUT/'manifest.json').exists():raise FileExistsError('AI release exists; preserve it and use another version')
    OUT.mkdir(exist_ok=True)
    review_policy=read(ROOT/'reports/math_pair_delegated_review_20261024/policy.json')
    for p,h in review_policy['inputs'].items():assert file_sha(ROOT/p)==h,p
    p2_policy=read(ROOT/'data/project2_ai_review_20261024/manifest.json')
    for p,h in p2_policy['inputs'].items():assert file_sha(ROOT/p)==h,p
    reviews=jsonl(ROOT/'reports/project2_ai_review_20261024/results.jsonl');lookup={r['record_key']:r for r in reviews}
    evidence=read(ROOT/'reports/project2_ai_review_20261024/independent_counterexample_checks.json')
    disagreements={evidence['record_key']}
    accepted={};rejected=[];all_dev=[]
    for domain,split,path in [('math','train','data/project2_code/math_train_prompts.jsonl'),('code','train','data/project2_code_v3_dev_verified/train_prompts.jsonl'),
        ('math','dev','data/math/math_dev.jsonl'),('code','dev','data/project2_code_v3_dev_verified/dev_verification.jsonl')]:
        rows=jsonl(ROOT/path);passed=[]
        for r in rows:
            k=f"{domain}/{split}/{r['sample_id']}";v=lookup.get(k,{'verdict':'not_reviewed'})
            status=v['verdict'];local_flags=[]
            if k in disagreements:status='uncertain_independent_reviewer_disagreement'
            if domain=='code' and r['interface']=='stdin_stdout' and re.search(r"(?i)you (?:do not|don't) need to read input",r['problem']) and not re.search(r'(?i)(?:input is given|first line (?:of|contains)|-----input-----)',r['problem']):
                status='uncertain_function_stdio_contract';local_flags.append('function-only instruction plus stdin wrapper without explicit general parsing contract')
            record={**r,'delegated_review_status':status,'review_method':'user_delegated_DeepSeek_AI_plus_recorded_deterministic_checks',
                'itemwise_human_review_performed':False,'approval_evidence_request_id':v.get('request_id'),'local_review_flags':local_flags,
                'training_approved':status=='pass' and split=='train'}
            if split=='dev':
                record['use_for_training']=False;record['evaluation_specification_review_status']=status
                if k=='math/dev/math/train/04117':
                    record['independent_answer_check']='sqrt(3) verified by Cauchy bound and explicit equality point; original answer unchanged'
                    record['auxiliary_reference_solution_status']='isolated_displayed_radical_typo; not a wrong dev final answer'
                if domain=='code' and re.search(r'(?is)(?:multiple|many|not unique).{0,100}(?:solution|answer|ordering)|(?:output|print|return)\s+any\b|any of the reorderings',r['problem']):
                    record['evaluation_oracle_review_required']='possible constructive/multiple-valid-output task; literal stdout equality is not a sufficient model grader'
                if r['sample_id']=='code/vfc_19821':
                    record['evaluation_oracle_review_required']='constructive permutation output requires semantic checker; source-reference pass is not a general student-output oracle'
                all_dev.append({'domain':domain,**record})
            elif record['training_approved']:passed.append(record)
            else:rejected.append({'domain':domain,'split':split,'sample_id':r['sample_id'],'review_status':status,'issues':v.get('issues',[]),'local_flags':local_flags})
        if split=='train':
            accepted[domain]=passed;write_jsonl(OUT/f'project2_{domain}_train_ai_approved.jsonl',passed)
    write_jsonl(OUT/'project2_fixed_dev_audit.jsonl',all_dev);write_jsonl(OUT/'project2_train_review_queue.jsonl',rejected)
    pairreviews=read(ROOT/'reports/math_pair_delegated_review_20261024/reviews.json');pairlookup={r['sample_id']:r for r in pairreviews}
    new=read(ROOT/'reports/project1_batch500_20261024/common_candidates.json');old=read(ROOT/'reports/compact_distillation_v4_20261024/common_candidates.json')
    pairrows={r['sample_id']:r for r in new+old};raw=[];compact=[];pairqueue=[]
    sources={r['sample_id']:r for r in jsonl(ROOT/'data/project1_native_v5/native_candidates.jsonl')}
    sources.update({r['sample_id']:r for r in read(ROOT/'data/compact_distillation_v4_20261024/verifier_references.json')})
    for v in pairreviews:
        r=pairrows[v['sample_id']]
        if v['verdict']!='pass':pairqueue.append({'sample_id':v['sample_id'],'cohort':v['cohort'],'review':v});continue
        base={'sample_id':r['sample_id'],'pair_id':r['sample_id'],'problem':r['problem'],'reference_answer':r['reference_answer'],'problem_type':r['problem_type'],
            'split':'train','cohort':v['cohort'],'training_approved':True,'review_method':'user_delegated_DeepSeek_AI_and_conservative_MathVerify',
            'source_revision':sources[r['sample_id']]['source_revision'],'source_position':sources[r['sample_id']].get('source_position'),
            'itemwise_human_review_performed':False,'approval_evidence_request_id':v.get('request_id')}
        raw.append({**base,'solution':r['raw_solution'],'arm':'raw'});compact.append({**base,'solution':r['compact_solution'],'arm':'compact'})
    assert {r['pair_id'] for r in raw}=={r['pair_id'] for r in compact}
    write_jsonl(OUT/'project1_raw_paired_ai_approved.jsonl',raw);write_jsonl(OUT/'project1_compact_paired_ai_approved.jsonl',compact)
    dump(OUT/'project1_pair_review_queue.json',pairqueue)
    geometry_manifest=ROOT/'data/project3_grounding_ai_review_20261024/manifest.json'
    geometry=read(geometry_manifest)
    gp=geometry_manifest.parent/'approved_train_prototype.jsonl';assert file_sha(gp)==geometry['files']['approved_train_prototype.jsonl']
    grounded=jsonl(gp);write_jsonl(OUT/'project3_perception_prototype_ai_approved.jsonl',grounded)
    summary={'release_kind':'itemwise_delegated_AI_engineering_subsets; not original full-pool or human-gold signoff',
        'project2_approved_train_counts':{k:len(v) for k,v in accepted.items()},
        'project2_code_interfaces':dict(Counter(r['interface'] for r in accepted['code'])),
        'project2_original_dev_ids_preserved':len(all_dev),'project2_dev_review_statuses':dict(Counter(r['delegated_review_status'] for r in all_dev)),
        'project2_original_2500_plus2500_full_train_approved':len(accepted['math'])==2500 and len(accepted['code'])==2500,
        'project1_ai_approved_paired_count':len(raw),'project1_cohorts':dict(Counter(r['cohort'] for r in raw)),
        'whole_native1500_approved':False,'project3_grounded_training_targets_approved':len(grounded),
        'project3_approval_scope':'perception-only prototypes, not full VL reasoning data or all matched splits',
        'training_approved':True,'approval_scope':'Only rows in the five *ai_approved.jsonl training files; no other data. Original quotas not silently weakened; train sampling/domain mix needs a versioned config.',
        'human_review_performed':False,'limitations':'AI process judgments can err; sample counterexample shows reviewer false positive. No benchmark performance, full test coverage, global contamination, or visual-gold claim.'}
    dump(OUT/'summary.json',summary)
    inputs=['reports/project2_ai_review_20261024/results.jsonl','reports/project2_ai_review_20261024/independent_counterexample_checks.json',
        'reports/project2_ai_review_20261024/independent_math_checks.json',
        'data/project3_grounding_ai_review_20261024/manifest.json',
        'reports/math_pair_delegated_review_20261024/reviews.json','reports/project1_batch500_20261024/common_candidates.json',
        'data/project2_code_v3_dev_verified/manifest.json','data/math_manifest.json','scripts/export_delegated_ai_releases.py']
    dump(OUT/'manifest.json',{'summary':summary,'approved_training_files':['project2_math_train_ai_approved.jsonl','project2_code_train_ai_approved.jsonl','project1_raw_paired_ai_approved.jsonl','project1_compact_paired_ai_approved.jsonl','project3_perception_prototype_ai_approved.jsonl'],
        'inputs':{p:file_sha(ROOT/p) for p in inputs},'files':{p.name:file_sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
