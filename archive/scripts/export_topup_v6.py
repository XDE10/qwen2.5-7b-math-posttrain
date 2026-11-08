"""Exact quota exports, only pass evidence; preserve all rejected/surplus rows."""
from collections import Counter
from scripts.topup_common import *
from src.data.screening_io import ScreeningSimilarityIndex
from src.data.perception_v4 import token_key
def run():
    out=DATA/'release';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('v6 release already sealed')
    old=ROOT/'data/delegated_ai_releases_20261024';oldmeta=read(old/'manifest.json')
    for name,h in oldmeta['files'].items():assert sha(old/name)==h
    candidates=rows(DATA/'project2/candidates.jsonl');requests=rows(REPORT/'p2_review.jsonl')
    if (DATA/'project2_dev_reserve/candidates.jsonl').exists():candidates+=rows(DATA/'project2_dev_reserve/candidates.jsonl')
    if (DATA/'project2_train_reserve/candidates.jsonl').exists():candidates+=rows(DATA/'project2_train_reserve/candidates.jsonl')
    if (REPORT/'p2_dev_reserve_review.jsonl').exists():requests+=rows(REPORT/'p2_dev_reserve_review.jsonl')
    review={r['id']:r for r in requests};new=[];bounded={}
    for folder in ['project2','project2_dev_reserve','project2_train_reserve']:
        path=DATA/folder/'audit_requests.jsonl'
        if path.exists():
            for r in rows(path):bounded[r['record_key']]=len(json.dumps({**r,'id':r['record_key']},ensure_ascii=False))<=60000
    for r in candidates:
        key=r['domain']+'/'+r['split']+'/'+r['sample_id'];v=review.get(key,{})
        if v.get('verdict')=='pass' and bounded.get(key,False):new.append({**r,'approval_request_id':v.get('request_id'),'approval_method':'user_delegated_AI_plus_unchanged_local_gates','human_review_performed':False})
    p1all=ScreeningSimilarityIndex(.85)
    native_fresh=rows(DATA/'project1/native_new1500.jsonl')
    if (DATA/'project1_supplement/native_candidates.jsonl').exists():native_fresh+=rows(DATA/'project1_supplement/native_candidates.jsonl')
    for r in rows(ROOT/'data/project1_native_v5/native_candidates.jsonl')+native_fresh:p1all.add(r)
    mathdev=rows(DATA/'project2/retained_math_dev.jsonl');codedev=rows(DATA/'project2/retained_code_dev.jsonl');surplus=[]
    for domain,current,target in [('math',mathdev,256),('code',codedev,200)]:
        for r in sorted([r for r in new if r['domain']==domain and r['split']=='dev'],key=lambda r:stable('devexport6/'+r['sample_id'])):
            if domain=='math' and p1all.match(r['problem']):surplus.append({'sample_id':r['sample_id'],'reason':'new_dev_overlaps_p1_native_pool; never recycle to train'});continue
            if domain=='code' and sum(x['interface']==r['interface'] for x in current)>=100:surplus.append({'sample_id':r['sample_id'],'reason':'dev_interface_quota_surplus'});continue
            if len(current)>=target:surplus.append({'sample_id':r['sample_id'],'reason':'dev_quota_surplus'});continue
            if domain=='code':assert r.get('execution',{}).get('passed') and not any(r['execution'].get('constant_mutant_passes',[True]))
            current.append(r)
    for r in mathdev+codedev:r.update(training_approved=False,evaluation_approved=True,split='dev',human_review_performed=False)
    devindex=ScreeningSimilarityIndex(.85)
    for r in mathdev+codedev:devindex.add(r)
    # All presealed dev candidates remain held out, including surplus/rejects.
    reserved_ids={r['sample_id'] for r in mathdev+codedev}
    for r in candidates:
        if r['split']=='dev':devindex.add(r);reserved_ids.add(r['sample_id'])
    for p in [ROOT/'data/project2_code/dev_verification.jsonl',ROOT/'data/delegated_ai_releases_20261024/project2_fixed_dev_audit.jsonl',DATA/'project2/dev_execution.jsonl',DATA/'project2_dev_reserve/execution.jsonl']:
        for r in rows(p):reserved_ids.add(r['sample_id'])
    dump(out/'reserved_dev_ids_never_train.json',{'sample_ids':sorted(reserved_ids),'rule':'Every sealed/attempted old and new dev ID stays withheld, including rejected, unused and surplus candidates. Never recycle to P2 train or RL seeds.'})
    mathtrain=rows(old/'project2_math_train_ai_approved.jsonl');codetrain=rows(old/'project2_code_train_ai_approved.jsonl')
    for domain,current,target in [('math',mathtrain,2500),('code',codetrain,2500)]:
        for r in sorted([r for r in new if r['domain']==domain and r['split']=='train'],key=lambda r:stable('trainexport6/'+r['sample_id'])):
            if devindex.match(r['problem']):surplus.append({'sample_id':r['sample_id'],'reason':'heldout_overlap'});continue
            if domain=='code' and sum(x['interface']==r['interface'] for x in current)>=1250:surplus.append({'sample_id':r['sample_id'],'reason':'train_interface_quota_surplus'});continue
            if len(current)>=target:surplus.append({'sample_id':r['sample_id'],'reason':'train_quota_surplus'});continue
            # OPD prompt-only exports do not expose audit reference code/tests.
            current.append({k:v for k,v in {**r,'training_approved':True,'split':'train'}.items() if k not in {'reference_code','test_cases','verification_info','solution','execution'}})
    for domain,train,dev in [('math',mathtrain,mathdev),('code',codetrain,codedev)]:
        assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in dev}
        assert not {r['sample_id'] for r in train}&reserved_ids
        write(out/f'project2_{domain}_train.jsonl',train);write(out/f'project2_{domain}_dev.jsonl',dev)
    raw=rows(old/'project1_raw_paired_ai_approved.jsonl');compact=rows(old/'project1_compact_paired_ai_approved.jsonl');pairreviews=rows(REPORT/'p1_pair_review.jsonl');pairmap={r['sample_id']:r for r in read(REPORT/'p1_local/common_candidates.json')};source={r['sample_id']:r for r in rows(DATA/'project1/native_new1500.jsonl')}
    if (REPORT/'p1_supplement_pair_review.jsonl').exists():
        pairreviews+=rows(REPORT/'p1_supplement_pair_review.jsonl');pairmap.update({r['sample_id']:r for r in read(REPORT/'p1_supplement_local/common_candidates.json')});source.update({r['sample_id']:r for r in rows(DATA/'project1_supplement/native_candidates.jsonl')})
    for v in sorted(pairreviews,key=lambda r:stable('pairexport6/'+r['sample_id'])):
        if v.get('verdict')!='pass':continue
        r=pairmap[v['sample_id']]
        if devindex.match(r['problem']):surplus.append({'sample_id':r['sample_id'],'reason':'paired_candidate_new_dev_overlap'});continue
        if len(raw)>=1000:surplus.append({'sample_id':r['sample_id'],'reason':'approved_pair_target_1000_surplus'});continue
        base={k:v for k,v in source[r['sample_id']].items() if k not in {'solution','original_problem'}};base.update(training_approved=True,pair_id=r['sample_id'],approval_request_id=v['request_id'],human_review_performed=False,review_method='unchanged_local_admission_plus_delegated_two_full_trajectory_review',cohort='v6_topup')
        raw.append({**base,'arm':'raw','solution':r['raw_solution']});compact.append({**base,'arm':'compact','solution':r['compact_solution']})
    assert [r['sample_id'] for r in raw]==[r['sample_id'] for r in compact]
    write(out/'project1_raw_paired.jsonl',raw);write(out/'project1_compact_paired.jsonl',compact)
    write(out/'project1_native_fresh1500_candidates_not_blanket_approved.jsonl',native_fresh)
    write(out/'project1_native_union_candidates_not_blanket_approved.jsonl',rows(ROOT/'data/project1_native_v5/native_candidates.jsonl')+native_fresh)
    rl=[r for r in rows(DATA/'project1/rl_official_level3to5_priority.jsonl') if not devindex.match(r['problem'])];write(out/'project1_rl_priority_not_rollout_selected.jsonl',rl)
    allcases=rows(DATA/'project3/cases.jsonl')
    if (DATA/'project3_reserve/cases.jsonl').exists():allcases+=rows(DATA/'project3_reserve/cases.jsonl')
    cases={r['sample_id']:r for r in allcases};visual=[];validation=[]
    for v in rows(REPORT/'p3_fact_review_reparsed.jsonl'):
        if not v.get('pruning',{}).get('accepted'):continue
        c=cases[v['sample_id']];p=v['pruning'];is_train=c['split']=='train'
        r={k:c[k] for k in ['sample_id','original_id','split','problem','reference_answer','image_path','image_sha256']};r.update(model_perception=p['model_perception'],certified_positive_tokens=p['positive_tokens'],certified_negative_tokens=p['negative_tokens'],masked_uncertain_facts=p['masked'],certified_fact_evidence=p['certified_facts'],training_approved=is_train,evaluation_approved=not is_train,perception_reward_enabled=is_train,perception_reward_mode='sparse_certified_v4',perception_reward_complete_categories=[],model_input_whitelist=['problem','image_path'],model_target_whitelist=['model_perception'],approval_request_id=v['request_id'],human_review_performed=False,certification_scope='delegated AI + deterministic guards; no perfect truth claim')
        (visual if is_train else validation).append(r)
    write(out/'project3_train_fact_pruned.jsonl',visual);write(out/'project3_validation_fact_pruned.jsonl',validation);write(out/'surplus_and_final_exclusions.jsonl',surplus)
    summary={'project1_approved_pairs':len(raw),'project1_new_approved':len(raw)-213,'project2_math_train':len(mathtrain),'project2_math_dev':len(mathdev),'project2_code_train':dict(Counter(r['interface'] for r in codetrain)),'project2_code_dev':dict(Counter(r['interface'] for r in codedev)),'project3_train':len(visual),'project3_validation':len(validation),'rl_priority_excluding_new_dev':len(rl),'project3_cases_with_explicit_negatives':sum(bool(r['certified_negative_tokens']) for r in visual),'targets_met':{'p1_800to1000':800<=len(raw)<=1000,'p2_math':len(mathtrain)==2500 and len(mathdev)==256,'p2_code':Counter(r['interface'] for r in codetrain)=={'function':1250,'stdin_stdout':1250} and Counter(r['interface'] for r in codedev)=={'function':100,'stdin_stdout':100},'p3_500plus':len(visual)>=500},'gpu_runs':0,'approval':'itemwise user-delegated AI engineering release; not human gold'}
    dump(out/'summary.json',summary);trainfiles=['project1_raw_paired.jsonl','project1_compact_paired.jsonl','project2_math_train.jsonl','project2_code_train.jsonl','project3_train_fact_pruned.jsonl'];trainfiles=[p for p in trainfiles if (out/p).stat().st_size]
    files={p.name:sha(p) for p in out.glob('*.jsonl')};files['reserved_dev_ids_never_train.json']=sha(out/'reserved_dev_ids_never_train.json')
    dump(out/'manifest.json',{'summary':summary,'approved_training_files':trainfiles,'approved_evaluation_files':['project2_math_dev.jsonl','project2_code_dev.jsonl','project3_validation_fact_pruned.jsonl'],'protected_dev_ids_file':'reserved_dev_ids_never_train.json','files':files,'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [old/'manifest.json',REPORT/'p1_pair_review.jsonl',REPORT/'p2_review.jsonl',REPORT/'p3_fact_review_reparsed.jsonl',REPORT/'frozen_protocol.json']},'no_unreviewed_or_short_solution_override':True});print(summary,flush=True)
if __name__=='__main__':run()
