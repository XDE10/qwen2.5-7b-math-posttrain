"""Versioned release of observed approvals, with exact counts and hashes."""
import shutil,re
from pathlib import Path
from collections import Counter
from scripts.strict_v8_common import *
from src.data.approved_training_inputs import load_approved,model_view

def run():
    assert (REPORT/'budget_after.json').exists(),'API owner has not finished'
    if (REPORT/'frozen_protocol_resume.json').exists():assert (REPORT/'budget_after_resume.json').exists()
    from scripts.export_strict_geometry_v8 import run as geometry
    geometry()
    out=DATA/'release';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('sealed release exists')
    old=ROOT/'data/topup_v6_20261024/release';om=read(old/'manifest.json')
    for n,h in om['files'].items():assert sha(old/n)==h
    raw=rows(old/'project1_raw_paired.jsonl');compact=rows(old/'project1_compact_paired.jsonl')
    quarantine=read(REPORT/'source_profile_quarantine.json')
    extra=rows(REPORT/'approved_extra_pairs.jsonl') if (REPORT/'approved_extra_pairs.jsonl').exists() else []
    for r in rows(REPORT/'approved_new_pairs.jsonl')+extra:
        if any(r['sample_id'].startswith(k) for k in quarantine if len(k)==8):continue
        base={k:v for k,v in r.items() if k not in {'solution','raw_solution','compact_solution','id'}}
        base.update(pair_id=r['sample_id'],cohort='strict_v8',training_approved=True)
        raw.append({**base,'arm':'raw','solution':r['raw_solution']});compact.append({**base,'arm':'compact','solution':r['compact_solution']})
    assert [r['sample_id'] for r in raw]==[r['sample_id'] for r in compact]
    assert all(a['problem']==b['problem'] for a,b in zip(raw,compact))
    assert len({r['sample_id'] for r in raw})==len(raw)
    write(out/'project1_raw_paired.jsonl',raw);write(out/'project1_compact_paired.jsonl',compact)
    carry=['project2_math_train.jsonl','project2_math_dev.jsonl','project2_code_train.jsonl','project2_code_dev.jsonl','reserved_dev_ids_never_train.json','project1_rl_priority_not_rollout_selected.jsonl']
    for n in carry:shutil.copyfile(old/n,out/n);assert sha(old/n)==sha(out/n)
    for split in ['train','validation']:shutil.copyfile(DATA/'geometry'/(split+'_fact_pruned.jsonl'),out/('project3_'+split+'_fact_pruned.jsonl'))
    image_links=rows(ROOT/'data/project3_audit/geometry3k_intergps_links.jsonl')
    held_pixels={r['image_pixel_sha256'] for r in image_links if r['wrapper_split']!='train'}
    test_pixels={r['image_pixel_sha256'] for r in image_links if r['wrapper_split']=='test'}
    assert not {r['image_pixel_sha256'] for r in rows(out/'project3_train_fact_pruned.jsonl')}&held_pixels
    assert not {r['image_pixel_sha256'] for r in rows(out/'project3_validation_fact_pruned.jsonl')}&test_pixels
    from scripts.compact_distillation_v3 import QUOTAS
    counts=dict(Counter(r['problem_type'] for r in raw));g=read(DATA/'geometry/summary.json')
    summary={'project1_approved_pairs':len(raw),'project1_type_counts':counts,'project1_exact_target_met':counts==QUOTAS,'project1_new_approved':len(raw)-977,'project2_math_train':2500,'project2_math_dev':256,'project2_code_train':{'function':1250,'stdin_stdout':1250},'project2_code_dev':{'function':100,'stdin_stdout':100},'project3_train':g['approved_by_split']['train'],'project3_validation':g['approved_by_split']['validation'],'project3_coverage':g['responses_by_split'],'rl_priority_excluding_new_dev':4810,'gpu_runs':0,'approval':'itemwise delegated AI plus deterministic checks; not human gold'}
    train=['project1_raw_paired.jsonl','project1_compact_paired.jsonl','project2_math_train.jsonl','project2_code_train.jsonl','project3_train_fact_pruned.jsonl']
    manifest={'summary':summary,'approved_training_files':train,'approved_evaluation_files':['project2_math_dev.jsonl','project2_code_dev.jsonl','project3_validation_fact_pruned.jsonl'],'protected_dev_ids_file':'reserved_dev_ids_never_train.json','files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [old/'manifest.json',REPORT/'frozen_protocol.json',REPORT/'approved_new_pairs.jsonl',DATA/'geometry/summary.json']},'no_unreviewed_or_short_solution_override':True}
    for p in [REPORT/'frozen_protocol_resume.json',REPORT/'approved_extra_pairs.jsonl',REPORT/'source_profile_quarantine.json',ROOT/'src/data/geometry_split_guard.py',ROOT/'data/project3_audit/geometry3k_intergps_links.jsonl']:
        if p.exists():manifest['inputs'][str(p.relative_to(ROOT))]=sha(p)
    dump(out/'manifest.json',manifest)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True);checks={}
    for name in train:
        rs=load_approved(out/'manifest.json',name);lengths=[]
        for r in rs:
            if name.startswith('project1'):
                v=model_view(r,'math_sft');messages=[{'role':'system','content':r'Please reason step by step, and put your final answer within \boxed{}.'},{'role':'user','content':v['input']['problem']}]
                prefix=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True);full=tok.apply_chat_template(messages+[{'role':'assistant','content':v['target']}],tokenize=True,add_generation_prompt=False)
                limit=4096 if name=='project1_raw_paired.jsonl' and r['problem_type']=='Inequalities' else 3800
                assert full[:len(prefix)]==prefix and len(prefix)<len(full)<=limit
                if 'compact' in name:assert 150<=len(tok.encode(r['solution'],add_special_tokens=False))<=1500
                lengths.append(len(full))
            elif name.startswith('project3'):
                v=model_view(r,'perception_sft');assert sha(Path(v['input']['image_path']))==r['image_sha256']
                assert r['split']=='train' and r['certified_positive_tokens'] and not re.search(r'radius_\d|source_radius_id',v['target'])
            else:model_view(r,'opd')
        checks[name]={'rows':len(rs),'max_tokens':max(lengths) if lengths else None,'passed':True}
    dump(REPORT/'approved_input_preflight.json',{'files':checks,'old_release_all_hashes_unchanged':True,'p2_all_files_byte_identical':True,'gpu_runs':0})
    ledger=read(BUDGET/'ledger.json');assert all(r['state']!='reserved' for r in ledger)
    for snapshot in [ROOT/'reports/finalfill_v7_20261024/before_ledger.json',REPORT/'before_ledger.json',REPORT/'before_resume_ledger.json']:
        if not snapshot.exists():continue
        before=read(snapshot);assert ledger[:len(before)]==before
    budget={'cap_total_usd':11,'dispatch_attempts':len(ledger),'completed_peak_usage_usd':sum(r.get('charged_upper_usd',0) for r in ledger if r['state']=='complete'),'unknown_error_retained_reserve_usd':sum(r['reserved_usd'] for r in ledger if r['state']!='complete'),'committed_upper_usd':sum(r.get('charged_upper_usd',r['reserved_usd']) for r in ledger),'live_pending_requests':0,'actual_account_debit':False}
    assert budget['committed_upper_usd']<=11;dump(REPORT/'budget_final.json',budget);dump(REPORT/'summary.json',summary)
    configs=DATA/'training_configs';configs.mkdir(exist_ok=True)
    for p in (ROOT/'data/topup_v6_20261024/training_configs').glob('*.json'):
        if p.name=='manifest.json':continue
        c=read(p);c['data_manifest']=str(out/'manifest.json')
        for k in ['output_dir','model']:
            if k in c and isinstance(c[k],str):c[k]=c[k].replace('topup_v6','strict_v8')
        if p.name=='project1_raw_sft_smoke.json':c['max_length']=4096
        dump(configs/p.name,c)
    dump(configs/'manifest.json',{'files':{p.name:sha(p) for p in configs.glob('*.json') if p.name!='manifest.json'},'approved_data_manifest_sha':sha(out/'manifest.json'),'gpu_executed':False})
    print({'summary':summary,'budget':budget},flush=True)
if __name__=='__main__':run()
