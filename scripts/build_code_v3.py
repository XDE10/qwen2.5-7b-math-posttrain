"""Versioned exact-wrapper removal; replacements ONLY from audited AceCode pool."""
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from src.data.code_triviality_v3 import analyze
from src.data.build_code_v2 import order,prompt_record
from src.data.code_v2_dedup import Protected
from src.data.screening_io import ScreeningSimilarityIndex,rows,sha,json_write
from src.data.prepare_math_data import write_jsonl
from src.data.code_sandbox import function_job,run_jobs,environment

OLD=Path('data/project2_code_v2');OUT=Path('data/project2_code_v3');REPORT=Path('reports/project2_code_v3')


def run():
    if (OUT/'manifest.json').exists():raise FileExistsError('code v3 sealed; choose another version')
    OUT.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    inputs={p.as_posix():sha(p) for p in OLD.iterdir() if p.is_file()}
    candidates=list(rows(OLD/'function_candidates.jsonl'));lookup={r['sample_id']:r for r in candidates}
    train=list(rows(OLD/'train_prompts.jsonl'));dev=list(rows(OLD/'dev_verification.jsonl'))
    original_ids={r['sample_id'] for r in train+dev};audit=[];clean={}
    for r in candidates:
        a=analyze(r['reference_code'],r['entry_point']);clean[r['sample_id']]=not a['isolate']
        audit.append({'sample_id':r['sample_id'],'source_key':r['source_key'],**a})
    write_jsonl(OUT/'complexity_audit.jsonl',audit)
    isolated=[];kepttrain=[];keptdev=[]
    for split,records,target in [('train',train,kepttrain),('dev',dev,keptdev)]:
        for r in records:
            if r['interface']=='function' and not clean[r['sample_id']]:
                isolated.append({'split':split,'sample_id':r['sample_id'],'analysis':analyze(lookup[r['sample_id']]['reference_code'],r['entry_point'])})
            else:target.append(r)
    write_jsonl(OUT/'isolated_wrappers.jsonl',isolated)
    needdev=100-sum(r['interface']=='function' for r in keptdev)
    needtrain=1250-sum(r['interface']=='function' for r in kepttrain)
    protected=Protected();index=ScreeningSimilarityIndex(.85)
    names=set();bodies=set();families=set()
    # All prior dev topics remain protected even when removed; no split recycling.
    for r in train+dev:
        index.add(r)
        for key,bucket in [('entry_point',names),('reference_body_sha256',bodies),('family_id',families)]:
            if r.get(key):bucket.add(r[key])
    tok=AutoTokenizer.from_pretrained('data/tokenizers/Qwen2.5-1.5B-Instruct',local_files_only=True)
    def eligible(r):
        return r['source_key']=='ace' and clean[r['sample_id']] and r['sample_id'] not in original_ids and not protected.match(r) and not index.match(r['problem']) and r['entry_point'] not in names and r['reference_body_sha256'] not in bodies and r['family_id'] not in families
    def claim(r):
        index.add(r);names.add(r['entry_point']);bodies.add(r['reference_body_sha256']);families.add(r['family_id'])
    def prompt(r,split):
        p=prompt_record(r,split);p['prompt_token_count']=len(tok.apply_chat_template([{'role':'user','content':p['prompt']}],tokenize=True,add_generation_prompt=True));return p
    ordered=sorted(candidates,key=order);replacementdev=[];replacementtrain=[];execution=[];pendingdev=[];infrastructure_error=None
    # New dev executes fresh, even if candidate appeared in an old execution cache.
    for r in ordered:
        if len(replacementdev)>=needdev:break
        if not eligible(r) or len({repr((t['args'],t['kwargs'])) for t in r['test_cases']})<3:continue
        p=prompt(r,'dev')
        if p['prompt_token_count']>2560:continue
        if infrastructure_error:
            pendingdev.append({**p,'reference_code':r['reference_code'],'test_cases':r['test_cases'],'execution_status':'pending_restricted_container_test','training_approved':False})
            claim(r)
            if len(pendingdev)>=needdev:break
            continue
        try:
            response=run_jobs([function_job(r)]);ex=response['results'][0]
        except RuntimeError as e:
            infrastructure_error=str(e)
            pendingdev.append({**p,'reference_code':r['reference_code'],'test_cases':r['test_cases'],'execution_status':'pending_restricted_container_test','training_approved':False})
            claim(r)
            if len(pendingdev)>=needdev:break
            continue
        execution.append(ex);write_jsonl(OUT/'new_dev_execution.jsonl',execution)
        if not ex.get('passed') or any(ex.get('constant_mutant_passes',[True])):continue
        replacementdev.append({**p,'reference_code':r['reference_code'],'test_cases':r['test_cases'],'verification_info':None,'execution':ex})
        claim(r)
    if len(replacementdev)+len(pendingdev)!=needdev:raise ValueError('Insufficient audited AceCode dev reserves')
    for r in ordered:
        if len(replacementtrain)>=needtrain:break
        if not eligible(r):continue
        p=prompt(r,'train')
        if p['prompt_token_count']>2560:continue
        replacementtrain.append(p);claim(r)
    if len(replacementtrain)!=needtrain:raise ValueError('Insufficient audited AceCode reserves')
    newtrain=kepttrain+replacementtrain;newdev=keptdev+replacementdev
    assert Counter(r['interface'] for r in newtrain)=={'function':1250,'stdin_stdout':1250}
    assert Counter(r['interface'] for r in newdev)=={'function':100-len(pendingdev),'stdin_stdout':100}
    assert not {r['sample_id'] for r in newtrain}&{r['sample_id'] for r in newdev}
    assert not {r['family_id'] for r in newtrain}&{r['family_id'] for r in newdev}
    finaldevindex=ScreeningSimilarityIndex(.85)
    for r in newdev:finaldevindex.add(r)
    assert not any(finaldevindex.match(r['problem']) for r in newtrain)
    assert all(sha(p)==h for p,h in inputs.items())
    write_jsonl(OUT/'train_prompts.jsonl',newtrain);write_jsonl(OUT/'dev_verification.jsonl',newdev)
    write_jsonl(OUT/'dev_replacement_plan_pending.jsonl',pendingdev)
    write_jsonl(OUT/'dev_metadata.jsonl',[{k:v for k,v in r.items() if k not in ['reference_code','test_cases','verification_info','execution']} for r in newdev])
    summary={'version':'code-v3-wrapper-filter-20261024','train':len(newtrain),'dev':len(newdev),
        'isolated_by_split':dict(Counter(r['split'] for r in isolated)),
        'isolated_reasons':dict(Counter(r['analysis']['reason'] for r in isolated)),
        'audited_ace_reserves':sum(r['source_key']=='ace' for r in candidates),
        'new_train_ids':[r['sample_id'] for r in replacementtrain],'new_dev_ids':[r['sample_id'] for r in replacementdev],
        'new_dev_fresh_container_attempts':len(execution),'new_dev_all_source_tests_passed':not pendingdev and all(r['execution']['passed'] for r in replacementdev),
        'new_dev_pending_container_ids':[r['sample_id'] for r in pendingdev],
        'sandbox_infrastructure_error':infrastructure_error,'release_ready':not pendingdev,
        'sandbox':environment(),'heldout_and_cross_split_checks':'Protected + ID + family/body/name + Jaccard .85 + prompt<=2560',
        'old_train_or_dev_recycled_to_other_split':False,'accuracy_scope':'provided tests, not independent semantic oracle; AST metrics are not difficulty labels',
        'human_review':'pending','training_approved':False,'paid_api_calls':0}
    json_write(REPORT/'summary.json',summary)
    (REPORT/'执行报告.md').write_text('# 代码 v3 包装函数筛查与替补\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    json_write(OUT/'manifest.json',{'inputs':inputs,'summary':summary,'files':{p.name:{'path':p.as_posix(),'sha256':sha(p)} for p in OUT.glob('*.jsonl')},'scripts':{p:sha(p) for p in ['src/data/code_triviality_v3.py','scripts/build_code_v3.py']}})
    print(json.dumps({k:v for k,v in summary.items() if not k.startswith('new_') or k.endswith('attempts')},ensure_ascii=False),flush=True)


if __name__=='__main__':run()
