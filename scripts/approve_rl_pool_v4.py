"""User-preauthorized conditional approval after complete, independently checked E0 sampling."""
import argparse
import json
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from src.training.frozen_verifier_v4 import Worker,sha,digest,grader_identity
from src.training.math_reward_v2 import category
from src.training.rollout_pipeline_v4 import read,jsonl,indexed,selection,write_new,load_prepared,BASE,REV,trial_seed


def collect(root,rows,protocol):
    done=read(root/'sampling_complete.json')
    if not done.get('complete') or done['identity_sha256']!=sha(root/'sampling_identity.json'):raise ValueError('Incomplete or changed sampling identity')
    for name,h in done['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise ValueError('Sampling journal digest/path changed')
    raw=[];graded=[]
    for shard in range(protocol['num_shards']):
        raw+=jsonl(root/'raw'/f'shard-{shard:05d}.jsonl')
        graded+=jsonl(root/'graded'/f'shard-{shard:05d}.jsonl')
    ri=indexed(raw);gi=indexed(graded)
    n=protocol['num_samples'];expected={(r['sample_id'],t) for r in rows for t in range(n)}
    if set(ri)!=expected or set(gi)!=expected:raise ValueError('Require every candidate and every trial; no partial-run approval')
    groups=defaultdict(list)
    for key,r in ri.items():
        if r['seed']!=trial_seed(protocol['seed'],r['sample_id'],r['trial']):raise ValueError('Trial seed changed')
        if gi[key]['raw_sha256']!=digest(r):raise ValueError('Raw/graded linkage changed')
        if gi[key]['seed']!=r['seed']:raise ValueError('Graded/raw trial seed changed')
        if r['finish_reason']!=gi[key]['finish_reason']:raise ValueError('Finish reason changed')
        if len(r['token_ids'])>protocol['max_new_tokens']:raise ValueError('Recorded token budget violation')
        if category(gi[key]['verification'])!=gi[key]['category']:raise ValueError('Recorded scorer category changed')
        if gi[key]['category'] in ('equivalent','mismatch') and bool(gi[key]['verification'].get('correct'))!=(gi[key]['category']=='equivalent'):
            raise ValueError('Recorded correctness contradicts category')
        groups[key[0]].append(gi[key])
    return ri,groups


def verified_selection(rows,raw,groups,protocol,checker,workers=1):
    n=protocol['num_samples'];accepted=[];decisions=[]
    def verify_row(row):
        sid=row['sample_id'];initial=selection(groups[sid],n)
        if initial['eligible']:
            fresh=[]
            original_categories={r['trial']:r['category'] for r in groups[sid]}
            category_changed=False
            for trial in range(n):
                old=raw[(sid,trial)]
                result=checker(old['text'],row['reference_answer'])
                category_changed=category_changed or category(result)!=original_categories[trial]
                fresh.append({'sample_id':sid,'trial':trial,'category':category(result),'finish_reason':old['finish_reason']})
            current=selection(fresh,n)
            if category_changed or current!=initial:
                current={'eligible':False,'reason':'independent_regrade_changed_or_failed'}
        else:current=initial
        return row,current
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row,current in pool.map(verify_row,rows):
            decisions.append({'sample_id':row['sample_id'],**current})
            if current['eligible']:accepted.append((row,current))
    accepted.sort(key=lambda pair:digest([protocol['seed'],pair[0]['sample_id']]))
    return accepted[:3000],len(accepted),decisions


def config_templates(out,manifest,sft_raw=None,sft_compact=None):
    folder=out/'configs';folder.mkdir()
    for stage,arms in [('E2',['shared']),('E3',['raw','compact']),('E4',['raw','compact'])]:
        for arm in arms:
            template=Path(f'reports/handover_20261024_v2/formal_v1/equal_exposure/{stage}_{arm}.draft.json')
            c=read(template)
            for k in ('budget_protocol','matrix_entry'):c.pop(k,None)
            c.update(draft=False,status='RL数据已条件批准；GPU/训练效果未验证',data_manifest='${APPROVED_RELEASE_ROOT}/manifest.json',
                     data_file='project1_rl_rollout_approved_v1.jsonl',output_dir='${OUTPUT_ROOT}/'+stage+'_'+arm,
                     model='${E0_MODEL_PATH}' if stage=='E2' else '${SFT_'+arm.upper()+'_MERGED}',
                     experiment=stage+'_'+arm,approval_schema='project1-rollout-approved-v4',
                     expected_grader_identity_sha256=manifest['grader_identity_sha256'],
                     max_steps=200,notes='20-update pilot and GPU acceptance precede formal200; no convergence claim')
            if stage!='E2':
                c['expected_sft']='BIND_EXACT_FORMAL_SFT_AT_RUNTIME'
                c['expected_sft_protocol']={'arm':arm,'max_steps':94,'gradient_accumulation_steps':16,'batch_size':1,'seed':42,'learning_rate':1e-5,
                                            'lora_r':16,'lora_alpha':32,'lora_dropout':0,'max_length':4096,'model_revision':REV,'data_file':f'project1_{arm}_paired.jsonl'}
            else:c.pop('expected_sft',None)
            write_new(folder/f'{stage}_{arm}.json',c)


def main():
    p=argparse.ArgumentParser();p.add_argument('--sampling-root',required=True);p.add_argument('--output-root',required=True)
    a=p.parse_args();root=Path(a.sampling_root).resolve();out=Path(a.output_root).resolve()
    if out.exists():raise ValueError('Approval output-root must be new; never overwrite a release')
    protocol=read(root/'sampling_identity.json')
    if protocol.get('test_fixture') or protocol['model_identity'].get('test_fixture'):
        raise ValueError('Test fixtures cannot become production approvals')
    if protocol['schema']!='project1-sampling-v4' or protocol['model_identity']['repo']!=BASE or protocol['model_identity']['revision']!=REV or not protocol['model_identity'].get('verified_against_hf_lfs'):
        raise ValueError('Approval requires verified official E0 sampling')
    project=Path(__file__).resolve().parents[1]
    if protocol['sampler_source_sha256']!=sha(project/'scripts/sample_rl_candidates_v4.py') or protocol['pipeline_source_sha256']!=sha(project/'src/training/rollout_pipeline_v4.py'):
        raise ValueError('Sampling code changed; preserve the exact sampled version')
    rows,seal=load_prepared(root/'candidate_pool.jsonl',root/'preparation_seal.json')
    if sha(root/'candidate_pool.jsonl')!=protocol['candidate_sha256'] or sha(root/'preparation_seal.json')!=protocol['preparation_sha256']:
        raise ValueError('Prepared pool is not this sampling run')
    if protocol['grader_identity_sha256']!=digest(grader_identity()):raise ValueError('Scorer version mismatch')
    raw,groups=collect(root,rows,protocol)
    # Independent regrading is CPU-only and uses exactly the frozen grader.
    from src.training.runner_support_v4 import PooledVerifier
    checker=PooledVerifier(4)
    try:selected,eligible_count,decisions=verified_selection(rows,raw,groups,protocol,checker.check,workers=4)
    finally:checker.close()
    if not selected:raise ValueError('No independently verified eligible candidates; no release created')
    selected_ids={r['sample_id'] for r,_ in selected}
    for decision in decisions:
        decision['selected_for_training']=decision['sample_id'] in selected_ids
        if decision['eligible'] and not decision['selected_for_training']:decision['selection_note']='eligible_but_seeded_cap3000'
    protected_bytes=(root/'protected_dev_ids_never_train.json').read_bytes()
    if sha(root/'protected_dev_ids_never_train.json')!=seal['protected_ids_sha256']:raise ValueError('Protected IDs changed')
    protected=set(json.loads(protected_bytes)['sample_ids'])
    if {r['sample_id'] for r,_ in selected}&protected:raise ValueError('Protected dev ID in selected set')
    out.mkdir(parents=True,exist_ok=False)
    approved=[];evidence=[]
    for original,decision in selected:
        row=dict(original);row.update(training_approved=True,requires_7b_rollout_selection=False,
                rollout_approval={'num_samples':protocol['num_samples'],'correct':decision['correct'],'scorer_sha256':protocol['grader_identity_sha256'],'sampling_identity_sha256':sha(root/'sampling_identity.json')})
        approved.append(row)
        for trial in range(protocol['num_samples']):evidence.append(raw[(row['sample_id'],trial)])
    train=out/'project1_rl_rollout_approved_v1.jsonl'
    train.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in approved),encoding='utf-8')
    (out/'protected_dev_ids_never_train.json').write_bytes(protected_bytes)
    (out/'selected_trial_evidence.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in evidence),encoding='utf-8')
    (out/'approval_decisions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in decisions),encoding='utf-8')
    for name in ('sampling_identity.json','sampling_complete.json','preparation_seal.json'):(out/name).write_bytes((root/name).read_bytes())
    write_new(out/'e0_model_identity.json',protocol['model_identity'])
    manifest={'schema':'project1-rollout-approved-v4','approved_training_files':[train.name],'approved_evaluation_files':[],
              'approval_tool_source_sha256':sha(__file__),
              'protected_dev_ids_file':'protected_dev_ids_never_train.json',
              'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
              'grader_identity_sha256':protocol['grader_identity_sha256'],'approval_authority':'user preauthorization 2026-10-25; conditional deterministic E0 rollout approval',
              'approval_method':'complete sampling, no unscorable/non-stop trials, independent CPU regrade; k/n in [0.25,0.75]; deterministic cap3000',
              'summary':{'candidate_count':len(rows),'eligible_count':eligible_count,'approved_count':len(approved),'num_samples':protocol['num_samples'],'human_gold':False},
              'sft_semantic_suspect_ids':seal['sft_semantic_suspect_ids']}
    config_templates(out,manifest)
    manifest['files']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    write_new(out/'manifest.json',manifest)
    print(json.dumps(manifest['summary'],ensure_ascii=False))


if __name__=='__main__':main()
