import copy
import json
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.training.frozen_verifier_v4 import Worker,grader_identity,digest,sha
from src.training.math_reward_v2 import category,check_math
from src.training.rollout_pipeline_v4 import selection,indexed,append,jsonl,trial_seed,write_new,BASE,REV,load_prepared
from scripts.sample_rl_candidates_v4 import complete
from scripts.approve_rl_pool_v4 import collect,verified_selection,config_templates


@pytest.mark.parametrize('n',[4,8])
def test_exact_selection_band(n):
    for k in range(n+1):
        records=[{'sample_id':'a','trial':i,'category':'equivalent' if i<k else 'mismatch','finish_reason':'stop'} for i in range(n)]
        r=selection(records,n)
        assert r['eligible']==(n//4<=k<=3*n//4)
        assert r['correct']==k


@pytest.mark.parametrize('status',['unsupported_relation','multiple_boxes','unclosed_box','parse_failure','timeout','verifier_error'])
def test_unscorable_is_not_wrong(status):
    records=[{'sample_id':'a','trial':i,'category':'equivalent' if i<2 else 'mismatch','finish_reason':'stop'} for i in range(4)]
    records[3]['category']=status
    assert selection(records,4)=={'eligible':False,'reason':'unscorable_trial'}


def test_partial_truncated_duplicate():
    records=[{'sample_id':'a','trial':i,'category':'equivalent' if i<2 else 'mismatch','finish_reason':'stop'} for i in range(4)]
    assert selection(records[:3],4)['reason']=='incomplete_trials'
    records[3]['finish_reason']='length'
    assert selection(records,4)['reason']=='truncated_or_aborted'
    with pytest.raises(ValueError,match='Duplicate'):selection(records+[records[0]],4)


def test_durable_journal_and_tail(tmp_path):
    path=tmp_path/'arbitrary local path'/'raw.jsonl'
    append(path,{'sample_id':'a','trial':0,'text':'测试'})
    assert jsonl(path)[0]['text']=='测试'
    with path.open('ab') as f:f.write(b'{"torn":')
    with pytest.raises(ValueError,match='tail'):jsonl(path)


def test_seeds_independent_of_batch_order():
    assert trial_seed(42,'a',0)==trial_seed(42,'a',0)
    assert len({trial_seed(42,s,i) for s in ['a','b'] for i in range(8)})==16


@pytest.mark.parametrize('text,answer',[(r'\boxed{2}','2'),(r'\boxed{3}','2'),('2','2'),(r'\boxed{x>1}','x>1'),(r'\boxed{1}\boxed{2}','1'),(r'\boxed{','2')])
def test_warm_transport_matches_original(text,answer):
    original=check_math(text,answer);worker=Worker()
    try:result=worker.check(text,answer)
    finally:worker.close()
    assert category(result)==category(original)
    assert result.get('correct')==original.get('correct')
    assert result.get('status')==original.get('status')


def test_timeout_closes_process(monkeypatch):
    worker=Worker();worker.proc=SimpleNamespace(poll=lambda:None,stdin=SimpleNamespace(write=lambda x:None,flush=lambda:None))
    def fail(**kwargs):raise queue.Empty()
    worker.q=SimpleNamespace(get=fail);closed=[]
    monkeypatch.setattr(worker,'close',lambda:closed.append(True))
    assert worker.check('x','1')['status']=='timeout' and closed


def fixture_run(root,n=4):
    root.mkdir()
    protocol={'schema':'project1-sampling-v4','num_samples':n,'num_shards':1,'seed':42,'max_new_tokens':8,
              'model_identity':{'repo':BASE,'revision':REV,'verified_against_hf_lfs':True,'test_fixture':True}}
    write_new(root/'sampling_identity.json',protocol)
    rows=[{'sample_id':'a','reference_answer':'2','split':'train','training_approved':False}]
    for trial in range(n):
        good=trial<n//2
        raw={'sample_id':'a','trial':trial,'seed':trial_seed(42,'a',trial),'text':'ok' if good else 'wrong','token_ids':[3,2],'finish_reason':'stop'}
        graded={'sample_id':'a','trial':trial,'seed':raw['seed'],'raw_sha256':digest(raw),'finish_reason':'stop',
                'category':'equivalent' if good else 'mismatch','verification':{'status':'equivalent' if good else 'mismatch','correct':good}}
        append(root/'raw/shard-00000.jsonl',raw);append(root/'graded/shard-00000.jsonl',graded)
    assert complete(root,rows,protocol)
    return protocol,rows


@pytest.mark.parametrize('n',[4,8])
def test_independent_regrade_and_complete_coverage(tmp_path,n):
    root=tmp_path/'run';protocol,rows=fixture_run(root,n)
    raw,groups=collect(root,rows,protocol)
    checker=lambda t,a:{'status':'equivalent' if t=='ok' else 'mismatch','correct':t=='ok'}
    picked,count,decisions=verified_selection(rows,raw,groups,protocol,checker,workers=2)
    assert len(picked)==count==1 and decisions[0]['correct']==n//2
    assert not verified_selection(rows,raw,groups,protocol,lambda *a:{'status':'timeout'},workers=2)[0]
    path=root/'raw/shard-00000.jsonl';path.write_text(path.read_text().replace('wrong','forged'))
    with pytest.raises(ValueError,match='digest'):collect(root,rows,protocol)


def test_regrade_cannot_hide_trial_changes_behind_same_count(tmp_path):
    protocol,rows=fixture_run(tmp_path/'run')
    raw,groups=collect(tmp_path/'run',rows,protocol)
    statuses=iter(['mismatch','equivalent','equivalent','mismatch'])
    selected,_,decisions=verified_selection(rows,raw,groups,protocol,lambda *a:{'status':next(statuses)})
    assert not selected and decisions[0]['reason']=='independent_regrade_changed_or_failed'


def test_no_incomplete_or_fixture_approval(tmp_path,monkeypatch):
    from scripts.approve_rl_pool_v4 import main
    root=tmp_path/'run';protocol,rows=fixture_run(root)
    (root/'sampling_complete.json').unlink()
    with pytest.raises(FileNotFoundError):collect(root,rows,protocol)
    monkeypatch.setattr(sys,'argv',['approve','--sampling-root',str(root),'--output-root',str(tmp_path/'approved')])
    with pytest.raises(ValueError,match='fixtures'):main()
    assert not (tmp_path/'approved').exists()


def test_conditional_cap3000():
    rows=[{'sample_id':str(i),'reference_answer':'2'} for i in range(3003)]
    raw={};groups={}
    for row in rows:
        sid=row['sample_id'];groups[sid]=[]
        for t in range(4):
            raw[(sid,t)]={'sample_id':sid,'trial':t,'text':'ok' if t==0 else 'wrong','finish_reason':'stop'}
            groups[sid].append({'sample_id':sid,'trial':t,'category':'equivalent' if t==0 else 'mismatch','finish_reason':'stop'})
    checker=lambda t,a:{'status':'equivalent' if t=='ok' else 'mismatch'}
    selected,total,_=verified_selection(rows,raw,groups,{'num_samples':4,'seed':42},checker,workers=4)
    assert total==3003 and len(selected)==3000
    assert len({r['sample_id'] for r,d in selected})==3000


def test_local_evidence_support_and_embeddings():
    out=Path('reports/local_cpu_ready_20261025_v4')
    support=json.loads((out/'support_summary.json').read_text(encoding='utf-8'))
    assert support['input_count']==4630 and support['supported_count']==4571
    pool=out/'rl_candidates_verifier_supported_semantic_screened_v3.jsonl'
    rows,seal=load_prepared(pool,out/'preparation_seal.json')
    assert len(rows)==4522 and seal['prompt_max']==424
    assert not seal['sft_semantic_suspect_ids']
    assert all(r['training_approved'] is False for r in rows)
    checks={r['sample_id']:r for r in jsonl(out/'support_checks.jsonl')}
    assert len(checks)==4630 and all(checks[r['sample_id']]['supported_and_consistent'] for r in rows)
    with pytest.raises(ValueError):load_prepared(out/'rl_candidates_verifier_supported_v2.jsonl',out/'preparation_seal.json')


def test_embedding_pooling_keeps_long_text():
    import torch
    from scripts.embedding_screen_project1_v4 import encode,nearest
    class Tok:
        def encode(self,t,**kwargs):return [3]*len(t)
        def build_inputs_with_special_tokens(self,ids):return [1]+ids+[2]
        def pad(self,data,**kwargs):
            n=max(map(len,data['input_ids']))
            return {'input_ids':torch.tensor([r+[0]*(n-len(r)) for r in data['input_ids']]),'attention_mask':torch.tensor([[1]*len(r)+[0]*(n-len(r)) for r in data['input_ids']])}
    class Model:
        config=SimpleNamespace(hidden_size=2)
        def __call__(self,input_ids,attention_mask):return SimpleNamespace(last_hidden_state=torch.stack([input_ids.float(),torch.ones_like(input_ids).float()],-1))
    vectors,counts=encode(['x'*300,'x'],Tok(),Model())
    assert counts==[2,1]
    assert nearest(vectors,vectors,k=1)[0].flatten().tolist()==[0,1]


def test_config_unlock_templates_are_portable(tmp_path):
    config_templates(tmp_path,{'grader_identity_sha256':'fixed'})
    files=list((tmp_path/'configs').glob('*.json'));assert len(files)==5
    for path in files:
        config=json.loads(path.read_text())
        assert config['draft'] is False and config['data_file']=='project1_rl_rollout_approved_v1.jsonl'
        assert 'PROJECT_DRIVE_ROOT' not in path.read_text()
        assert config['data_manifest']=='${APPROVED_RELEASE_ROOT}/manifest.json'


def test_v4_resume_uses_extended_identity(tmp_path,monkeypatch):
    import src.training.guards_v4 as guards
    monkeypatch.setattr(guards,'validate_model',lambda *a:None)
    from src.training.guards_v4 import approved_config
    config=json.loads(Path('configs/project1_raw_sft_smoke.json').read_text())
    config.update(arm='raw',output_dir=str(tmp_path/'run'),e0_model_identity_path='synthetic-fixture-only')
    cp=tmp_path/'config.json';cp.write_text(json.dumps(config))
    _,_,identity=approved_config(cp)
    checkpoint=tmp_path/'checkpoint-1';checkpoint.mkdir()
    for name in ['optimizer.pt','scheduler.pt','rng_state.pth']:(checkpoint/name).write_bytes(b'fixture')
    (checkpoint/'trainer_state.json').write_text('{"global_step":1}')
    (checkpoint/'run_identity.json').write_text(json.dumps(identity))
    config['resume_from_checkpoint']=str(checkpoint);cp.write_text(json.dumps(config))
    c,_,restored=approved_config(cp)
    assert restored==identity and c['resume_from_checkpoint']==str(checkpoint)
    (checkpoint/'scheduler.pt').unlink()
    with pytest.raises(ValueError,match='Incomplete'):approved_config(cp)


def test_official_metadata_local_validation(tmp_path):
    import hashlib
    from scripts.pin_e0_snapshot_v4 import verify_local
    weight=tmp_path/'model.safetensors';weight.write_bytes(b'tiny synthetic fixture only')
    config=tmp_path/'config.json';config.write_bytes(b'{}')
    metadata={'repo':BASE,'revision':REV,'files':[{'name':weight.name,'lfs_sha256':sha(weight),'git_blob_id':None},
                 {'name':config.name,'lfs_sha256':None,'git_blob_id':hashlib.sha1(b'blob 2\0{}').hexdigest()}]}
    result=verify_local(tmp_path,metadata)
    assert result['files'][weight.name]==sha(weight)
    weight.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='SHA mismatch'):verify_local(tmp_path,metadata)


def test_hub_snapshot_logical_paths_allow_blob_symlink_resolution(tmp_path,monkeypatch):
    from scripts.pin_e0_snapshot_v4 import verify_local
    from src.training.rollout_pipeline_v4 import validate_model
    weight=tmp_path/'model.safetensors';weight.write_bytes(b'synthetic-cache-target')
    expected=sha(weight)
    metadata={'repo':BASE,'revision':REV,'files':[{'name':'model.safetensors','lfs_sha256':expected,'git_blob_id':None}]}
    original=Path.resolve
    def physical_location(path,*args,**kwargs):
        if path.name=='model.safetensors':return tmp_path.parent/'blobs'/'outside-snapshot-blob'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',physical_location)
    identity=verify_local(tmp_path,metadata)
    ip=tmp_path/'identity.json';ip.write_text(json.dumps(identity))
    assert validate_model(tmp_path,ip)['files']['model.safetensors']==expected
    metadata['files'][0]['name']='../escape.safetensors'
    with pytest.raises(ValueError,match='Escaping'):verify_local(tmp_path,metadata)


def test_local_cli_no_drive_and_no_implicit_gpu(tmp_path,monkeypatch):
    import src.training.approved_hf_v4 as runner
    release=tmp_path/'release';(release/'configs').mkdir(parents=True)
    path=release/'configs/E2.json'
    path.write_text(json.dumps({'stage':'rl','vision':False,'initial_state':'instruct','model':'${E0_MODEL_PATH}',
        'data_manifest':'${APPROVED_RELEASE_ROOT}/manifest.json','output_dir':'${PROJECT_DRIVE_ROOT}/obsolete','experiment':'E2_shared'}))
    calls=[];monkeypatch.setattr(runner,'run',lambda p,m:calls.append(json.loads(Path(p).read_text())))
    output=tmp_path/'any local directory'
    monkeypatch.setattr(sys,'argv',['train','--config',str(path),'--output-root',str(output),'--e0-model-path',str(tmp_path/'local E0'),'--swanlab-mode','disabled'])
    runner.cli('rl',False)
    assert Path(calls[0]['output_dir'])==output.resolve()/'E2_shared'
    assert calls[0]['data_manifest']==str(release.resolve())+'/manifest.json'


def test_v4_rejects_unapproved_release(tmp_path):
    from src.training.guards_v4 import approved_config
    config=json.loads(Path('configs/project1_raw_grpo_smoke.json').read_text())
    config.update(arm='raw',initial_state='instruct',model=BASE,model_revision=REV,max_prompt_length=1752)
    cp=tmp_path/'config.json';cp.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='conditional rollout approval'):approved_config(cp)


def test_tail_repair_backups_and_preserves_valid_records(tmp_path):
    from scripts.repair_trial_journal_v4 import repair
    journal=tmp_path/'sampling/raw/shard-00000.jsonl'
    append(journal,{'sample_id':'a','trial':0})
    original=journal.read_bytes()+b'{"broken":'
    journal.write_bytes(original)
    result=repair(journal,tmp_path/'backup')
    assert Path(result['backup']).read_bytes()==original
    assert jsonl(journal)==[{'sample_id':'a','trial':0}]
    with pytest.raises(ValueError):repair(tmp_path/'candidate_pool.jsonl',tmp_path/'backup')


def test_swanlab_real_offline(tmp_path):
    from src.training.rollout_pipeline_v4 import Metrics
    logger=Metrics(tmp_path,mode='offline',name='cpu-test-only',config={'test_fixture':True})
    logger.log({'loss':.5,'secret':'excluded-string'},1)
    logger.finish();logger.finish()
    assert jsonl(tmp_path/'metrics.jsonl')==[{'step':1,'metrics':{'loss':.5}}]
    assert (tmp_path/'swanlog').exists()


def test_no_online_login_without_environment_key(tmp_path,monkeypatch):
    from src.training.rollout_pipeline_v4 import Metrics
    monkeypatch.delenv('SWANLAB_API_KEY',raising=False)
    with pytest.raises(ValueError,match='securely'):Metrics(tmp_path,mode='online')


def test_mock_backend_crash_reuses_raw_and_completes(tmp_path,monkeypatch):
    import importlib.metadata
    import scripts.sample_rl_candidates_v4 as sampler
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda index:'CPU-TEST-MOCK-NOT-A-GPU-RUN')
    from types import ModuleType
    prep=tmp_path/'prep';prep.mkdir()
    rows=[{'sample_id':'a','problem':'Find 2','reference_answer':'2','split':'train','training_approved':False}]
    candidate=prep/'pool.jsonl';candidate.write_text(json.dumps(rows[0])+'\n')
    tokenpath=prep/'prompt_tokens.jsonl';tokenpath.write_text('{"sample_id":"a","prompt_token_ids":[1,3]}\n')
    seal=prep/'preparation_seal.json';seal.write_text('{}')
    modelid={'repo':BASE,'revision':REV,'verified_against_hf_lfs':True,'test_fixture':True}
    monkeypatch.setattr(sampler,'load_prepared',lambda *a:(rows,{'prompt_tokens_sha256':sha(tokenpath),'grader_identity_sha256':'fixture'}))
    monkeypatch.setattr(sampler,'validate_model',lambda *a:modelid)
    real_version=importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda n:'0.11.2' if n=='vllm' else real_version(n))
    calls=[]
    class Engine:
        def __init__(self,**kwargs):calls.append('init')
        def generate(self,prompts,params,**kwargs):
            calls.append('generate')
            return [SimpleNamespace(outputs=[SimpleNamespace(text=r'\boxed{2}' if i==0 else r'\boxed{3}',token_ids=[3,2],finish_reason='stop',stop_reason=None)]) for i,_ in enumerate(prompts)]
    fake=ModuleType('vllm');fake.LLM=Engine;fake.SamplingParams=lambda **k:SimpleNamespace(**k)
    monkeypatch.setitem(sys.modules,'vllm',fake)
    class FakeWorker:
        def check(self,text,answer):return {'status':'equivalent' if text==r'\boxed{2}' else 'mismatch','correct':text==r'\boxed{2}'}
        def close(self):pass
    monkeypatch.setattr(sampler,'Worker',FakeWorker)
    root=tmp_path/'arbitrary local output'
    argv=['sample','--candidates',str(candidate),'--preparation-seal',str(seal),'--model-path',str(tmp_path/'dummy'),
          '--model-identity',str(tmp_path/'dummy.json'),'--output-root',str(root),'--num-samples','4','--num-shards','1','--swanlab-mode','disabled']
    monkeypatch.setattr(sys,'argv',argv)
    real_grade=sampler.grade_one
    def crash(*args):raise RuntimeError('simulated interruption after raw commit')
    monkeypatch.setattr(sampler,'grade_one',crash)
    with pytest.raises(RuntimeError,match='simulated'):sampler.main()
    assert len(jsonl(root/'raw/shard-00000.jsonl'))==4
    before=sha(root/'raw/shard-00000.jsonl')
    monkeypatch.setattr(sampler,'grade_one',real_grade)
    sampler.main()
    assert calls==['init','generate'] and sha(root/'raw/shard-00000.jsonl')==before
    assert json.loads((root/'sampling_complete.json').read_text())['complete']
    # 4/8 mode switch cannot reuse persisted predictions under another protocol.
    monkeypatch.setattr(sys,'argv',[*argv[:argv.index('--num-samples')],'--num-samples','8',*argv[argv.index('--num-samples')+2:]])
    with pytest.raises(ValueError,match='identity changed'):sampler.main()


def test_pipeline_approves_only_after_sampling_subprocess(tmp_path,monkeypatch):
    import scripts.run_rl_screening_pipeline_v4 as pipeline
    root=tmp_path/'arbitrary root';approval=tmp_path/'new release';calls=[]
    def fake_run(args,check):
        calls.append(args)
        if 'scripts.sample_rl_candidates_v4' in args:
            root.mkdir(exist_ok=True);(root/'sampling_complete.json').write_text('{"complete":true}')
    monkeypatch.setattr(pipeline.subprocess,'run',fake_run)
    monkeypatch.setattr(sys,'argv',['pipeline','--output-root',str(root),'--approval-output-root',str(approval),'--num-samples','4'])
    pipeline.main()
    assert 'scripts.sample_rl_candidates_v4' in calls[0] and 'scripts.approve_rl_pool_v4' in calls[1]


def test_v4_reference_cli_tiny(tmp_path,monkeypatch):
    from scripts.verify_reference_v4 import main
    monkeypatch.setattr(sys,'argv',['reference','--tiny','--output-root',str(tmp_path/'arbitrary output')])
    main()
    result=json.loads((tmp_path/'arbitrary output/reference_check.json').read_text(encoding='utf-8'))
    assert result['status']=='passed' and result['updated_disabled_max_abs']==0
    assert result['real_7b_gpu']=='未验证'
    monkeypatch.setattr(sys,'argv',['reference','--config',str(tmp_path/'not-read.json'),'--rl-adapter',str(tmp_path/'missing-adapter'),'--output-root',str(tmp_path/'other')])
    with pytest.raises(ValueError,match='existing local'):main()
