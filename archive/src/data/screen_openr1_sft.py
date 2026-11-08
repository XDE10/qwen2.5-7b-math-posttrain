"""Per-trajectory OpenR1 screening, fixed target tokenizer and protected splits.

CPU only. Upstream math-verify flags are selection evidence, not an independent
proof of every reasoning step. Accepted output remains pending human audit.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from src.data.prepare_math_data import SimilarityIndex, fingerprint, write_jsonl
from src.data.screening_io import rows, sha, json_write, source_files, parquet_rows
from src.evaluation.math_verifier import _braced, InvalidAnswer

PROMPT = 'Solve the math problem step by step. Keep reasoning concise and put the final answer in \\boxed{}.'


def sealed_math(root=Path('.')):
    manifest = json.loads((root/'data/math_manifest.json').read_text(encoding='utf-8'))
    files = {}
    for key in ['math_train','math_dev','math500_test','gsm8k_test']:
        meta = manifest['files'][key]
        p = root/meta['path']
        if sha(p)!=meta['sha256']: raise ValueError('Sealed math file changed: '+key)
        files[key] = list(rows(p))
    if len(files['math_dev'])!=256 or len(files['math_train'])!=7184:
        raise ValueError('Unexpected shared math identity')
    return manifest,files


def answer_is_complete(text):
    if not isinstance(text,str) or not text.strip(): return False
    if text.count('<think>')!=text.count('</think>') or text.count('<think>')>1: return False
    tail = text.split('</think>')[-1]
    matches = list(re.finditer(r'\\boxed\s*\{',tail))
    if not matches: return False
    try:
        for match in matches:
            body,_ = _braced(tail,match.end()-1)
            if not body.strip(): return False
    except InvalidAnswer:
        return False
    return True


def trajectory_candidates(row,tokenizer,min_tokens=500,max_tokens=1600,total_limit=4096):
    generations,complete,correct = (row.get(k) for k in ['generations','is_reasoning_complete','correctness_math_verify'])
    if not all(isinstance(x,list) for x in [generations,complete,correct]) or not generations or not(len(generations)==len(complete)==len(correct)):
        return [],[{'trajectory_index':None,'reason':'unaligned_or_missing_flags'}]
    finishes = row.get('finish_reasons')
    if finishes is not None and (not isinstance(finishes,list) or len(finishes)!=len(generations)):
        return [],[{'trajectory_index':None,'reason':'unaligned_finish_reasons'}]
    accepted,audit = [],[]
    for i,text in enumerate(generations):
        reason = None
        if correct[i] is not True: reason='not_math_verify_true'
        elif complete[i] is not True: reason='not_reasoning_complete_true'
        elif finishes is not None and finishes[i] not in {None,'stop','eos','eos_token','stop_sequence'}: reason='non_stop_finish'
        elif not answer_is_complete(text): reason='incomplete_text_structure'
        if reason:
            audit.append({'trajectory_index':i,'reason':reason});continue
        count = len(tokenizer.encode(text,add_special_tokens=False))
        if not min_tokens<=count<=max_tokens:
            audit.append({'trajectory_index':i,'reason':'response_length','response_token_count':count});continue
        messages = [{'role':'system','content':PROMPT},{'role':'user','content':row['problem']},
                    {'role':'assistant','content':text}]
        total = len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False))
        if total>total_limit:
            audit.append({'trajectory_index':i,'reason':'sft_total_length','response_token_count':count,'total_token_count':total});continue
        accepted.append({'trajectory_index':i,'solution':text,'response_token_count':count,
                         'total_token_count':total,'source_math_verify':True,'source_reasoning_complete':True,
                         'source_finish_reason':finishes[i] if finishes is not None else None})
        audit.append({'trajectory_index':i,'reason':'eligible','response_token_count':count,'total_token_count':total})
    return accepted,audit


def run(root=Path('.'),target=1500):
    from transformers import AutoTokenizer
    source = source_files('openr1',root)
    if not all('path' in f for f in source['files']): raise ValueError('Complete source download first')
    math_manifest,protected = sealed_math(root)
    out = root/'data/project1_openr1'
    out.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(root/'data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    token_audit = json.loads((root/'reports/tokenizer_alignment_report.json').read_text(encoding='utf-8'))
    token_identity = next(m for m in token_audit['models'] if m['model']=='Qwen/Qwen2.5-7B-Instruct')
    identity = {'source_revision':source['revision'],'target':target,'range':[500,1600],'total_limit':4096,
                'protected_hashes':{k:math_manifest['files'][k]['sha256'] for k in protected},
                'tokenizer_revision':token_identity['revision'],'tokenizer_files':token_identity['files'],
                'script_sha256':sha(__file__),'seed':42}
    identity_hash = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    checkpoint = out/'checkpoint.json'
    state = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else None
    if state and state['identity_hash']!=identity_hash: raise ValueError('Refuse mismatched screening resume')
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if old['identity_hash']==identity_hash:
            print('OpenR1 sealed output already complete',flush=True);return old
    index = SimilarityIndex(.85)
    for key in ['math_dev','math500_test','gsm8k_test']:
        for r in protected[key]: index.add(r)
    selected = list(rows(out/'selected.partial.jsonl')) if state else []
    seen = SimilarityIndex(.85)
    for row in selected: seen.add(row)
    scanned = state['scanned_rows'] if state else 0
    counts = Counter(state['counts']) if state else Counter()
    # Deterministic shard permutation; stop at target. Coverage is disclosed.
    source = {**source,'files':list(source['files'])}
    random.Random(42).shuffle(source['files'])
    audit_path = out/'trajectory_audit.jsonl'
    if state:
        # Keep only committed audit bytes after an interrupted checkpoint.
        with audit_path.open('r+b') as f: f.truncate(state['audit_bytes'])
    mode = 'a' if state else 'w'
    with audit_path.open(mode,encoding='utf-8',newline='\n') as audit_file:
        def save():
            audit_file.flush()
            write_jsonl(out/'selected.partial.jsonl',selected)
            json_write(checkpoint,{'identity_hash':identity_hash,'scanned_rows':scanned,'counts':dict(counts),
                                   'audit_bytes':audit_path.stat().st_size})
        try:
            for pos,row in enumerate(parquet_rows(source,root)):
                if pos<scanned: continue
                if len(selected)>=target: break
                scanned = pos+1
                problem = row.get('problem','')
                sid = 'openr1/'+str(row.get('uuid',pos))
                counts['source_rows_scanned']+=1
                eligible,details = trajectory_candidates(row,tokenizer)
                for d in details: counts['trajectory_'+d['reason']]+=1
                reason,overlap = None,None
                if eligible:
                    overlap = index.match(problem)
                    if overlap: reason='protected_overlap'
                    elif seen.match(problem): reason='duplicate_selected_problem'
                if eligible and not reason:
                    chosen = min(eligible,key=lambda x:(x['response_token_count'],x['trajectory_index']))
                    result = {**chosen,'sample_id':sid,'problem':problem,'source':source['repo'],
                              'source_revision':source['revision'],'source_split':'train','split':'train',
                              'reference_answer':row.get('answer'),'family_id':fingerprint(problem),
                              'problem_hash':fingerprint(problem),'domain':'math',
                              'problem_type':row.get('problem_type'),'question_type':row.get('question_type'),
                              'source_category':row.get('source'),'quality_status':'source_verified_pending_human_audit'}
                    selected.append(result);seen.add(result);counts['accepted_unique_problems']+=1
                elif reason: counts[reason]+=1
                audit_file.write(json.dumps({'sample_id':sid,'trajectories':details,'problem_rejection':reason,
                                             'overlap_evidence':overlap},ensure_ascii=False)+'\n')
                if scanned%250==0:
                    save()
                    print(f'OpenR1 scanned={scanned}, selected={len(selected)}/{target}',flush=True)
        finally: save()
    output = out/'sft_train.jsonl'
    write_jsonl(output,selected)
    audit_candidates = sorted(selected,key=lambda r:hashlib.sha256((r['sample_id']+'human-audit').encode()).hexdigest())[:50]
    write_jsonl(out/'human_audit_50.jsonl',[{**r,'human_verdict':None,'human_notes':''} for r in audit_candidates])
    manifest = {**identity,'identity_hash':identity_hash,'selected_rows':len(selected),'target_met':len(selected)>=target,
                'source_rows_scanned':scanned,'scan_complete':counts['source_rows_scanned']==93733,
                'scan_rule':'Seed-42 shuffled shards, original order within shard; stop once target unique questions accepted. Not a full-source prevalence survey.',
                'counts':dict(counts),'source_category_counts':dict(Counter(r['source_category'] for r in selected)),
                'response_length_buckets':dict(Counter('500-799' if r['response_token_count']<800 else '800-1199' if r['response_token_count']<1200 else '1200-1600' for r in selected)),
                'output':{'path':output.relative_to(root).as_posix(),'sha256':sha(output)},
                'manual_audit':'50 samples exported; not yet manually reviewed; source correctness flags are not step-level proofs',
                'no_api_calls':True,'no_model_weights':True}
    json_write(out/'manifest.json',manifest)
    json_write(root/'reports/openr1_screening_report.json',manifest)
    print(f'OpenR1 finished: {len(selected)} unique candidates; scanned {scanned}',flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',type=int,default=1500)
    args=p.parse_args();run(target=args.target)
