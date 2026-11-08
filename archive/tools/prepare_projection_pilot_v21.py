"""Freeze 32 length-stratified, unapproved AIMO2 sampling inputs (CPU only)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.projection_pilot_v21.pilot import sha, save
from src.data.decontamination_v2 import Index, exact
from src.training.frozen_verifier_v4 import Worker, grader_identity
from src.training.math_reward_v2 import category
from scripts.math_verify_pilot_worker import boxes

DATA_REV = '237f575f59fda9b79de5f4f9e9c7860099f84349'
DATA_SHA = 'df3d6c3efafae50db8784a96d13e9f61525639f0a22d426abf577c42f39f6f28'
MODEL_REV = '916b56a44061fd5cd7d6a8fb632557ed4f724f60'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-json', required=True)
    p.add_argument('--tokenizer-directory', required=True)
    a = p.parse_args()
    report = ROOT/'reports/projection_pilot_20261026_v21'
    report.mkdir(parents=True, exist_ok=True)
    target = report/'selected32.jsonl'
    if target.exists():
        raise ValueError('Selection already frozen; preserve it and use a new version to resample')
    source = Path(a.source_json)
    if sha(source) != DATA_SHA:
        raise ValueError('Wrong source JSON bytes')
    data = json.loads(source.read_text(encoding='utf-8'))
    from tokenizers import Tokenizer
    directory = Path(a.tokenizer_directory)
    tok = Tokenizer.from_file(str(directory/'tokenizer.json'))
    protected = []
    files = ['data/math/math_dev.jsonl','data/math/math500_test.jsonl','data/math/gsm8k_test.jsonl']
    for name in files:
        for line in (ROOT/name).read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                protected.append({'problem': row['problem'], 'set': name, 'sample_id': row['sample_id']})
    index = Index(protected)
    bins = [(0,4096),(4096,8192),(8192,12288),(12288,16384)]
    counts = Counter()
    candidates = list(range(len(data)))
    random.Random(42).shuffle(candidates)
    selected, audit, seen = [], [], set()
    worker = Worker()
    try:
        for i in candidates:
            row = data[i]
            conv = row['conversations']
            if len(conv) != 2 or [m['role'] if 'role' in m else m.get('from') for m in conv] not in [['user','assistant']]:
                audit.append({'source_index':i,'reason':'unexpected_conversation'});continue
            question, solution = conv[0]['value'], conv[1]['value']
            identity = exact(question)
            if identity in seen:
                audit.append({'source_index':i,'reason':'normalized_duplicate'});continue
            seen.add(identity)
            # This is a fixed conservative screening rule, not a claim of purity.
            hits = index.query(question, 0.50)
            if hits:
                audit.append({'source_index':i,'reason':'protected_lexical_neighbor',
                    'hits':[{'set':protected[j]['set'],'id':protected[j]['sample_id'],'similarity':s} for j,s in hits]});continue
            length = len(tok.encode(solution, add_special_tokens=False).ids)
            b = next((f'{lo+1}-{hi}' for lo,hi in bins if lo < length <= hi), None)
            if b is None or counts[b] >= 8:
                continue
            try:
                values = boxes(solution)
            except ValueError:
                audit.append({'source_index':i,'reason':'unclosed_box'});continue
            reference = row.get('stdans')
            reference_source = 'provided_stdans'
            if reference is None or not str(reference).strip():
                if not values:
                    audit.append({'source_index':i,'reason':'no_reference_or_box'});continue
                reference = values[-1]
                reference_source = 'expert_box_consistency_only_not_independent_gold'
            reference = str(reference).strip()
            embedded = boxes(reference)
            if embedded: reference = embedded[-1]
            check = worker.check(solution,reference)
            status = category(check)
            if status != 'equivalent':
                audit.append({'source_index':i,'reason':'source_verifier_'+status,'result':check});continue
            result = {'sample_id':identity,'source_index':i,'question':question,'solution':solution,
                'source':row.get('source',''),'solution_tokens':length,'length_bin':b,
                'reference_answer':reference,'reference_source':reference_source,
                'source_grade':check,'training_approved':False}
            selected.append(result);counts[b]+=1
            print(f'Selected {len(selected)}/32: source row {i}, {length} tokens, bin {b}',flush=True)
            if len(selected)==32:break
    finally:worker.close()
    if len(selected)!=32:
        save(report/'preparation_failure.json',{'selected':len(selected),'bins':dict(counts),'audit':audit})
        raise ValueError('Could not obtain 8 supported questions per length stratum; do not relax automatically')
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    (report/'selection_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit),encoding='utf-8')
    upstream = report/'upstream'
    upstream.mkdir(exist_ok=True)
    meta = json.load(urllib.request.urlopen('https://api.github.com/repos/aakaran/finetuning-with-sampling/commits/main', timeout=40))
    commit = meta['sha']
    provenance = {}
    for name in ('boost_math.py','constants.py','README.md'):
        url = f'https://raw.githubusercontent.com/aakaran/finetuning-with-sampling/{commit}/{name}'
        raw = urllib.request.urlopen(url,timeout=40).read()
        (upstream/name).write_bytes(raw)
        provenance[name]={'url':url,'sha256':sha(upstream/name)}
    config = {'experiment':'projection_pilot_v21','stage':'inference_only_no_parameter_updates',
        'model':'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B','model_revision':MODEL_REV,
        'data_file':target.relative_to(ROOT).as_posix(),'data_sha256':sha(target),
        'tokenizer_sha256':{n:sha(directory/n) for n in ['tokenizer.json','tokenizer_config.json']},
        'seed':42,'max_new_tokens':8192,'block_num':8,'updates_per_block':[0,2],
        'max_model_len':32768,'gpu_memory_utilization':0.88,
        'acceptance_rule':'strict improvement in mean response logprob under question-only model; approximate, not exact MH',
        'correctness_policy':'unchanged strict-v3 verifier after generation; no approval; no silent deletion or resampling on failure',
        'upstream_commit':commit,'sampling_note':'paper used 32 blocks / 10 updates / 1856 tokens; this is a reduced-budget R1 pilot',
        'training_approved':False}
    save(ROOT/'configs/project1_projection_pilot_v21.json',config)
    seal = {'dataset':{'repo':'imagination-research/aimo2-datasets','revision':DATA_REV,'file':'sft-4k.json','sha256':DATA_SHA,'rows':len(data)},
        'selection':{'rows':32,'bins':dict(counts),'answer_token_sum':sum(r['solution_tokens'] for r in selected),
            'reference_sources':dict(Counter(r['reference_source'] for r in selected)),'selected_sha256':sha(target)},
        'protected_inputs':{n:sha(ROOT/n) for n in files},'lexical_threshold':0.50,
        'decontamination_scope':'exact/normalized and char-5gram Jaccard against project dev/MATH500/GSM8K-test; not semantic or pretraining contamination certification',
        'upstream':provenance,'grader_identity':grader_identity(),
        'original_release_manifest_sha256':sha(ROOT/'data/portable_release/manifest.json'),
        'training_approved':False,'gpu_run':'not performed'}
    save(report/'preparation.json',seal)
    print(json.dumps(seal['selection'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
