"""Freeze the first 1000 seeded, complete <=8192-token original AIMO2 traces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.aimo_sft_v25.common import encode,read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    a=p.parse_args()
    source=Path(a.source)
    expected='df3d6c3efafae50db8784a96d13e9f61525639f0a22d426abf577c42f39f6f28'
    if sha(source)!=expected:
        raise ValueError('Source differs from the frozen sft-4k version')
    report=ROOT/'reports/aimo_sft_20261026_v25'
    target=report/'train.jsonl'
    if target.exists():
        raise ValueError('Frozen selection already exists; do not overwrite')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/DeepSeek-R1-Distill-Qwen-7B',local_files_only=True)
    data=read(source);order=list(range(len(data)));random.Random(42).shuffle(order)
    selected=[];counts=Counter();lengths=[];supervised=[]
    for index in order:
        item=data[index];conv=item.get('conversations',[])
        if len(conv)!=2 or [m.get('role',m.get('from')) for m in conv]!=['user','assistant']:
            counts['unexpected_structure']+=1;continue
        question,solution=conv[0]['value'],conv[1]['value']
        if not question.strip() or not solution.strip():
            counts['empty']+=1;continue
        try:
            encoded=encode(tok,question,solution,8192)
        except OverflowError:
            counts['over_8192']+=1;continue
        except ValueError:
            counts['missing_complete_think']+=1;continue
        question_id=hashlib.sha256(question.encode('utf-8')).hexdigest()
        identity=hashlib.sha256((str(index)+'\0'+question).encode('utf-8')).hexdigest()
        selected.append({'sample_id':identity,'question_id':question_id,'source_index':index,'question':question,'solution':solution,
            'total_tokens':encoded['length'],'supervised_tokens':encoded['supervised_tokens'],
            'training_approved':True,
            'approval_basis':'Human user explicitly requested direct SFT from original sft-4k; this new release is separate from v21 sampling-only rows'})
        lengths.append(encoded['length']);supervised.append(encoded['supervised_tokens'])
        if len(selected)==1000:
            break
    if len(selected)<500:
        raise ValueError('Fewer than 500 complete eligible traces; do not truncate')
    report.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    save(report/'manifest.json',{'dataset':'imagination-research/aimo2-datasets','file':'sft-4k.json',
        'revision':'237f575f59fda9b79de5f4f9e9c7860099f84349','source_sha256':expected,
        'approved_training_files':[target.relative_to(ROOT).as_posix()],
        'train_sha256':sha(target),'train_count':len(selected),'selection_seed':42,
        'unique_question_count':len({r['question_id'] for r in selected}),
        'eligibility':'Two turns; complete original think; full prompt+response+EOS <=8192; no truncation',
        'decontamination':'Reuse user-confirmed existing decontamination; no repeated decontamination run',
        'scanned_until_target':sum(counts.values())+len(selected),'skipped':dict(counts),
        'tokens':{'min':min(lengths),'max':max(lengths),'mean':sum(lengths)/len(lengths),
                  'supervised_sum':sum(supervised)},'approval_basis':'Direct human instruction in current conversation',
        'old_approved_manifest_modified':False})
    print(json.dumps(read(report/'manifest.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
