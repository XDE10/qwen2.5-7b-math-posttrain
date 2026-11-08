"""CPU-only chunked MiniLM cosine screening; preserve every sealed source."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from src.training.frozen_verifier_v4 import sha,digest


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def encode(texts,tok,model,batch_size=32):
    chunks=[];owners=[];counts=[]
    for i,text in enumerate(texts):
        ids=tok.encode(text,add_special_tokens=False)
        # Non-overlapping chunks preserve every token; no silent truncation.
        pieces=[ids[j:j+224] for j in range(0,len(ids),224)] or [[]]
        counts.append(len(pieces))
        for part in pieces:
            chunks.append(tok.build_inputs_with_special_tokens(part));owners.append(i)
    sums=np.zeros((len(texts),model.config.hidden_size),dtype=np.float32)
    for start in range(0,len(chunks),batch_size):
        part=chunks[start:start+batch_size]
        batch=tok.pad({'input_ids':part},padding=True,return_attention_mask=True,return_tensors='pt')
        with torch.inference_mode():
            h=model(**batch).last_hidden_state
            mask=batch['attention_mask'].unsqueeze(-1)
            vectors=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
            vectors=torch.nn.functional.normalize(vectors,dim=-1).cpu().numpy()
        for owner,v in zip(owners[start:start+batch_size],vectors):sums[owner]+=v
        if start%1024==0:print(json.dumps({'chunks_encoded':min(start+batch_size,len(chunks)),'chunks_total':len(chunks)}),flush=True)
    sums/=np.maximum(np.linalg.norm(sums,axis=1,keepdims=True),1e-12)
    return sums,counts


def nearest(query,target,k=3):
    sims=query@target.T
    idx=np.argsort(-sims,axis=1,kind='stable')[:,:min(k,len(target))]
    return idx,np.take_along_axis(sims,idx,axis=1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-root',required=True)
    p.add_argument('--model-record',required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();out=Path(a.output_root).resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads)
    download=json.loads(Path(a.model_record).read_text(encoding='utf-8'))
    model_path=Path(download['files']['model.safetensors']).parent
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModel.from_pretrained(model_path,local_files_only=True).to('cpu').eval()
    paths={'sft':'data/portable_release/project1_raw_paired.jsonl','rl':str(out/'rl_candidates_verifier_supported_v2.jsonl'),
           'dev':'data/math/math_dev.jsonl','math500':'data/math/math500_test.jsonl','gsm8k':'data/math/gsm8k_test.jsonl'}
    data={k:read_rows(v) for k,v in paths.items()}
    embeddings={};chunk_counts={}
    for name,rows in data.items():
        cache=out/f'embedding_{name}.npz'
        ident={'input_sha256':sha(paths[name]),'model_sha256':sha(model_path/'model.safetensors'),'chunk_size':224,'pooling':'chunk-mean then normalized document-mean'}
        if cache.exists():
            old=np.load(cache,allow_pickle=False)
            if str(old['identity'])!=digest(ident):raise ValueError('Embedding cache identity changed')
            embeddings[name]=old['vectors'];chunk_counts[name]=old['chunks'].tolist()
        else:
            embeddings[name],chunk_counts[name]=encode([r['problem'] for r in rows],tok,model,a.batch_size)
            np.savez(cache,vectors=embeddings[name],chunks=chunk_counts[name],identity=digest(ident))
    # Threshold determined only from training-side paired identity and synthetic family probes.
    base=[r['problem'] for r in data['sft'][:50]]
    rewrite=['Solve this mathematical problem. '+t+' Explain your answer.' for t in base]
    numeric=[re.sub(r'\d+',lambda m:str(int(m[0])+1),t) for t in base]
    probe,_=encode(rewrite+numeric,tok,model,a.batch_size)
    v=embeddings['sft'][:50]
    positive=np.r_[np.ones(50),np.sum(v*probe[:50],axis=1),np.sum(v*probe[50:],axis=1)]
    negative=np.sum(v*embeddings['sft'][347:397],axis=1)
    grid=[{'threshold':t,'controlled_positive_recall':float((positive>=t).mean()),'unrelated_false_positive_rate':float((negative>=t).mean())} for t in [.75,.8,.85,.9,.92,.95,.97,.99]]
    eligible=[g for g in grid if g['controlled_positive_recall']>=.95 and g['unrelated_false_positive_rate']<=.05]
    if not eligible:raise ValueError('No threshold satisfies calibration; do not tune on heldout hits')
    threshold=max(g['threshold'] for g in eligible)
    hits=[];nearest_records=[]
    for source,target in [('sft',t) for t in ('dev','math500','gsm8k')]+[('rl',t) for t in ('dev','math500','gsm8k','sft')]:
        idx,score=nearest(embeddings[source],embeddings[target])
        for row,js,values in zip(data[source],idx,score):
            rec={'source':source,'target':target,'source_id':row['sample_id'],
                 'neighbors':[{'target_id':data[target][int(j)]['sample_id'],'cosine':float(s)} for j,s in zip(js,values)]}
            nearest_records.append(rec)
            for j,s in zip(js,values):
                if float(s)>=threshold:hits.append({'source':source,'target':target,'source_id':row['sample_id'],'target_id':data[target][int(j)]['sample_id'],'cosine':float(s)})
    removed={h['source_id'] for h in hits if h['source']=='rl'}
    lines=Path(paths['rl']).read_bytes().splitlines(keepends=True)
    final=out/'rl_candidates_verifier_supported_semantic_screened_v3.jsonl'
    quarantine=out/'rl_candidates_embedding_quarantine_v3.jsonl'
    final.write_bytes(b''.join(l for l in lines if l.strip() and json.loads(l)['sample_id'] not in removed))
    quarantine.write_bytes(b''.join(l for l in lines if l.strip() and json.loads(l)['sample_id'] in removed))
    (out/'embedding_nearest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in nearest_records),encoding='utf-8')
    report={'model':{'repo':download['repo'],'revision':download['revision'],'sha256':sha(model_path/'model.safetensors')},
            'method':'mean pooling with mask; full-text 224-token chunks; normalized document mean; exact cosine top3',
            'threshold':threshold,'calibration':grid,'threshold_scope':'training-controlled probes only; not independent semantic duplicate recall',
            'input_identity':{k:{'rows':len(data[k]),'sha256':sha(v)} for k,v in paths.items()},
            'multi_chunk_documents':{k:sum(n>1 for n in v) for k,v in chunk_counts.items()},'hits':hits,
            'hit_counts':dict(Counter(h['source']+'->'+h['target'] for h in hits)),
            'sft_suspect_ids':sorted({h['source_id'] for h in hits if h['source']=='sft'}),'sft_action':'sealed paired files preserved; any suspect IDs remain unresolved, no automatic SFT approval change',
            'rl_removed_ids':sorted(removed),'rl_removed_count':len(removed),'rl_final_count':len(data['rl'])-len(removed),
            'final_candidate_sha256':sha(final),'training_approved':False,'claim_limit':'completed fixed-rule semantic screening, not a proof of zero contamination or reasoning correctness'}
    (out/'embedding_screening.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['threshold','hit_counts','rl_removed_count','rl_final_count','sft_suspect_ids']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
