"""Offline replay of the fixed four-model tokenizer audit; no weights/network."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from src.teachers.tokenizer_audit import compare_vocabularies,examples
from src.data.screening_io import sha,json_write


def run():
    report_path=Path('reports/tokenizer_alignment_report.json')
    report=json.loads(report_path.read_text(encoding='utf-8'));tokenizers=[];verified=[]
    for model in report['models']:
        folder=Path('data/tokenizers')/model['model'].split('/')[-1]
        for name,digest in model['files'].items():
            if sha(folder/name)!=digest:raise ValueError('Tokenizer file identity changed: '+model['model']+'/'+name)
        config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        assert config['vocab_size']==model['config_vocab_size']
        tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
        assert len(tokenizer)==model['len_tokenizer']
        assert tokenizer.get_added_vocab()==model['added_tokens']
        assert tokenizer.all_special_ids==model['special_token_ids']
        assert hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()==model['chat_template_sha256']
        tokenizers.append(tokenizer)
        verified.append({'model':model['model'],'revision':model['revision'],
                         'config_output_rows':config['vocab_size'],'valid_token_ids':len(tokenizer),
                         'outside_common_support_rows':config['vocab_size']-report['common_support']['size']})
    support,differences=compare_vocabularies([t.get_vocab() for t in tokenizers])
    assert support['ids']==report['common_support']['ids'] and not differences
    original={c['text']:c for c in report['encoding_cases']};results=[]
    for case in examples():
        raw=[t.encode(case['text'],add_special_tokens=False) for t in tokenizers]
        chat=[t.apply_chat_template([{'role':'user','content':case['text']}],tokenize=True,add_generation_prompt=True) for t in tokenizers]
        assert raw==original[case['text']]['input_ids']
        assert all(ids==raw[0] for ids in raw)
        results.append({'category':case['category'],'raw_identical':True,
                        'independent_chat_templates_identical':all(ids==chat[0] for ids in chat),
                        'chat_token_counts':[len(ids) for ids in chat]})
    proof={'date':'2026-10-23','offline':True,'original_report_path':report_path.as_posix(),
        'original_report_sha256':sha(report_path),'models':verified,'case_count':len(results),
        'case_categories':dict(Counter(c['category'] for c in results)),
        'all_raw_encodings_identical':True,'all_chat_encodings_identical':all(c['independent_chat_templates_identical'] for c in results),
        'mapping_difference_count':0,'common_support_size':support['size'],
        'exclusive_upper_bound':support['exclusive_upper_bound'],'cases':results,
        'logits_rule':{'effective_ids':'0 <= token_id < 151665','generation':'Mask output rows >=151665 BEFORE sampling/softmax for each model.',
          'kl':'FP32 log_softmax(logits[..., :151665], dim=-1) independently for student and teacher; this is a KL conditioned on common support.',
          'prefix':'Render once with the frozen student template; use identical token-ID prefixes for every teacher; do not independently apply the Math teacher template.',
          'removed_mass':'Before masking, compute 1 - exp(logsumexp(valid_logits)-logsumexp(all_logits)); actual trained-model tail mass remains unmeasured.',
          'temperature':'Same documented temperature for teacher/student; baseline T=1.'},
        'scope':'Full cached token->ID maps and 64 examples replayed. No model forward, GPU compatibility or real-logit mass acceptance.',
        'script_sha256':sha(__file__)}
    json_write('reports/tokenizer_alignment_revalidation_20261023.json',proof)
    print('Offline tokenizer revalidation:',len(results),'cases; common support',support['size'],
          '; independent chat templates identical:',proof['all_chat_encodings_identical'],flush=True)
    return proof


if __name__=='__main__':run()
