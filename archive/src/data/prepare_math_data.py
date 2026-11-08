"""Download pinned public math datasets; seal dev and remove known overlaps.

No GPU or model weights. Conservative lexical 3-gram Jaccard is not semantic
equivalence and cannot establish absence of pretraining contamination.
"""
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from src.evaluation.math_verifier import _braced

SOURCES = {'gsm8k':('openai/gsm8k','main'),
           'math':('DigitalLearningGmbH/MATH-lighteval','default'),
           'math500':('HuggingFaceH4/MATH-500',None)}


def normalize(text):
    text = unicodedata.normalize('NFKC',text).casefold()
    text = text.replace('\\left','').replace('\\right','').replace('\\!','')
    return ' '.join(re.findall(r'\w+|[^\w\s]',text))


def fingerprint(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def shingles(text):
    words = normalize(text).split()
    n = 3 if len(words)>=3 else 1
    return frozenset(tuple(words[i:i+n]) for i in range(len(words)-n+1))


class SimilarityIndex:
    def __init__(self, threshold=.85):
        if not 0<threshold<=1: raise ValueError('threshold must be (0,1]')
        self.threshold = threshold
        self.entries, self.exact, self.raw_exact, self.postings = [], {}, {}, defaultdict(list)

    def add(self, row):
        key = fingerprint(row['problem'])
        grams = shingles(row['problem'])
        idx = len(self.entries)
        self.entries.append((row['sample_id'],grams))
        self.exact.setdefault(key,row['sample_id'])
        self.raw_exact.setdefault(hashlib.sha256(row['problem'].encode()).hexdigest(),row['sample_id'])
        for gram in grams: self.postings[gram].append(idx)

    def match(self, text):
        raw_key = hashlib.sha256(text.encode()).hexdigest()
        if raw_key in self.raw_exact:
            return {'matched_id':self.raw_exact[raw_key],'similarity':1.,'kind':'raw_exact'}
        key = fingerprint(text)
        if key in self.exact:
            return {'matched_id':self.exact[key],'similarity':1.,'kind':'normalized_exact'}
        grams = shingles(text)
        overlap = Counter(i for g in grams for i in self.postings.get(g,()))
        for idx,count in sorted(overlap.items(),key=lambda v:(-v[1],v[0])):
            name,other = self.entries[idx]
            needed = self.threshold*(len(grams)+len(other))/(1+self.threshold)
            if count+1e-12 < needed: continue
            score = count/len(grams|other)
            if score+1e-12>=self.threshold:
                return {'matched_id':name,'similarity':score,'kind':'jaccard_3gram'}
        return None


def partition_math(rows, references, seed=42, dev_size=256, threshold=.85):
    reference_index, seen = SimilarityIndex(threshold), SimilarityIndex(threshold)
    for r in references: reference_index.add(r)
    clean, rejected = [], []
    for row in rows:
        match = reference_index.match(row['problem'])
        reason = 'evaluation_overlap'
        if match is None:
            match = seen.match(row['problem'])
            reason = 'internal_duplicate'
        if match:
            rejected.append({**row,'filter_reason':reason,'overlap_evidence':match})
        else:
            clean.append(row)
            seen.add(row)
    if len(clean)<dev_size: raise ValueError('Not enough clean unique questions for sealed dev')
    order = list(range(len(clean)))
    random.Random(seed).shuffle(order)
    chosen = set(order[:dev_size])
    dev = [{**r,'split':'dev'} for i,r in enumerate(clean) if i in chosen]
    train = [{**r,'split':'train'} for i,r in enumerate(clean) if i not in chosen]
    return train,dev,rejected


def reference_answer(row):
    if row.get('answer') is not None:
        return str(row['answer'])
    solution = row['solution']
    if '####' in solution: return solution.rsplit('####',1)[1].strip()
    answers, position = [], 0
    while match := re.search(r'\\boxed\s*\{',solution[position:]):
        start = position+match.end()-1
        body,end = _braced(solution,start)
        answers.append(body)
        position = end
    if not answers:
        # Two real MATH train records use valid single-token TeX: \\boxed 2.
        # Support only an explicit numeric token here; do not mine reasoning.
        bare = re.findall(r'\\boxed\s+([-+]?\d+(?:\.\d+)?)(?![\w.])',solution)
        if bare: return bare[-1]
        raise ValueError('Missing source reference answer')
    return answers[-1]


def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')


def run(root=Path('.')):
    from datasets import load_dataset, disable_progress_bars
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    disable_progress_bars()
    tokenizer = AutoTokenizer.from_pretrained(root/'data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    api = HfApi()
    all_rows, source_meta = {}, {}
    for key,(name,config) in SOURCES.items():
        revision = api.dataset_info(name).sha
        dataset = load_dataset(name,config,revision=revision,cache_dir=str(root/'data/raw_math_cache'))
        split_rows = {}
        for split,records in dataset.items():
            converted = []
            for i,record in enumerate(records):
                problem = record.get('problem',record.get('question'))
                solution = record.get('solution',record.get('answer',''))
                answer = reference_answer({**record,'solution':solution}) if key!='gsm8k' else solution.rsplit('####',1)[-1].strip()
                converted.append({'sample_id':f'{key}/{split}/{i:05d}','source':name,
                    'source_revision':revision,'source_split':split,'source_id':record.get('unique_id',str(i)),
                    'split':split,'problem':problem,'reference_answer':answer,'solution':solution,
                    'domain':'math','token_count':len(tokenizer.encode(problem+'\n'+solution,add_special_tokens=False)),
                    'filter_reason':None,'problem_hash':fingerprint(problem),'family_id':fingerprint(problem),
                    'raw_problem_sha256':hashlib.sha256(problem.encode()).hexdigest(),
                    'level':record.get('level'),'problem_type':record.get('type',record.get('subject'))})
            split_rows[split] = converted
        all_rows[key] = split_rows
        source_meta[key] = {'repo':name,'config':config,'revision':revision,
                            'raw_counts':{k:len(v) for k,v in split_rows.items()},
                            'features':{k:str(v.features) for k,v in dataset.items()}}
        print('Dataset verified:',name,source_meta[key]['raw_counts'],flush=True)
    references = all_rows['gsm8k']['test']+all_rows['math500']['test']
    train,dev,rejected = partition_math(all_rows['math']['train'],references)
    out = root/'data/math'
    exports = {'math_train':train,'math_dev':dev,'math_quarantine':rejected,
               'gsm8k_train':all_rows['gsm8k']['train'],'gsm8k_test':all_rows['gsm8k']['test'],
               'math500_test':all_rows['math500']['test'],'math_source_test':all_rows['math']['test']}
    files = {}
    for name,rows in exports.items():
        path = out/(name+'.jsonl')
        write_jsonl(path,rows)
        files[name] = {'path':path.relative_to(root).as_posix(),'rows':len(rows),
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {'seed':42,'dev_size':256,'sources':source_meta,'files':files,
                'dedup':{'normalization':'NFKC/casefold/latex spacing/lexical tokens',
                         'similarity':'raw exact SHA256 then normalized SHA256 then token 3-gram Jaccard >= 0.85',
                         'threshold':.85,'threshold_validation':'Synthetic positive/negative unit tests; no claim of calibrated semantic recall.',
                         'quarantine_reasons':dict(Counter(r['filter_reason'] for r in rejected))},
                'known_limits':['Lexical near-duplicate detection cannot prove semantic or pretraining decontamination.',
                                'GSM8K train is preserved source material, not automatically approved SFT data.',
                                'MATH source test overlaps MATH500 by design: never use it for tuning.',
                                'Source answers are preserved, not declared semantically verified.'],
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'token_count_definition':'Qwen2.5-7B tokenizer, problem + newline + full solution; no chat template'}
    (root/'data/math_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v['rows'] for k,v in files.items()}))
    return manifest


if __name__=='__main__': run()
