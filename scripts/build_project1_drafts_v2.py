"""Generate unfrozen, execution-blocked experiment drafts. Never changes release."""
import json
import hashlib
import math
import random
from pathlib import Path
from transformers import AutoTokenizer
from src.training.approved_hf import encode_sft
from src.data.approved_training_inputs_v2 import load_approved

OUT = Path('reports/handover_20261024_v2')
BASE = 'Qwen/Qwen2.5-7B-Instruct'
REV = 'a09a35458c702b33eeacc393d103063234e8bc28'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def simulate(lengths, target):
    total = 0; exposures = 0; updates = 0
    for epoch in range(20):
        order = list(range(len(lengths)))
        random.Random(42+epoch).shuffle(order)
        for start in range(0, len(order), 16):
            batch = order[start:start+16]
            total += sum(lengths[i] for i in batch)
            exposures += len(batch); updates += 1
            if total >= target:
                return dict(optimizer_steps=updates, exposures=exposures, effective_epochs=exposures/1500, supervised_tokens=total, overshoot=total-target)
    raise ValueError('Budget not reached')


def main():
    manifest = 'data/portable_release/manifest.json'
    tok = AutoTokenizer.from_pretrained('data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    encoded = {arm: encode_sft(load_approved(manifest,f'project1_{arm}_paired.jsonl'),tok,4096 if arm=='raw' else 3800) for arm in ['raw','compact']}
    lengths = {arm:[sum(x!=-100 for x in row['labels'][1:]) for row in data] for arm,data in encoded.items()}
    target = sum(lengths['raw'])
    totals = {arm:sum(v) for arm,v in lengths.items()}
    plans = {
        'equal_exposure': {'draft':True, 'status':'草稿，未冻结', 'epochs':1, 'examples':1500, 'effective_batch':16,
                           'expected_steps_per_arm':94, 'stop_rule':'one complete epoch including partial final batch; measure actual token exposure', 'supervised_tokens':totals},
        'equal_supervised_tokens': {'draft':True, 'status':'草稿，未冻结', 'target_supervised_tokens':target, 'effective_batch':16,
                                    'stop_rule':'stop only after an optimizer update reaches or exceeds target; keep all samples/EOS intact; report overshoot; never truncate labels to hit budget',
                                    'planning_order':'random.Random(42+epoch).shuffle, single device, batches of16, flush partial batch at each epoch',
                                    'plans':{arm:simulate(v,target) for arm,v in lengths.items()},
                                    'runtime_status':'未验证：current HF sampler is not this planning order; exact token counter and sampler must be implemented and tested before freezing'}
    }
    for policy, plan in plans.items():
        folder = OUT/'formal_v1'/policy
        write(folder/'protocol.draft.json',plan)
        release=json.loads(Path(manifest).read_text())
        math_manifest=json.loads(Path('data/math_manifest.json').read_text())
        write(folder/'identities.draft.json',{'draft':True,'status':'草稿，未冻结',
              'release_manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              'training_files':{k:release['files'][k] for k in release['approved_training_files']},
              'evaluation_files':{k:math_manifest['files'][k] for k in ('math_dev','math500_test','gsm8k_test')},
              'rl_pool':'not selected or approved'})
        matrix=[]
        for stage in ['E0','E1','E2','E3','E4']:
            for arm in (['raw','compact'] if stage in ('E1','E3','E4') else ['shared']):
                entry={'experiment':stage,'arm':arm,'draft':True,'status':'草稿，未冻结', 'seed':42,
                       'model':BASE if stage in ('E0','E1','E2') else f'${{PROJECT_DRIVE_ROOT}}/runs/{policy}/E1_{arm}/merged',
                       'model_revision':REV if stage in ('E0','E1','E2') else 'local-merged-with-weight-manifest',
                       'base_model_revision':REV,'tokenizer_path':'data/tokenizers/Qwen2.5-7B-Instruct',
                       'system_prompt':r'Please reason step by step, and put your final answer within \boxed{}.',
                       'start': 'E0' if stage in ('E0','E1','E2') else f'E1_{arm}',
                       'data':f'project1_{arm}_paired.jsonl' if stage=='E1' else ('math_dev; final MATH-500/GSM8K after freeze' if stage=='E0' else 'RL_POOL_PENDING_SEPARATE_APPROVAL'),
                       'grader':'strict-v3 + audited-status-v2; selection scorer NOT FROZEN; relation-capable scoring required before RL selection',
                       'generation':{'do_sample':False,'max_new_tokens':2048,'max_prompt_length':1752} if stage=='E0' else (None if stage=='E1' else {'do_sample':True,'temperature':.8,'top_p':.95,'top_k':0,'max_new_tokens':2048,'max_prompt_length':1752,'num_generations':4}),
                       'algorithm':None if stage=='E0' else ('sft' if stage=='E1' else 'dr_grpo' if stage=='E4' else 'grpo'),
                       'reference':None if stage in ('E0','E1') else ('E0' if stage=='E2' else f'exact shared E1_{arm} weights'),
                       'training_budget':None if stage=='E0' else (plan if stage=='E1' else {'pilot_updates':20,'common_checkpoints':[100,200],'extensions':[400,600],'extension_rule':'dev-only predeclared paired extension; fixed scheduler horizon;未验证'}),
                       'notes':'E2 shared across arms; E3/E4 share the identical E1 within each arm; no new approval'}
                matrix.append(entry)
                if stage=='E0':write(folder/'E0_shared.draft.json',entry)
                if stage!='E0':
                    cfg={'draft':True,'status':'草稿，未冻结','stage':'sft' if stage=='E1' else 'rl','vision':False,'arm':arm if arm!='shared' else 'raw',
                         'model':BASE if stage in ('E1','E2') else f'${{PROJECT_DRIVE_ROOT}}/runs/{policy}/E1_{arm}/merged',
                         'model_revision':REV if stage in ('E1','E2') else 'local-merged-with-weight-manifest',
                         'tokenizer_path':'data/tokenizers/Qwen2.5-7B-Instruct','data_manifest':manifest,'data_file':entry['data'],
                         'output_dir':f'${{PROJECT_DRIVE_ROOT}}/runs/{policy}/{stage}_{arm}', 'seed':42,'batch_size':1,'gradient_accumulation_steps':16 if stage=='E1' else 8,
                         'learning_rate':1e-5 if stage=='E1' else 5e-6,'lora_r':16,'lora_alpha':32,'lora_dropout':0,'save_steps':20,'max_steps':94 if policy=='equal_exposure' and stage=='E1' else plans['equal_supervised_tokens']['plans'].get(arm,{'optimizer_steps':200})['optimizer_steps'] if stage=='E1' else 200,
                         'max_length':4096 if stage=='E1' else 3800,'export_merged_base':stage=='E1','budget_protocol':plan,'matrix_entry':entry}
                    if stage!='E1':cfg.update(initial_state='instruct' if stage=='E2' else 'sft',max_prompt_length=1752,max_new_tokens=2048,temperature=.8,top_p=.95,num_generations=4,generation_batch_size=4,beta=.04,algorithm=entry['algorithm'],expected_sft='PIN_AFTER_SFT_EXPORT' if stage!='E2' else None)
                    write(folder/f'{stage}_{arm}.draft.json',cfg)
        write(folder/'experiment_matrix.draft.json',matrix)
    # v2 engineering smoke remains a separate, non-formal configuration family.
    for arm in ('raw','compact'):
        original=json.loads(Path(f'configs/project1_{arm}_sft_smoke.json').read_text())
        original.update(arm=arm,output_dir=f'${{PROJECT_DRIVE_ROOT}}/runs/v2_{arm}_sft_smoke')
        write(OUT/'colab/configs'/f'project1_{arm}_sft_smoke_v2.json',original)
        for algo in ('grpo','dr_grpo'):
            cfg=json.loads(Path(f'configs/project1_{arm}_{algo}_smoke.json').read_text())
            cfg.update(arm=arm,max_prompt_length=1752,max_length=3800,temperature=.8,top_p=.95,
                       model=f'${{PROJECT_DRIVE_ROOT}}/runs/v2_{arm}_sft_smoke/merged',output_dir=f'${{PROJECT_DRIVE_ROOT}}/runs/v2_{arm}_{algo}_smoke',
                       expected_sft={'arm':arm,'output_dir':f'${{PROJECT_DRIVE_ROOT}}/runs/v2_{arm}_sft_smoke',
                                     'data_sha256':json.loads(Path(manifest).read_text())['files'][f'project1_{arm}_paired.jsonl'],'identity_sha256':'PIN_AFTER_SFT_EXPORT','weight_manifest_sha256':'PIN_AFTER_SFT_EXPORT'})
            write(OUT/'colab/configs'/f'project1_{arm}_{algo}_smoke_v2.json',cfg)
    write(OUT/'formal_v1/budget_totals.json',{'supervised_token_totals':totals,'ratio':totals['raw']/totals['compact'],'planning':plans})


if __name__=='__main__':main()
