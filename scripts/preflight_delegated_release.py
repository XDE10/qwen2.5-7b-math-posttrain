"""Real approved-row/CPU-tokenizer preflight; not GPU training or model smoke."""
import json
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from src.data.approved_training_inputs import load_approved,model_view
from src.teachers.opd_contract import render_student_prompt
from scripts.compact_distillation_v2 import ROOT
from src.synthesis.deepseek_pilot import dump


def run():
    base=ROOT/'data/delegated_ai_releases_20261024';manifest=base/'manifest.json'
    meta=json.loads(manifest.read_text(encoding='utf-8'));student=AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-1.5B-Instruct',local_files_only=True)
    math=AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    summary={}
    for name in meta['approved_training_files']:
        rows=load_approved(manifest,name);counts=[]
        profile='opd' if name.startswith('project2') else 'math_sft' if name.startswith('project1') else 'perception_sft'
        for row in rows:
            view=model_view(row,profile)
            assert not any(k in view['input'] for k in ['reference_answer','solution','raw_review','audit_only_grounded_facts'])
            if profile=='opd':
                ids=render_student_prompt(student,view['input']['prompt'],view['input']['domain']);counts.append(len(ids))
            elif profile=='math_sft':
                msgs=[{'role':'system','content':r'Please reason step by step, and put your final answer within \boxed{}.'},{'role':'user','content':view['input']['problem']}]
                prefix=math.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True)
                full=math.apply_chat_template(msgs+[{'role':'assistant','content':view['target']}],tokenize=True,add_generation_prompt=False)
                assert full[:len(prefix)]==prefix and len(full)>len(prefix) and len(full)<=3800
                labels=[-100]*len(prefix)+full[len(prefix):];assert any(x!=-100 for x in labels)
                counts.append(len(full))
            else:
                import hashlib
                p=Path(view['input']['image_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['image_sha256']
                assert row['perception_reward_enabled'] is False
                counts.append(len(view['target']))
        summary[name]={'rows':len(rows),'profile':profile,'actual_max_prompt_or_sft_tokens':max(counts) if profile!='perception_sft' else None,
            'visual_target_max_characters':max(counts) if profile=='perception_sft' else None,'checks_passed':True}
    result={'files':summary,'no_gpu_model_loaded':True,'no_optimizer_step':True,'no_weights_downloaded':True,
        'not_a_training_smoke':'CPU verified approval scope, real strings/tokenizer prefixes, masks and budgets; full model/vision processor/GPU forward-backward/resume remain separate acceptance.'}
    dump(ROOT/'reports/data_release_preparation_20261024/approved_input_preflight.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
