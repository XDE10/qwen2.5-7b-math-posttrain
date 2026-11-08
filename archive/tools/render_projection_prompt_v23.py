"""Print full ChatML for empty prefix and a synthetic 256-token continuation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.projection_pilot_v21.pilot import read,rows,save
from experiments.projection_pilot_v23.pilot import proposal_ids,render,generation_user


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='reports/projection_pilot_20261026_v23/prompt_render_check.txt')
    args = p.parse_args()
    from transformers import AutoTokenizer
    c = read(ROOT/'configs/project1_projection_pilot_v23.json')
    tok = AutoTokenizer.from_pretrained(ROOT/'data/tokenizers/Qwen2.5-7B-Instruct',local_files_only=True)
    selected = rows(ROOT/c['data_file'])
    row = min(selected,key=lambda r:r['solution_tokens'])
    # This is a synthetic prefix for wiring inspection, not a generated solution.
    prefix = tok.encode('Let us derive the result from the stated conditions. '*100,
                        add_special_tokens=False)[:256]
    empty = proposal_ids(tok,row,[])
    nonempty = proposal_ids(tok,row,prefix)
    assert nonempty == empty+prefix
    assert empty == render(tok,generation_user(row))
    assert len(prefix)==256
    text = ('Sample: '+row['sample_id']+'\n'
        'BLOCK 0: empty assistant prefix; FULL decoded token IDs\n'+tok.decode(empty,skip_special_tokens=False)
        +'\n\nBLOCK 1: SYNTHETIC 256-token assistant prefix for inspection; FULL decoded token IDs\n'
        +tok.decode(nonempty,skip_special_tokens=False)+'\n')
    destination = ROOT/args.output
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(text,encoding='utf-8')
    lengths = []
    for item in selected:
        count = len(proposal_ids(tok,item,[]))
        lengths.append(count)
        if count+c['max_new_tokens'] > c['max_model_len']:
            raise ValueError('Selected row cannot fit fixed expert + full output budget')
    save(destination.with_suffix('.json'),{'sample_id':row['sample_id'],
        'block0_prompt_tokens':len(empty),'block1_prompt_tokens':len(nonempty),
        'prefix_tokens':len(prefix),'exact_token_concat_verified':True,
        'all32_initial_prompt_min':min(lengths),'all32_initial_prompt_max':max(lengths),
        'max_initial_plus_output':max(lengths)+c['max_new_tokens'],
        'prefix_is_synthetic':True})
    print(text)


if __name__=='__main__':
    main()
