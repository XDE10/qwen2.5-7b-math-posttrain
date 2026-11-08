"""Create new sampler/approval versions, preserving the frozen v4 sources."""
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected v4 source; refusing an ambiguous patch: ' + old[:70])
    return text.replace(old, new, 1)


def build(root):
    root = Path(root)
    sampler = (root / 'scripts/sample_rl_candidates_v4.py').read_text(encoding='utf-8')
    sampler = replace_once(sampler,
        "    if version('vllm')!='0.11.2' or version('transformers')!='4.57.6':raise ValueError('Use the pinned separate sampler environment')",
        "    # v8 uses the documented token-prompt API; actual versions are recorded below.\n"
        "    from vllm import LLM, SamplingParams\n"
        "    if not callable(LLM.generate):raise RuntimeError('vLLM LLM.generate unavailable')")
    sampler = replace_once(sampler, "    a=p.parse_args()",
        "    p.add_argument('--generation-only',action='store_true',help='Persist GPU predictions; grade in a separate CPU process')\n    a=p.parse_args()")
    sampler = replace_once(sampler, "'max_prompt_length':1752,'max_model_len':1752+a.max_new_tokens,",
        "'max_prompt_length':max(map(len,tokens.values())),'max_model_len':max(map(len,tokens.values()))+a.max_new_tokens,")
    sampler = replace_once(sampler, "    workers=[Worker() for _ in range(4)]",
        "    workers=[] if a.generation_only else [Worker() for _ in range(4)]")
    sampler = replace_once(sampler,
        "                todo=[(r,t) for r in batch for t in range(a.num_samples) if (r['sample_id'],t) not in graded]",
        "                todo=[] if a.generation_only else [(r,t) for r in batch for t in range(a.num_samples) if (r['sample_id'],t) not in graded]")
    sampler = replace_once(sampler, "params,use_tqdm=False)", "params,use_tqdm=True)")
    sampler = replace_once(sampler,
        "        print(json.dumps({'all_shards_complete':done,'approval':'run approve_rl_pool_v4.py only after all shards complete'}))",
        "        print(json.dumps({'generation_phase_finished':True,'all_shards_complete':done,'next':'grade_rl_candidates_v8.py then approve_rl_pool_v8.py'}),flush=True)")
    approval = (root / 'scripts/approve_rl_pool_v4.py').read_text(encoding='utf-8')
    approval = replace_once(approval, "project/'scripts/sample_rl_candidates_v4.py'", "project/'scripts/sample_rl_candidates_v8.py'")
    outputs = {'scripts/sample_rl_candidates_v8.py': sampler,
               'scripts/approve_rl_pool_v8.py': approval}
    for name, source in outputs.items():
        compile(source, name, 'exec')
        path = root / name
        if path.exists() and path.read_text(encoding='utf-8') != source:
            raise ValueError('Existing v8 differs; preserve it: ' + name)
        if not path.exists():
            path.write_text(source, encoding='utf-8', newline='\n')
    return outputs


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[1])
    print('v8 sampler/approval ready; original sources preserved; GPU not invoked.')
