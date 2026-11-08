"""CPU-only grading after v8 GPU generation exits; no model imports/downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.training.frozen_verifier_v4 import sha
from src.training.rollout_pipeline_v4 import read, load_prepared, jsonl, indexed, append
from src.training.runner_support_v4 import PooledVerifier
from scripts.sample_rl_candidates_v4 import grade_one, complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling-root', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    root = Path(args.sampling_root).resolve()
    protocol = read(root / 'sampling_identity.json')
    source = Path(__file__).with_name('sample_rl_candidates_v8.py')
    if protocol['sampler_source_sha256'] != sha(source):
        raise ValueError('Not this v8 sampler source')
    rows, _ = load_prepared(root / 'candidate_pool.jsonl', root / 'preparation_seal.json')
    from tqdm import tqdm
    checker = PooledVerifier(args.workers)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for shard in range(protocol['num_shards']):
                subset = [r for i, r in enumerate(rows) if i % protocol['num_shards'] == shard]
                raw = indexed(jsonl(root / 'raw' / f'shard-{shard:05d}.jsonl'))
                path = root / 'graded' / f'shard-{shard:05d}.jsonl'
                graded = indexed(jsonl(path))
                expected = {(r['sample_id'], t) for r in subset for t in range(protocol['num_samples'])}
                if set(raw) != expected or not set(graded) <= expected:
                    raise ValueError('Incomplete/foreign GPU generation; resume generation first')
                todo = [(raw[(r['sample_id'], t)], r) for r in subset
                        for t in range(protocol['num_samples']) if (r['sample_id'], t) not in graded]
                def check(pair):
                    return grade_one(pair[0], pair[1], checker)
                for record in tqdm(pool.map(check, todo), total=len(todo), desc=f'CPU grade shard {shard}'):
                    append(path, record)
        if not complete(root, rows, protocol):
            raise ValueError('Sampling is incomplete')
        print('All GPU predictions graded; sampling_complete.json written.', flush=True)
    finally:
        checker.close()


if __name__ == '__main__':
    main()
