"""CPU + DeepSeek review of synced final-model outputs on this Windows PC.

No torch/vLLM import, no model weights, and no GPU generation. Sync raw outputs
and generation identity from Drive into the local snapshot before running.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['dev', 'math500', 'hard637'], default='dev')
    parser.add_argument('--snapshot-root', type=Path, default=ROOT / 'artifacts/final_eval_local_snapshot')
    parser.add_argument('--credentials', type=Path,
                        default=Path.home() / '.codex/private/project1/deepseek.json')
    args = parser.parse_args()
    tool = ROOT / 'tools/continue_final_eval_v33.py'
    spec = importlib.util.spec_from_file_location('review_continuation_local', tool)
    continuation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(continuation)
    stage = 'review-dev' if args.benchmark == 'dev' else args.benchmark
    suffix = 'final_dev256_32k_v32' if args.benchmark == 'dev' else f'final_{args.benchmark}_32k_v33'
    directory = args.snapshot_root / 'evaluations' / suffix
    raw_path = directory / ('predictions.jsonl' if args.benchmark == 'dev' else 'raw.jsonl')
    if not raw_path.exists() or not (directory / 'identity.json').exists():
        raise SystemExit('Local snapshot not ready: sync completed raw outputs and identity.json from Drive first. No GPU/API task started.')
    data, settings = continuation.sources(stage, ROOT, args.snapshot_root.parent)
    raw = continuation.base.journal(raw_path)
    if args.benchmark == 'dev':
        raw = [{'sample_id': r['sample_id'], 'draw': 0, 'text': r['text'],
                'output_tokens': r['generated_tokens'], 'finish_reason': r['finish_reason']} for r in raw]
    if continuation.inspect_raw(raw, data, settings['samples']):
        raise SystemExit('Local snapshot is incomplete. Sync all responses first; no partial full score or GPU task started.')
    api_key = continuation.judge.credential(args.credentials)
    with continuation.judge.locked(directory / 'local_review_v34.lock'):
        continuation.review(raw, data, settings, directory, ROOT, api_key)
    print('LOCAL_REVIEW_COMPLETE: all scoring ran on local CPU and DeepSeek API; no GPU model loaded.', flush=True)


if __name__ == '__main__':
    main()
