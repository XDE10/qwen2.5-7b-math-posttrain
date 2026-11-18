"""Replay the first two stored 16K token-mask pilot batches with both MMR reward rules on CPU."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from experiments.qfft_v93.common import load_config, read_jsonl, sha256_file, write_json
from experiments.qfft_v93.mmr_grpo import JinaEmbeddings, MmrReward


def advantages(rewards):
    # The pinned Trainer represents None as NaN, then centers each group and zeros unscorable rows.
    values = torch.tensor([torch.nan if value is None else value for value in rewards], dtype=torch.float32)
    return (values - values.nanmean()).nan_to_num(nan=0).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollouts', type=Path, required=True)
    parser.add_argument('--jina-tokenizer', type=Path, required=True)
    parser.add_argument('--jina-onnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    onnx_sha = sha256_file(args.jina_onnx)
    if onnx_sha != 'c5412c83114b16f121ddead44e2887218f16dece45efb6681dac4dd82e4f3aaa':
        raise ValueError('Expected the mean-pooled Jina ONNX from the frozen embedding revision')
    old_cfg = load_config('grpo/grpo_qfft_r1_mmr_dr50.json')
    new_cfg = load_config('grpo/grpo_qfft_r1_mmr_dr150.json')
    embed = object.__new__(JinaEmbeddings)
    embed.tokenizer = AutoTokenizer.from_pretrained(args.jina_tokenizer, local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 1
    embed.session = ort.InferenceSession(str(args.jina_onnx), sess_options=options, providers=['CPUExecutionProvider'])
    embed.input_names = [node.name for node in embed.session.get_inputs()]
    embed.max_length = old_cfg['mmr_dr_grpo']['embedding_max_length']
    vectors = {}

    def cache_embeddings(texts):
        missing = list(dict.fromkeys(text for text in texts if text not in vectors))
        if missing:
            vectors.update(zip(missing, embed(missing), strict=True))
        return np.stack([vectors[text] for text in texts])

    tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=1)
    old = MmrReward(tokenizer, old_cfg, cache_embeddings)
    new = MmrReward(tokenizer, new_cfg, cache_embeddings)
    groups = defaultdict(list)
    for row in read_jsonl(args.rollouts):
        if row['step'] in (0, 1):
            groups[(row['step'], row['sample_id'])].append(row)
    assert len(groups) == 16 and all(len(rows) == 8 for rows in groups.values())

    evidence = []
    for (step, sid), rows in sorted(groups.items()):
        chosen = sorted(rows, key=lambda row: row['index'])[:old.group]
        clipped = [row['trl_clipped'] for row in chosen]
        if not any(clipped):
            continue
        texts = [row['vllm_text'] for row in chosen]
        # The historical diagnostics retain reported length and TRL mask, but not actual IDs.
        # The reward path only reads the length and whether the final ID is EOS/PAD.
        ids = [[2] * (row['tokens'] - 1) + [2 if mask else 0]
               for row, mask in zip(chosen, clipped, strict=True)]
        inputs = dict(completions=[[{'role': 'assistant', 'content': text}] for text in texts],
                      completion_ids=ids, reference_answer=[row['reference_answer'] for row in chosen],
                      sample_id=[sid] * old.group, log_metric=lambda *_: None)
        before, after = old(**inputs), new(**inputs)
        assert [value is None for value in after] == clipped
        for row, mask, r0, r1, a0, a1 in zip(chosen, clipped, before, after, advantages(before),
                                               advantages(after), strict=True):
            evidence.append({'step': step + 1, 'sample_id': sid, 'index': row['index'], 'masked': mask,
                             'tokens': row['tokens'], 'old_adjusted_reward': r0, 'new_adjusted_reward': r1,
                             'old_advantage': a0, 'new_advantage': a1,
                             'sign_flip': not mask and a0 * a1 < 0 and min(abs(a0), abs(a1)) > 1e-6})
        print(f'Replayed historical step {step + 1}, {sid}', flush=True)

    normal = sum(not row['masked'] for row in evidence)
    flips = sum(row['sign_flip'] for row in evidence)
    write_json(args.output, {'pilot': 'grpo_qfft_r1_16k_tokenmask10_v93', 'rollouts_sha256': sha256_file(args.rollouts),
                            'embedding_revision': old_cfg['mmr_dr_grpo']['embedding_revision'], 'onnx_sha256': onnx_sha,
                            'group_selection': 'first six stored samples of each prompt at pilot steps 1 and 2',
                            'normal_in_affected_groups': normal, 'sign_flips': flips, 'rows': evidence})
    if (normal, flips) != (16, 4):
        raise RuntimeError(f'Prior result did not reproduce: {flips}/{normal} sign flips')
    print(f'MMR_REPLAY_OK sign_flips={flips}/{normal}, {args.output}')


if __name__ == '__main__':
    main()
