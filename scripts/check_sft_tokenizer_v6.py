"""Offline full-pair encoding check; no model loading, training, or installs."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from src.data.approved_training_inputs_v2 import load_approved
from src.training.sft_encoding_v6 import encode_sft

ROOT = Path(__file__).resolve().parents[1]


def check(output_root):
    from transformers import AutoTokenizer
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'tokenizer_report.json'
    if destination.exists():
        raise ValueError('Report exists; use a new output-root')
    manifest = ROOT / 'data/portable_release/manifest.json'
    report = {
        'status': 'running', 'training_run': False, 'model_weights_loaded': False,
        'transformers': version('transformers'),
        'encoder_source_sha256': hashlib.sha256(
            (ROOT / 'src/training/sft_encoding_v6.py').read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'arms': {},
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            ROOT / 'data/tokenizers/Qwen2.5-7B-Instruct', local_files_only=True)
        report['chat_template_sha256'] = hashlib.sha256(
            tokenizer.chat_template.encode('utf-8')).hexdigest()
        for arm in ('raw', 'compact'):
            name = f'project1_{arm}_paired.jsonl'
            rows = load_approved(manifest, name)
            encoded = encode_sft(rows, tokenizer, max_length=4096)
            if len(encoded) != len(rows) or len(rows) != 1500:
                raise ValueError('Expected exactly 1500 paired training rows per arm')
            # Digest includes actual input_ids AND labels, in source order.
            digest = hashlib.sha256()
            for row in encoded:
                digest.update((json.dumps(row, separators=(',', ':')) + '\n').encode())
            stats = {
                'rows': len(encoded),
                'max_tokens': max(len(x['input_ids']) for x in encoded),
                'supervised_tokens': sum(sum(t != -100 for t in x['labels']) for x in encoded),
                'encoded_ids_labels_sha256': digest.hexdigest(),
            }
            report['arms'][arm] = stats
            print(arm, json.dumps(stats), flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                               encoding='utf-8')
    print('两臂全量编码通过；未加载模型权重、未训练。', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    check(parser.parse_args().output_root)


if __name__ == '__main__':
    main()
