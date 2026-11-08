"""Derive online SFT version while retaining v10 and its training budget."""
import json
from pathlib import Path
from scripts.build_screening_v8 import replace_once


def build(root):
    root = Path(root)
    source = (root / 'src/training/approved_hf_v10.py').read_text(encoding='utf-8')
    source = replace_once(source, 'from src.training.guards_v10 import approved_config',
                          'from src.training.guards_v11 import approved_config')
    source = replace_once(source, 'from src.training.runner_support_v4 import PooledVerifier,swanlab_callback',
                          'from src.training.runner_support_v4 import PooledVerifier\nfrom src.training.swanlab_online_v11 import swanlab_callback')
    source = replace_once(source, "def run(config_path,swanlab_mode='offline'):", "def run(config_path,swanlab_mode='online'):")
    source = replace_once(source, "default='offline')", "default='online')")
    outputs = {'src/training/approved_hf_v11.py': source,
               'scripts/train_project1_sft_v11.py': "from src.training.approved_hf_v11 import cli\n\nif __name__ == '__main__':\n    cli('sft', False)\n"}
    for arm in ('raw', 'compact'):
        config = json.loads((root / f'configs/project1_{arm}_sft_equal_exposure_v10.json').read_text(encoding='utf-8'))
        config['experiment'] = f'E1_{arm}_equal_exposure_v11_online'
        config['output_dir'] = 'runs/' + config['experiment']
        config['logging_protocol'] = 'SwanLab online; full config and actual TrainingArguments; key from process environment only'
        outputs[f'configs/project1_{arm}_sft_equal_exposure_v11.json'] = json.dumps(config, ensure_ascii=False, indent=2) + '\n'
    for name, contents in outputs.items():
        path = root / name
        if name.endswith('.py'):
            compile(contents, name, 'exec')
        if path.exists() and path.read_text(encoding='utf-8') != contents:
            raise ValueError('Existing v11 differs: ' + name)
        if not path.exists():
            path.write_text(contents, encoding='utf-8', newline='\n')
    return outputs


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[1])
