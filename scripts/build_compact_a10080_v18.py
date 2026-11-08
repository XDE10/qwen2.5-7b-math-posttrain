"""Derive a large-microbatch variant without changing the effective batch."""
import json
from pathlib import Path
from scripts.build_screening_v8 import replace_once


def build(root):
    root=Path(root);source=(root/'src/training/approved_hf_v17.py').read_text(encoding='utf-8')
    source=replace_once(source,'from src.training.guards_v17 import approved_config','from src.training.guards_v18 import approved_config')
    source=replace_once(source,'from src.training.swanlab_online_v11 import swanlab_callback',
                        'from src.training.swanlab_training_v18 import swanlab_callback')
    source=replace_once(source,"    if torch.cuda.device_count()!=1:raise RuntimeError('Initial configuration supports exactly one visible GPU')",
                        "    if torch.cuda.device_count()!=1:raise RuntimeError('Initial configuration supports exactly one visible GPU')\n"
                        "    if torch.cuda.get_device_properties(0).total_memory/(1024**3)<c['minimum_gpu_memory_gib']:raise RuntimeError('v18 microbatch16 requires the requested 80GB-class GPU; use an appropriate versioned configuration for smaller cards')")
    source=replace_once(source,'    import atexit',
                        "    if swan_metrics.swan:swan_metrics.swan.config.update({'run_identity':identity})\n    import atexit")
    source=replace_once(source,"        else:data=Dataset.from_list(encode_sft(rows,tok,c['max_length']));collator=CompletionCollator(tok.pad_token_id)",
                        "        else:\n"
                        "            encoded=encode_sft(rows,tok,c['max_length'])\n"
                        "            token_stats={'rows':len(encoded),'max_input_tokens':max(len(r['input_ids']) for r in encoded),'input_tokens_per_epoch':sum(len(r['input_ids']) for r in encoded),'supervised_tokens_per_epoch':sum(sum(t!=-100 for t in r['labels']) for r in encoded)}\n"
                        "            (out/'dataset_token_statistics.json').write_text(json.dumps(token_stats,indent=2),encoding='utf-8')\n"
                        "            if swan_metrics.swan:swan_metrics.swan.config.update({'encoded_data_statistics':token_stats})\n"
                        "            data=Dataset.from_list(encoded);collator=CompletionCollator(tok.pad_token_id)")
    source=source.replace('Version6 SFT-only compatibility derivative of v4; GPU acceptance unverified.',
                          'Version18 compact-only A10080 SFT with explicit cosine and live telemetry; GPU unverified.')
    config=json.loads((root/'configs/project1_compact_cosine_v17.json').read_text(encoding='utf-8'))
    config.update(experiment='E1_compact_r32_lr1e4_e3_b32_cosine_A10080_v18',
                  output_dir='runs/E1_compact_r32_lr1e4_e3_b32_cosine_A10080_v18',
                  batch_size=16,gradient_accumulation_steps=2,minimum_gpu_memory_gib=70,
                  hardware_policy='A10080 target; measure real peaks, no artificial memory reservation; microbatch16 not yet GPU-verified',
                  logging_policy='SwanLab online for training, full parameters and CUDA memory curves; standalone future evaluation defaults to local files')
    outputs={'src/training/approved_hf_v18.py':source,
             'configs/project1_compact_cosine_A10080_v18.json':json.dumps(config,ensure_ascii=False,indent=2)+'\n',
             'scripts/train_project1_sft_v18.py':"from src.training.approved_hf_v18 import cli\n\nif __name__=='__main__':\n    cli('sft',False)\n"}
    for name,contents in outputs.items():
        if name.endswith('.py'):compile(contents,name,'exec')
        path=root/name
        if path.exists() and path.read_text(encoding='utf-8')!=contents:raise ValueError('Existing v18 differs: '+name)
        if not path.exists():path.write_text(contents,encoding='utf-8',newline='\n')
    return outputs


if __name__=='__main__':build(Path(__file__).resolve().parents[1])
