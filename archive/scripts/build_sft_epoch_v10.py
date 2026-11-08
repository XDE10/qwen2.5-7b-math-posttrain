"""Derive isolated epoch-based SFT runner/configs without modifying v6/drafts."""
import json
from pathlib import Path
from scripts.build_screening_v8 import replace_once


def build(root):
    root = Path(root)
    source = (root / 'src/training/approved_hf_v6.py').read_text(encoding='utf-8')
    source = replace_once(source, 'from src.training.guards_v6 import approved_config',
                          'from src.training.guards_v10 import approved_config\nfrom src.training.sft_epoch_v10 import training_schedule')
    source = replace_once(source, "max_steps=c['max_steps'],learning_rate=", "**training_schedule(c),learning_rate=")
    source = replace_once(source, 'save_only_model=False,seed=', 'save_total_limit=2,save_only_model=False,seed=')
    source = replace_once(source, "def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()",
                          "def sha(p):\n    from src.training.guards_v2 import sha as streamed_sha\n    return streamed_sha(p)")
    source = replace_once(source, "    if a.max_steps is not None:\n        if a.max_steps<1:raise ValueError('max-steps must be positive')\n        c['max_steps']=a.max_steps",
                          "    if a.max_steps is not None:raise ValueError('v10 stops at one epoch; do not override max-steps')")
    source = replace_once(source, "    trainer.train(resume_from_checkpoint=resume);trainer.save_model(out/'final');processor.save_pretrained(out/'final')",
                          "    trainer.train(resume_from_checkpoint=resume)\n"
                          "    completion={'global_step':trainer.state.global_step,'epoch':trainer.state.epoch,'expected_updates':94,'num_examples':len(rows),'actual_trainer_schedule':training_schedule(c)}\n"
                          "    (out/'training_completion.json').write_text(json.dumps(completion,indent=2),encoding='utf-8')\n"
                          "    if trainer.state.global_step!=94 or abs(trainer.state.epoch-1.0)>1e-6:raise ValueError('Epoch/update budget differs; inspect training_completion.json')\n"
                          "    trainer.save_model(out/'final');processor.save_pretrained(out/'final')")
    outputs = {'src/training/approved_hf_v10.py': source,
               'scripts/train_project1_sft_v10.py': "from src.training.approved_hf_v10 import cli\n\nif __name__ == '__main__':\n    cli('sft', False)\n"}
    for arm in ('raw', 'compact'):
        draft_path = f'reports/handover_20261024_v2/formal_v1/equal_exposure/E1_{arm}.draft.json'
        config = json.loads((root / draft_path).read_text(encoding='utf-8'))
        config.pop('matrix_entry')
        config.update(draft=False, status='SFT执行协议已固定；正式训练结果未验证',
                      experiment=f'E1_{arm}_equal_exposure_v10', num_train_epochs=1,
                      output_dir=f'runs/E1_{arm}_equal_exposure_v10',
                      protocol_source=draft_path,
                      budget_protocol={'name':'equal_question_exposure','stop_rule':'one complete epoch; Trainer max_steps=-1',
                                       'num_examples':1500,'num_train_epochs':1,'effective_batch':16,
                                       'expected_optimizer_updates':94,'supervised_token_budget_matched':False})
        outputs[f'configs/project1_{arm}_sft_equal_exposure_v10.json'] = json.dumps(config, ensure_ascii=False, indent=2) + '\n'
    for name, contents in outputs.items():
        if name.endswith('.py'):
            compile(contents, name, 'exec')
        path = root / name
        if path.exists() and path.read_text(encoding='utf-8') != contents:
            raise ValueError('Existing v10 differs: ' + name)
        if not path.exists():
            path.write_text(contents, encoding='utf-8', newline='\n')
    return outputs


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[1])
    print('v10 epoch SFT files ready; no model loaded or training started.')
