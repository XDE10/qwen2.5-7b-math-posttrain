"""Small versioned derivative; preserve every previously executed runner."""
from pathlib import Path
from scripts.build_screening_v8 import replace_once


def build(root):
    root=Path(root)
    source=(root/'src/training/approved_hf_v11.py').read_text(encoding='utf-8')
    source=replace_once(source,'from src.training.guards_v11 import approved_config','from src.training.guards_v17 import approved_config')
    source=replace_once(source,'from src.training.sft_epoch_v10 import training_schedule','from src.training.sft_schedule_v17 import training_schedule')
    source=replace_once(source,"lora_dropout=0,target_modules=","lora_dropout=c['lora_dropout'],target_modules=")
    source=replace_once(source,'save_total_limit=2,save_only_model=False',"save_total_limit=c['save_total_limit'],save_only_model=False")
    source=replace_once(source,"'expected_updates':94,","'expected_updates':c['expected_optimizer_updates'],")
    source=replace_once(source,"trainer.state.global_step!=94 or abs(trainer.state.epoch-1.0)>1e-6",
                        "trainer.state.global_step!=c['expected_optimizer_updates'] or abs(trainer.state.epoch-c['num_train_epochs'])>1e-6")
    source=replace_once(source,"raise ValueError('v10 stops at one epoch; do not override max-steps')",
                        "raise ValueError('v17 uses the configured epoch budget; do not override max-steps')")
    source=replace_once(source,'    trainer.train(resume_from_checkpoint=resume)',
                        "    from src.training.swanlab_online_v11 import safe_config\n"
                        "    (out/'actual_training_arguments.json').write_text(json.dumps(safe_config(trainer.args.to_dict()),ensure_ascii=False,indent=2),encoding='utf-8')\n"
                        "    print(json.dumps({'stage':'training','arm':c['arm'],'epochs':c['num_train_epochs'],'expected_updates':c['expected_optimizer_updates'],'learning_rate':c['learning_rate'],'schedule':training_schedule(c),'effective_batch':c['batch_size']*c['gradient_accumulation_steps']},ensure_ascii=False),flush=True)\n"
                        "    trainer.train(resume_from_checkpoint=resume)")
    outputs={'src/training/approved_hf_v17.py':source,
             'scripts/train_project1_sft_v17.py':"from src.training.approved_hf_v17 import cli\n\nif __name__=='__main__':\n    cli('sft',False)\n"}
    for name,contents in outputs.items():
        compile(contents,name,'exec');path=root/name
        if path.exists() and path.read_text(encoding='utf-8')!=contents:raise ValueError('Existing v17 differs: '+name)
        if not path.exists():path.write_text(contents,encoding='utf-8',newline='\n')
    return outputs


if __name__=='__main__':build(Path(__file__).resolve().parents[1])
