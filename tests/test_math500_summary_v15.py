import json
from pathlib import Path
from scripts.summarize_project1_v15 import training_metrics, render_report


def test_parse_last_training_runtime_without_executing_log_text():
    text="noise\n{'train_runtime': '1259', 'train_loss': '0.5964', 'epoch': '1'}\n"
    assert training_metrics(text)=={'train_runtime':1259.0,'train_loss':0.5964,'epoch':1.0}
    assert training_metrics('no final metrics') is None


def test_protocol_includes_e0_and_all500_with_same_greedy_budget():
    config=json.loads(Path('configs/project1_math500_protocol_v15.json').read_text(encoding='utf-8'))
    assert config['models']==['E0','raw','compact']
    assert config['count']==500 and config['temperature']==0 and config['max_new_tokens']==2048


def test_pending_test_not_presented_as_zero_accuracy():
    root=Path('reports/experiment_summary_20261025_v15')
    comparison=json.loads((root/'source_comparison.json').read_text(encoding='utf-8'))
    metrics=json.loads((root/'training_metrics.json').read_text(encoding='utf-8'))
    report=render_report(comparison,metrics,{'E0':None,'raw':None,'compact':None})
    assert report.count('未执行或结果尚未落盘')==3
    assert '2.044' in report
    assert '不能写成“compact几乎不截断”' in report
