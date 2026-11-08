"""Exercise versioned smoke configs through the real v6/v4/v2 guards on CPU."""
import json
from pathlib import Path

import pytest

from src.training.guards_v6 import approved_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arm', ['raw', 'compact'])
def test_smoke_config_reaches_e0_guard_and_binds_arm(tmp_path, monkeypatch, arm):
    monkeypatch.chdir(ROOT)
    from src.training import guards_v4
    calls = []
    # No model weights are present locally. Only the E0 weight check is mocked;
    # real approved rows, pairing, dev protection, hashes and identities run.
    monkeypatch.setattr(guards_v4, 'validate_model', lambda m, i: calls.append((m, i)))
    config = json.loads((ROOT / f'configs/project1_{arm}_sft_smoke_v7.json').read_text())
    config['model'] = str(tmp_path / 'model')
    config['e0_model_identity_path'] = str(tmp_path / 'e0_model_identity.json')
    path = tmp_path / 'resolved.json'
    path.write_text(json.dumps(config))
    actual, rows, identity = approved_config(path)
    assert len(rows) == 1500
    assert actual['arm'] == arm
    assert actual['max_steps'] == 2
    assert actual['experiment'] == f'project1_{arm}_sft_smoke_v7'
    assert identity['config']['arm'] == arm
    assert calls == [(config['model'], config['e0_model_identity_path'])]
    assert 'sft_encoder_v6_sha256' in identity


@pytest.mark.parametrize('arm', ['raw', 'compact'])
def test_smoke_config_wrong_arm_still_rejected(tmp_path, monkeypatch, arm):
    monkeypatch.chdir(ROOT)
    config = json.loads((ROOT / f'configs/project1_{arm}_sft_smoke_v7.json').read_text())
    config['arm'] = 'compact' if arm == 'raw' else 'raw'
    path = tmp_path / 'wrong.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='differs from declared arm'):
        approved_config(path)
