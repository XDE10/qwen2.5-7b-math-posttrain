from pathlib import Path

import pytest

from scripts.build_screening_v8 import build, replace_once


def test_builder_preserves_old_sources_and_is_repeatable(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / 'scripts').mkdir()
    for name in ('sample_rl_candidates_v4.py', 'approve_rl_pool_v4.py'):
        (tmp_path / 'scripts' / name).write_bytes((root / 'scripts' / name).read_bytes())
    before = {p.name: p.read_bytes() for p in (tmp_path / 'scripts').iterdir()}
    output = build(tmp_path)
    assert output == build(tmp_path)
    for name, data in before.items():
        assert (tmp_path / 'scripts' / name).read_bytes() == data
    sampler = output['scripts/sample_rl_candidates_v8.py']
    assert "version('vllm')!='0.11.2'" not in sampler
    assert "'sampler_source_sha256':sha(__file__)" in sampler
    assert "'versions':{n:version(n)" in sampler
    assert 'use_tqdm=True' in sampler
    assert 'max(map(len,tokens.values()))+a.max_new_tokens' in sampler
    assert 'a.generation_only else' in sampler
    assert 'scripts/sample_rl_candidates_v8.py' in output['scripts/approve_rl_pool_v8.py']
    (tmp_path / 'scripts/sample_rl_candidates_v8.py').write_text('changed')
    with pytest.raises(ValueError, match='Existing v8 differs'):
        build(tmp_path)


@pytest.mark.parametrize('text', ['missing', 'a a'])
def test_unexpected_source_rejected(text):
    with pytest.raises(ValueError):
        replace_once(text, 'a', 'b')
