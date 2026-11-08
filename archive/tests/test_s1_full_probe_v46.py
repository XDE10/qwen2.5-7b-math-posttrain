"""CPU checks for the new full-SFT capacity probe, without GPU initialization."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('probe_v46', ROOT / 'tools/probe_s1_full_sft_v46.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_native_rope_accepts_transformers4_and5_config_shapes(self):
        native = {'rope_theta': 1000000.0, 'max_position_embeddings': 32768}
        probe.native_rope(native)
        params = {'rope_type': 'default', 'rope_theta': 1000000.0}
        probe.native_rope(dict(native, rope_scaling=params, rope_parameters=params))

    def test_native_rope_rejects_real_yarn_and_scaled_or_changed_theta(self):
        for params in [{'rope_type': 'yarn', 'factor': 4},
                       {'type': 'linear', 'factor': 4},
                       {'rope_type': 'default', 'factor': 4},
                       {'rope_type': 'default', 'rope_theta': 10000}]:
            for field in ['rope_scaling', 'rope_parameters']:
                with self.assertRaises(ValueError):
                    probe.native_rope({'max_position_embeddings': 32768, field: params})

    def test_known_zero_update_rope_failure_is_archived(self):
        previous = {'script_sha256': 'd1f9f6a44359796cae68fd78028b21d13af7e56da18c0a6d7cfe887942f184ae',
                    'manifest_sha256': 'same', 'model_revision': probe.BASE_REV}
        identity = dict(previous, script_sha256='new')
        for progress in [False, True]:
            with tempfile.TemporaryDirectory() as folder:
                p = Path(folder)
                probe.save(p / 'identity.json', previous)
                probe.save(p / 'PROBE_FAILED.json', {'type': 'ValueError',
                    'completed_updates': 0, 'full_training_started': False})
                (p / 'launcher.log').write_text('ValueError: Unexpected RoPE override in original base model')
                if progress:
                    probe.save(p / 'progress.json', {'completed_updates': 1})
                    with self.assertRaisesRegex(ValueError, 'identity differs'):
                        probe.prepare_probe_identity(p, identity)
                    self.assertEqual(probe.read(p / 'identity.json'), previous)
                else:
                    probe.prepare_probe_identity(p, identity)
                    self.assertEqual(probe.read(p / 'identity.json'), identity)
                    self.assertTrue((p / 'initialization_failures/rope_guard_d1f9f6a44359/PROBE_FAILED.json').exists())

    def test_deepspeed_source_is_separate_from_binary_runtime_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(probe.importlib.util, 'find_spec', return_value=None), \
                 mock.patch.object(probe.subprocess, 'run') as run, \
                 mock.patch.object(probe, 'numpy_runtime', return_value={'version': '2.2.6', 'file': '/active/numpy/__init__.py'}):
                path = Path(folder) / 'constraints.txt'
                probe.ensure_dependencies(path)
                self.assertIn('numpy==2.2.6\n', path.read_text())
                self.assertNotIn('numpy==1.26.4', path.read_text())
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertIn('--only-binary=:all:', commands[0])
        self.assertIn('--no-binary=pwinput', commands[0])
        self.assertNotIn('--no-build-isolation', commands[0])
        self.assertNotIn('deepspeed==0.19.8', commands[0])
        self.assertIn('--no-deps', commands[1])
        self.assertIn('--no-build-isolation', commands[1])
        self.assertEqual(commands[1][-1], 'deepspeed==0.19.8')
        self.assertNotIn('torch', commands[0])
        self.assertNotIn('transformers', commands[0])

    def test_pwinput_source_exception_keeps_compiled_packages_binary_only(self):
        # Check pip's real option semantics, not just the generated flag text.
        from pip._internal.models.format_control import FormatControl
        policy = FormatControl(no_binary={'pwinput'}, only_binary={':all:'})
        self.assertEqual(policy.get_allowed_formats('pwinput'), frozenset({'source'}))
        for name in ['numpy', 'scipy', 'torch', 'ninja', 'swanlab', 'hjson']:
            self.assertEqual(policy.get_allowed_formats(name), frozenset({'binary'}))

    def test_numpy126_is_not_locked_on_python313(self):
        self.assertFalse(probe.numpy_usable({'version': '1.26.4'}, (3, 13)))
        self.assertTrue(probe.numpy_usable({'version': '2.2.6'}, (3, 13)))
        self.assertTrue(probe.numpy_usable({'version': '1.26.4'}, (3, 12)))

    def test_broken_numpy_repairs_only_numpy_before_deepspeed(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(probe, 'numpy_runtime', side_effect=[None, {'version': '2.2.6'}]), \
                 mock.patch.object(probe.importlib.util, 'find_spec', return_value=object()), \
                 mock.patch.object(probe.subprocess, 'run') as run:
                probe.ensure_dependencies(Path(folder) / 'constraints.txt')
        command = run.call_args_list[0].args[0]
        self.assertIn('--no-deps', command)
        self.assertIn('--only-binary=:all:', command)
        self.assertEqual(command[-1], 'numpy==2.2.6')
        self.assertEqual(len(run.call_args_list), 2)

    def test_frozen_examples_include_worst_window_and_mask_prompt(self):
        rows, manifest = probe.load_assets(ROOT / 'tools')
        self.assertEqual(len(rows), 64)
        self.assertEqual(len(rows[0]['input_ids']), 16384)
        self.assertEqual(manifest['candidate_pool_count'], 994)
        self.assertEqual(len(manifest['excluded_exact_dev_math500_rows']), 6)
        self.assertFalse(manifest['full_training_approved_by_this_manifest'])

    def test_asset_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            for suffix in ['.json', '.jsonl.gz']:
                (p / (probe.ASSETS + suffix)).write_bytes((ROOT / 'tools' / (probe.ASSETS + suffix)).read_bytes())
            blob = p / (probe.ASSETS + '.jsonl.gz')
            blob.write_bytes(blob.read_bytes() + b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                probe.load_assets(p)

    def test_prompt_loss_leak_rejected_even_with_updated_blob_hash(self):
        rows, manifest = probe.load_assets(ROOT / 'tools')
        rows = copy.deepcopy(rows)
        rows[0]['labels'][0] = rows[0]['input_ids'][0]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            raw = ''.join(json.dumps(r) + '\n' for r in rows).encode()
            blob = gzip.compress(raw, mtime=0)
            (p / (probe.ASSETS + '.jsonl.gz')).write_bytes(blob)
            manifest = dict(manifest, probe_file_sha256=hashlib.sha256(blob).hexdigest())
            (p / (probe.ASSETS + '.json')).write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'loss mask'):
                probe.load_assets(p)

    def test_optimizer_is_cpu_offloaded_adamw(self):
        config = probe.ds_config()
        self.assertEqual(config['zero_optimization']['stage'], 2)
        self.assertEqual(config['zero_optimization']['offload_optimizer']['device'], 'cpu')
        self.assertEqual(config['optimizer']['type'], 'AdamW')
        self.assertTrue(config['bf16']['enabled'])

    def test_eta_excludes_initial_compile_update(self):
        a = probe.eta_hours([9999] + [10] * 7, [100] * 64, 6400)
        b = probe.eta_hours([1] + [10] * 7, [100] * 64, 6400)
        self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(a, 480 / 3600)


if __name__ == '__main__':
    unittest.main()
