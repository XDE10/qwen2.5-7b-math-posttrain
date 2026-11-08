"""CPU-only checks for paid-request routing, no duplication, cleanup and review completeness."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import sqlite3
import unittest
from unittest.mock import patch
from datetime import datetime

TOOLS = Path(__file__).resolve().parents[2] / 'tools/qfft'
sys.path.insert(0, str(TOOLS))
import qfft_ds_common_v66 as c
import build_qfft_ds_dataset_v66 as b


def row(index=0):
    return {'sample_id': 'test/' + str(index), 'index': index, 'batch': index // 50 + 1,
            'problem': 'Find 1+1.', 'reference_answer': '2', 'reference_provenance': 'test fixture',
            'source': 'fixture', 'category': 'Arithmetic',
            'difficulty': c.difficulty({}, 'fixture')}


def response():
    return {'model': 'fixture', 'choices': [{'finish_reason': 'stop', 'message': {
        'reasoning_content': 'Adding one and one gives two.', 'content': r'\boxed{2}'}}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}


class NoNetworkTests(unittest.TestCase):
    def test_blind_payload_and_prompt(self):
        r = row(); r['reference_answer'] = 'SECRET_GOLD'
        p = c.payload(r, 'go')
        self.assertNotIn('SECRET_GOLD', json.dumps(p))
        self.assertEqual(len(p['messages']), 2)
        self.assertEqual(p['messages'][1]['content'], r['problem'])
        self.assertNotIn('verbosity', json.dumps(p))
        self.assertNotIn('writing strategy', c.SYSTEM)
        self.assertIn('global bound', c.SYSTEM)

    def test_teacher32k_student16k_and_both_effort_routes(self):
        for provider in c.PROVIDERS:
            for effort in ['high', 'max']:
                r = row(); r['difficulty']['reasoning_effort'] = effort
                p = c.payload(r, provider)
                self.assertEqual(p['max_tokens'], 32768)
                self.assertEqual(p['reasoning_effort'], effort)
                self.assertEqual(c.WINDOW, 16384)
                self.assertNotIn('temperature', p)

    def test_long_complete_raw_and_clean_math_are_not_truncated(self):
        class Tok:
            def encode(self, text, **kwargs):
                class Enc: ids = range(len(text))
                return Enc()
        reasoning = 'x=1. ' * 3600 + 'Desired oververbosity 5.' + 'Check x+1=2.'
        record = {'identity': 'id', 'provider': 'go', 'model_requested': 'fixture',
                  'reasoning_effort_requested': 'max', 'response': response()}
        record['response']['choices'][0]['message']['reasoning_content'] = reasoning
        with patch.object(b, 'grade', return_value={'correct': True}): result = b.candidate(row(), record, Tok())
        self.assertEqual(result['length_class'], 'complete_16k_to_32k')
        self.assertEqual(result['reasoning_content_raw'], reasoning)
        self.assertEqual(result['reasoning_content_cleaned'], reasoning.replace('Desired oververbosity 5.', ''))
        self.assertEqual(result['final_content_raw'], r'\boxed{2}')
        self.assertFalse(result['training_approved'])
        self.assertFalse(result['fits_current_training_window'])
        self.assertEqual(result['automatic_resampling_after_completion'], 0)

    def test_truncated_response_not_called_complete_even_if_reasoning_contains_answer(self):
        class Tok:
            def encode(self, text, **kwargs):
                class Enc: ids = range(len(text))
                return Enc()
        record = {'identity': 'id', 'provider': 'go', 'model_requested': 'fixture',
                  'reasoning_effort_requested': 'high', 'response': response()}
        choice = record['response']['choices'][0]
        choice['finish_reason'] = 'length'
        choice['message'] = {'reasoning_content': 'Answer could be 2. But wait...', 'content': ''}
        with patch.object(b, 'grade', return_value={'correct': False}): result = b.candidate(row(), record, Tok())
        self.assertEqual(result['length_class'], 'incomplete')
        self.assertFalse(result['structurally_complete'])
        self.assertEqual(result['final_content_raw'], '')
        self.assertIn('missing_final', result['structural_flags'])

    def test_difficulty_scales_not_conflated(self):
        self.assertEqual(c.difficulty({'level': 5}, 'qfq/openaimath/Algebra')['reasoning_effort'], 'max')
        self.assertEqual(c.difficulty({'level': 4}, 'qfq/openaimath/Algebra')['reasoning_effort'], 'high')
        self.assertEqual(c.difficulty({'difficulty': 8}, 'KbsdJames/Omni-MATH')['reasoning_effort'], 'max')
        self.assertEqual(c.difficulty({'difficulty': 5}, 'KbsdJames/Omni-MATH')['reasoning_effort'], 'high')
        self.assertEqual(c.difficulty({'level': 5}, 'unknown_scale')['reasoning_effort'], 'high')

    def test_time_gate(self):
        for hour, minute, allowed in [(11, 59, False), (12, 0, True), (13, 59, True), (14, 0, False), (18, 0, True)]:
            now = datetime(2026, 10, 29, hour, minute, tzinfo=c.TZ)
            if allowed: c.check_time(now)
            else:
                with self.assertRaises(RuntimeError): c.check_time(now)

    def test_only_explicit_quota_switches(self):
        self.assertTrue(c.quota_error(429, 'Go monthly usage limit reached'))
        self.assertTrue(c.quota_error(402, 'quota_exceeded'))
        self.assertFalse(c.quota_error(429, 'Too many requests per second'))
        self.assertFalse(c.quota_error(401, 'quota invalid key'))
        self.assertFalse(c.quota_error(400, 'Global regions required'))

    def test_cleanup_exact_and_reversible_evidence(self):
        text = 'Let desired = 5.\nDesired oververbosity 5.\nx^2=25.\nSince desired oververbosity 5, step-by-step.'
        cleaned, edits = c.clean_meta(text)
        self.assertIn('Let desired = 5.', cleaned)
        self.assertIn('x^2=25.', cleaned)
        self.assertEqual(len(edits), 2)
        for edit in edits: self.assertEqual(text[edit['start']:edit['end']], edit['removed'])

    @patch.object(b, 'check_time')
    def test_go_quota_fallback_is_sticky_and_resume_reuses(self, _):
        calls = []
        def transport(provider, *args):
            calls.append(provider)
            if provider == 'go': raise b.Rejected(429, 'Go weekly usage limit reached')
            return response()
        with tempfile.TemporaryDirectory() as d:
            router = b.Router(Path(d), 'id', transport=transport, key_loader=lambda p: 'not-a-real-key')
            router.generate(row(0)); router.generate(row(1))
            self.assertEqual(calls, ['go', 'official', 'official'])
            restored = b.Router(Path(d), 'id', transport=transport, key_loader=lambda p: 'not-a-real-key')
            restored.generate(row(0))
            self.assertEqual(len(calls), 3)
            self.assertTrue(restored.fallback)
            self.assertGreater(restored.spent, 0)

    @patch.object(b, 'check_time')
    def test_nonquota_does_not_spend_official(self, _):
        calls = []
        def transport(provider, *args):
            calls.append(provider); raise b.Rejected(429, 'Too many requests per second')
        with tempfile.TemporaryDirectory() as d:
            router = b.Router(Path(d), 'id', transport=transport, key_loader=lambda p: 'not-a-real-key')
            with self.assertRaises(RuntimeError): router.generate(row())
            self.assertEqual(calls, ['go'])
            self.assertFalse(router.fallback)

    @patch.object(b, 'check_time')
    def test_transport_failure_is_never_resent(self, _):
        calls = []
        def transport(provider, *args):
            calls.append(provider); raise TimeoutError()
        with tempfile.TemporaryDirectory() as d:
            router = b.Router(Path(d), 'id', transport=transport, key_loader=lambda p: 'not-a-real-key')
            with self.assertRaises(RuntimeError): router.generate(row())
            router.fallback = True
            with self.assertRaises(RuntimeError): router.generate(row())
            self.assertEqual(calls, ['go'])

    @patch.object(b, 'check_time')
    def test_received_go_response_reused_after_route_change(self, _):
        calls = []
        def transport(provider, *args): calls.append(provider); return response()
        with tempfile.TemporaryDirectory() as d:
            router = b.Router(Path(d), 'id', transport=transport, key_loader=lambda p: 'not-a-real-key')
            router.generate(row()); router.fallback = True
            reused = router.generate(row())
            self.assertEqual(calls, ['go'])
            self.assertEqual(reused['provider'], 'go')

    def test_budget_reserves_inflight(self):
        with tempfile.TemporaryDirectory() as d:
            router = b.Router(Path(d), 'id', official_cap=.030)
            router.reserve(c.payload(row(), 'official'))
            with self.assertRaises(RuntimeError): router.reserve(c.payload(row(1), 'official'))

    def test_budget_reservation_covers_full_32k_output(self):
        with tempfile.TemporaryDirectory() as d:
            router = b.Router(Path(d), 'id', official_cap=5)
            body = c.payload(row(), 'official')
            reserved = router.reserve(body)
            self.assertGreaterEqual(reserved, 32768 * .6 / 1e6)

    def test_go_saved_credential_readonly_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'opencode.db'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE credential (integration_id TEXT, active INTEGER, value TEXT)')
                db.execute('INSERT INTO credential VALUES (?,?,?)',
                           ('opencode-go', 1, json.dumps({'type': 'key', 'key': 'fixture-not-a-real-key'})))
            db.close()
            before = c.sha_file(path)
            with patch.object(c, 'GO_DB', path):
                self.assertEqual(c.key_for('go'), 'fixture-not-a-real-key')
            self.assertEqual(c.sha_file(path), before)

    def test_unknown_dispatch_blocks_new_spending_after_restart(self):
        with tempfile.TemporaryDirectory() as d:
            c.save(Path(d) / 'calls/0000/official/request.json', {'identity': 'id'})
            with self.assertRaises(RuntimeError): b.Router(Path(d), 'id')

    def test_os_lock_blocks_duplicate_run(self):
        with tempfile.TemporaryDirectory() as d:
            with b.run_lock(Path(d)):
                with self.assertRaises(RuntimeError):
                    with b.run_lock(Path(d)): pass
            with b.run_lock(Path(d)): pass

    def test_export_keeps_all_full_trajectories_and_all_ids(self):
        class Tok:
            def encode(self, text, **kwargs):
                class Enc: ids = range(len(text))
                return Enc()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'out'; report = Path(d) / 'report'; rr = [row(i) for i in range(50)]
            for r in rr:
                c.save(out / 'candidates' / f'{r["index"]:04d}.json', {
                    'provider': 'go', 'model_requested': 'fixture', 'model_returned': 'fixture',
                    'finish_reason': 'stop', 'reasoning_content_raw': '完整推导' * 100,
                    'final_content_raw': r'\boxed{2}', 'cleaning_edits': [],
                    'sample_id': r['sample_id'], 'index': r['index'],
                    'raw_student_tokens': 100, 'cleaned_student_tokens': 100,
                    'length_class': 'complete_within_16k', 'structurally_complete': True,
                    'fits_current_training_window': True})
            self.assertEqual(b.export_review(rr, out, report, Tok()), 50)
            path = out / 'gemini_batches/batch_01'
            master = c.read(path / '完整50题审查数据.json')
            self.assertEqual(len(master['records']), 50)
            parts = [c.read(p) for p in sorted(path.glob('part_*.json'))]
            self.assertEqual(len(parts), 5)
            allrows = [r for p in parts for r in p['records']]
            self.assertEqual([r['sample_id'] for r in allrows], [r['sample_id'] for r in rr])
            self.assertTrue(all(r['reasoning_content_raw'] == '完整推导' * 100 for r in allrows))


if __name__ == '__main__': unittest.main()
