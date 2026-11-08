import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
loader = importlib.machinery.SourceFileLoader('v86', str(TOOLS / 'retry_length_max_v86.pyw'))
spec = importlib.util.spec_from_loader(loader.name, loader)
m = importlib.util.module_from_spec(spec); loader.exec_module(m)


class Tests(unittest.TestCase):
    def test_selection_only_length_and_original_max(self):
        selected, skipped = m.select([{'index': 1, 'finish_reason': 'length', 'reasoning_effort_requested': 'max'},
                                      {'index': 2, 'finish_reason': 'length', 'reasoning_effort_requested': 'high'},
                                      {'index': 3, 'finish_reason': 'stop', 'reasoning_effort_requested': 'max'}])
        self.assertEqual(selected, [1]); self.assertEqual(skipped, [2])

    def test_max_success_no_high(self):
        calls = []
        def attempt(i, e):
            calls.append(e); return {'success_structural': True, 'candidate_path': 'fixture'}
        self.assertEqual(m.cascade(1, attempt)['selected_effort'], 'max')
        self.assertEqual(calls, ['max'])

    def test_max_failure_high_success_exactly_two(self):
        calls = []
        def attempt(i, e):
            calls.append(e); return {'success_structural': e == 'high', 'candidate_path': 'fixture'}
        self.assertEqual(m.cascade(1, attempt)['selected_effort'], 'high')
        self.assertEqual(calls, ['max', 'high'])

    def test_both_fail_no_third_attempt(self):
        calls = []
        def attempt(i, e):
            calls.append(e); return {'success_structural': False}
        self.assertEqual(m.cascade(1, attempt)['outcome'], 'both_failed_skip')
        self.assertEqual(calls, ['max', 'high'])

    def test_global_error_never_falls_back(self):
        calls = []
        def attempt(i, e):
            calls.append(e); raise RuntimeError('circuit or auth')
        with self.assertRaises(RuntimeError): m.cascade(1, attempt)
        self.assertEqual(calls, ['max'])


if __name__ == '__main__': unittest.main()
