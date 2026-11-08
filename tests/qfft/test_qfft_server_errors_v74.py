"""Offline behavior tests: isolated Go 5xx never retries or spends official funds."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/qfft'))
import build_qfft_ds_dataset_v66 as b
import qfft_ds_common_v66 as c


def row(i):
    return {'index': i, 'sample_id': f'test/{i}', 'batch': 1, 'problem': '1+1?',
            'reference_answer': '2', 'difficulty': {'reasoning_effort': 'high'}}


def good():
    return {'choices': [{'finish_reason': 'stop', 'message': {'reasoning_content': '1+1=2', 'content': '2'}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 2}}


class ServerErrors(unittest.TestCase):
    @patch.object(b, 'check_time')
    def test_one_go_500_deferred_next_question_succeeds_no_retry(self, _):
        calls=[]
        def transport(provider, body, *args):
            calls.append(provider)
            if len(calls)==1: raise b.Rejected(500,'Unknown Error')
            return good()
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); router=b.Router(out,'id',transport=transport,key_loader=lambda _: 'fixture')
            with self.assertRaises(b.DeferredGoServerFailure) as caught: router.generate(row(0))
            self.assertFalse(caught.exception.circuit_open)
            with self.assertRaises(RuntimeError): router.generate(row(0))
            router.generate(row(1))
            self.assertEqual(calls,['go','go'])
            self.assertEqual(router.spent,0)
            self.assertFalse(router.fallback)
            restored=b.Router(out,'id',transport=transport,key_loader=lambda _: 'fixture')
            self.assertEqual(set(restored.deferred),{0})
            self.assertEqual(restored.server_policy['consecutive_go_5xx'],0)
            self.assertEqual(restored.server_policy['total_go_5xx'],1)

    @patch.object(b, 'check_time')
    def test_three_consecutive_go_5xx_opens_circuit(self, _):
        def transport(*args): raise b.Rejected(503,'Unavailable')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); router=b.Router(out,'id',transport=transport,key_loader=lambda _: 'fixture')
            for i in range(3):
                with self.assertRaises(b.DeferredGoServerFailure) as caught: router.generate(row(i))
                self.assertEqual(caught.exception.circuit_open,i==2)
            with self.assertRaises(RuntimeError): b.Router(out,'id')

    @patch.object(b, 'check_time')
    def test_official_500_keeps_reservation_blocks_restart(self, _):
        def transport(*args): raise b.Rejected(502,'Gateway error')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); router=b.Router(out,'id',transport=transport,key_loader=lambda _: 'fixture')
            router.fallback=True
            with self.assertRaises(RuntimeError): router.generate(row(0))
            self.assertGreater(router.reserved,0)
            self.assertFalse(router.deferred)
            with self.assertRaises(RuntimeError): b.Router(out,'id')

    def test_persisted_total_circuit_not_reset_by_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            c.save(out/'server_error_policy_v74.json',{'identity':'id','consecutive_go_5xx':0,'total_go_5xx':10})
            with self.assertRaises(RuntimeError): b.Router(out,'id')


if __name__ == '__main__': unittest.main()
