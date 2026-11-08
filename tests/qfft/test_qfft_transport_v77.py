"""Offline URL-failure isolation and diagnostics; no live requests."""
from pathlib import Path
import ssl
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/qfft'))
import build_qfft_ds_dataset_v66 as b


def row(i):
    return {'index': i, 'sample_id': f'test/{i}', 'batch': 1, 'problem': '1+1?',
            'reference_answer': '2', 'difficulty': {'reasoning_effort': 'high'}}


def good():
    return {'choices': [{'finish_reason': 'stop', 'message': {'reasoning_content': '1+1=2', 'content': '2'}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 2}}


class TransportTests(unittest.TestCase):
    @patch.object(b, 'check_time')
    def test_go_url_error_deferred_without_retry_next_question_succeeds(self, _):
        calls=[]
        def transport(provider, *args):
            calls.append(provider)
            if len(calls)==1: raise urllib.error.URLError(ConnectionResetError('connection reset'))
            return good()
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); router=b.Router(out,'id',transport=transport,key_loader=lambda _: 'fixture-secret')
            with self.assertRaises(b.DeferredGoServerFailure) as e: router.generate(row(0))
            self.assertEqual(e.exception.kind,'url_transport')
            with self.assertRaises(RuntimeError): router.generate(row(0))
            router.generate(row(1))
            self.assertEqual(calls,['go','go'])
            self.assertFalse(router.fallback)
            self.assertEqual(router.server_policy['total_go_transient'],1)
            self.assertEqual(router.server_policy['total_go_5xx'],0)
            self.assertEqual(router.server_policy['consecutive_go_transient'],0)
            error=b.read(out/'calls/0000/go/error.json')
            self.assertEqual(error['reason_type'],'ConnectionResetError')
            self.assertEqual(error['phase'],'transport_or_response_decode')

    def test_diagnostic_redacts_secrets_and_urls(self):
        error=urllib.error.URLError('fixture-secret https://user:password@proxy.local/x Bearer abcdef')
        detail=b.safe_failure_details(error,'fixture-secret','response_save',1.2345)
        self.assertNotIn('fixture-secret',detail['reason'])
        self.assertNotIn('password',detail['reason'])
        self.assertNotIn('abcdef',detail['reason'])
        self.assertEqual(detail['phase'],'response_save')

    @patch.object(b, 'check_time')
    def test_certificate_error_stops_without_auto_deferral(self, _):
        def transport(*args): raise urllib.error.URLError(ssl.SSLCertVerificationError('certificate verify failed'))
        with tempfile.TemporaryDirectory() as d:
            router=b.Router(Path(d),'id',transport=transport,key_loader=lambda _: 'fixture')
            with self.assertRaises(RuntimeError): router.generate(row(0))
            self.assertFalse(router.deferred)

    @patch.object(b, 'check_time')
    def test_mixed_go_errors_share_three_failure_circuit(self, _):
        calls=[]
        def transport(*args):
            calls.append(1)
            if len(calls)==2: raise urllib.error.URLError('temporary DNS failure')
            raise b.Rejected(503,'unavailable')
        with tempfile.TemporaryDirectory() as d:
            router=b.Router(Path(d),'id',transport=transport,key_loader=lambda _: 'fixture')
            for i in range(3):
                with self.assertRaises(b.DeferredGoServerFailure) as e: router.generate(row(i))
                self.assertEqual(e.exception.circuit_open,i==2)
            with self.assertRaises(RuntimeError): b.Router(Path(d),'id')

    @patch.object(b, 'check_time')
    def test_official_url_failure_keeps_reservation_and_blocks_resume(self, _):
        def transport(*args): raise urllib.error.URLError('temporary connection loss')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); router=b.Router(out,'id',transport=transport,key_loader=lambda _: 'fixture')
            router.fallback=True
            with self.assertRaises(RuntimeError): router.generate(row(0))
            self.assertGreater(router.reserved,0)
            self.assertFalse(router.deferred)
            with self.assertRaises(RuntimeError): b.Router(out,'id')


if __name__ == '__main__': unittest.main()
