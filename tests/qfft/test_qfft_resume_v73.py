"""Offline regression checks: reviewed deferral never resends unknown requests."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_qfft_ds_dataset_v66 as b
import qfft_ds_common_v66 as c


def row(i):
    return {'index': i, 'sample_id': f'test/{i}', 'batch': 1, 'problem': '1+1?',
            'reference_answer': '2', 'difficulty': {'reasoning_effort': 'high'}}


def fixture(out, provider='go', identity='id'):
    p = out / 'calls/0000' / provider / 'request.json'
    c.save(p, {'identity': identity, 'sample_id': 'test/0', 'body': c.payload(row(0), provider),
               'request_sha256': c.digest(c.payload(row(0), provider))})
    item = {'index': 0, 'sample_id': 'test/0', 'provider': provider,
            'request_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    c.save(out / 'reviewed_unresolved_v73.json', {'identity': identity,
           'action': 'defer_without_retry', 'records': [item]})
    return p


class ResumeTests(unittest.TestCase):
    def test_reviewed_go_defers_but_cannot_be_resent(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); request = fixture(out); before = request.read_bytes(); calls = []
            def transport(provider, body, *_):
                calls.append(provider)
                return {'choices': [{'finish_reason': 'stop', 'message': {'reasoning_content': '1+1=2', 'content': '2'}}],
                        'usage': {'prompt_tokens': 1, 'completion_tokens': 2}}
            router = b.Router(out, 'id', transport=transport, key_loader=lambda _: 'fixture')
            self.assertEqual(set(router.deferred), {0})
            with self.assertRaises(RuntimeError): router.generate(row(0))
            with patch.object(b, 'check_time'): router.generate(row(1))
            self.assertEqual(calls, ['go'])
            self.assertEqual(request.read_bytes(), before)

    def test_changed_request_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); p=fixture(out); p.write_text(p.read_text()+'\n')
            with self.assertRaises(RuntimeError): b.Router(out,'id')

    def test_official_unknown_still_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); fixture(out,provider='official')
            with self.assertRaises(RuntimeError): b.Router(out,'id')

    def test_new_unreviewed_unknown_still_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); fixture(out)
            c.save(out/'calls/0002/go/request.json',{'identity':'id'})
            with self.assertRaises(RuntimeError): b.Router(out,'id')

    def test_wrong_identity_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); fixture(out)
            with self.assertRaises(RuntimeError): b.Router(out,'different')

    def test_late_response_requires_reconciliation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); fixture(out); c.save(out/'calls/0000/go/response.json',{'received':True})
            with self.assertRaises(RuntimeError): b.Router(out,'id')


if __name__ == '__main__': unittest.main()
