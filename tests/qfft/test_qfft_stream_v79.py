"""Offline SSE reconstruction, partial preservation and legacy cache compatibility."""
import http.client
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/qfft'))
import build_qfft_ds_dataset_v66 as b
import qfft_ds_common_v66 as c
import qfft_stream_transport_v79 as stream


def event(delta=None, finish=None, usage=None):
    obj={'id':'fixture','model':'fixture-model','choices':[] if usage else
         [{'index':0,'delta':delta or {},'finish_reason':finish}]}
    if usage:obj['usage']=usage
    return [('data: '+json.dumps(obj,ensure_ascii=False)+'\n').encode(),b'\n']


def row():
    return {'index':0,'sample_id':'fixture/0','batch':1,'problem':'1+1?',
            'difficulty':{'reasoning_effort':'high'}}


class StreamTests(unittest.TestCase):
    def test_complete_reasoning_content_usage_and_raw_chunks(self):
        lines=event({'role':'assistant','reasoning_content':'推导'})+event({'reasoning_content':'完整'})
        lines+=event({'content':r'\boxed{2}'})+event(finish='stop')
        lines+=event(usage={'prompt_tokens':12,'completion_tokens':34})+[b'data: [DONE]\n',b'\n']
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);r=stream.read_sse(iter(lines),out)
            self.assertEqual(r['choices'][0]['message']['reasoning_content'],'推导完整')
            self.assertEqual(r['choices'][0]['message']['content'],r'\boxed{2}')
            self.assertEqual(r['usage']['completion_tokens'],34)
            self.assertTrue(r['stream_complete'])
            self.assertEqual(len((out/'stream_chunks.jsonl').read_text(encoding='utf-8').splitlines()),5)
            self.assertEqual(c.read(out/'stream_partial.json')['state'],'received_complete')

    def test_disconnect_keeps_received_reasoning_without_fake_final(self):
        def broken():
            yield from event({'reasoning_content':'already received reasoning'})
            raise http.client.RemoteDisconnected('Remote end closed connection without response')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            with self.assertRaises(http.client.RemoteDisconnected):stream.read_sse(broken(),out)
            partial=c.read(out/'stream_partial.json')
            self.assertEqual(partial['state'],'interrupted')
            self.assertEqual(partial['response']['choices'][0]['message']['reasoning_content'],'already received reasoning')
            self.assertEqual(partial['response']['choices'][0]['message']['content'],'')
            self.assertFalse((out/'stream_complete.json').exists())

    def test_missing_done_is_incomplete_even_with_final(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(stream.SSETransportError):
                stream.read_sse(iter(event({'content':'2'},finish='stop')),Path(d))
            self.assertFalse(c.read(Path(d)/'stream_partial.json')['response']['stream_complete'])

    @patch.object(b,'check_time')
    def test_legacy_nonstream_response_replays_without_network(self,_):
        calls=[]
        def transport(provider,body,*args):
            calls.append(body)
            return {'choices':[{'finish_reason':'stop','message':{'reasoning_content':'1+1=2','content':'2'}}],
                    'usage':{'prompt_tokens':1,'completion_tokens':2}}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); router=b.Router(out,'id',transport=transport,key_loader=lambda _:'fixture')
            router.generate(row())
            self.assertTrue(calls[0]['stream'])
            req=out/'calls/0000/go/request.json';resp=req.with_name('response.json')
            old=c.read(req);old['body']['stream']=False;old['body'].pop('stream_options')
            old['request_sha256']=c.digest(old['body']);c.save(req,old)
            rec=c.read(resp);rec['request_sha256']=old['request_sha256'];c.save(resp,rec)
            restored=b.Router(out,'id',transport=transport,key_loader=lambda _:'fixture')
            restored.generate(row())
            self.assertEqual(len(calls),1)

    def test_direct_disconnect_and_incomplete_read_classified(self):
        self.assertTrue(b.transient_go_url_error(http.client.RemoteDisconnected('closed')))
        self.assertTrue(b.transient_go_url_error(http.client.IncompleteRead(b'partial')))
        self.assertTrue(b.transient_go_url_error(stream.SSETransportError('lost DONE')))

    def test_completed_journal_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'stream_chunks.jsonl').write_text('old evidence')
            with self.assertRaises(FileExistsError):stream.read_sse(iter([]),p)
            self.assertEqual((p/'stream_chunks.jsonl').read_text(),'old evidence')


if __name__=='__main__':unittest.main()
