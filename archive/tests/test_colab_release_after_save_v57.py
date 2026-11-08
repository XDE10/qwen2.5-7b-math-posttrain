"""CPU-only tests. No Colab runtime operation or network request is made."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('release57', Path(__file__).resolve().parents[1]/'tools/colab_release_after_save_v57.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ReleaseTests(unittest.TestCase):
    def fixture(self, root):
        names = ['final_model/config.json', 'final_model/tokenizer.json', 'final_model/tokenizer_config.json',
                 'final_model/chat_template.jinja', 'identity.json', 'training_manifest.json', 'train_metrics.json',
                 'final_model/model-00001-of-00001.safetensors']
        for name in names:
            p=root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'test')
        index=root/'final_model/model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'a':'model-00001-of-00001.safetensors'}, 'metadata':{'total_size':4}}))
        names.append('final_model/model.safetensors.index.json')
        receipt={'status':'success','epoch':6,'global_step':750,'files':{name:m.file_sha(root/name) for name in names}}
        (root/'FINAL_EXPORT_COMPLETE.json').write_text(json.dumps(receipt))
        return receipt

    def test_complete_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            self.assertEqual(m.verify_receipt_files(root,'FINAL_EXPORT_COMPLETE.json')['model_shards'],1)

    def test_missing_or_corrupt_shard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            (root/'final_model/model-00001-of-00001.safetensors').write_bytes(b'bad')
            with self.assertRaises(ValueError):m.verify_receipt_files(root,'FINAL_EXPORT_COMPLETE.json')

    def test_incomplete_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.fixture(root);r['global_step']=749
            (root/'FINAL_EXPORT_COMPLETE.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):m.verify_receipt_files(root,'FINAL_EXPORT_COMPLETE.json')

    def test_temporary_disk_is_not_persistent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            with self.assertRaises(ValueError):m.verify_saved_run(root,'FINAL_EXPORT_COMPLETE.json')

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):m.inside(Path(d).resolve(),'../outside')

    def test_release_uses_post_without_real_network(self):
        calls=[]
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):pass
        def fake(request,timeout):
            calls.append((request.full_url,request.method));return Response()
        m.release_request('127.0.0.1:12345',opener=fake)
        self.assertEqual(calls,[('http://127.0.0.1:12345/unassign','POST')])
        with self.assertRaises(RuntimeError):m.release_request('example.com:80',opener=fake)
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
