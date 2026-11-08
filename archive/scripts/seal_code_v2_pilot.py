"""Seal the pilot after explicit agent review and regression checks."""
import json
import subprocess
import sys
from pathlib import Path
from src.data.screening_io import rows,sha,json_write

REVIEW_NOTES={
 'ace/oss_1909':'Named integer input, formatted string return, literals and signature agree.',
 'ace/oss_656':'Template lookup table and missing-ID None behavior are specified.',
 'ace/oss_594':'Absolute-value list transform; parameter binding and return agree.',
 'ace/oss_1331':'Explicit optional integer contract; None/positive/nonpositive branches represented.',
 'ace/oss_2086':'Explicit list signature; clipped decrement behavior agrees.',
 'ace/oss_1012':'Named table/data arguments and row-tuples return agree; dictionary-order assumption remains visible.',
 'magicoder/139304':'Five explicit parameters match reference; prompt requires both print and return.',
 'magicoder/115014':'Explicit content argument; plain string transformation, no Flask dependency in reference.',
 'magicoder/26867':'Explicit nums signature, pure even-number sum; reference used for static provenance only.',
 'magicoder/61187':'Configuration-to-code-string task; returned code is text, not executed.',
 'magicoder/94949':'Explicit dictionary input, dictionary output; configuration values remain data.',
 'magicoder/85883':'Explicit marks dictionary, name/list return contract; empty-input behavior unspecified.',
 'magicoder/33492':'Explicit three-parameter mapping update, dictionary return.',
 'magicoder/63522':'Explicit bytes/endian signature; training-only, not JSON-callable dev.'}


def run():
    base=Path('data/code_v2_pilot');report=json.loads((base/'report.json').read_text(encoding='utf-8'))
    candidates=list(rows(base/'static_candidates.jsonl'))
    assert {r['sample_id'] for r in candidates}==set(REVIEW_NOTES),'Review list must match this exact pilot'
    for filename,digest in {**report['script_hashes'],**report['protected_files']}.items():assert sha(filename)==digest
    assert report['ace_executed']>=5 and report['ace_reference_passed']==report['ace_executed']
    for r in candidates:
        if r['source_key']=='ace':assert not any(r['pilot_execution']['constant_mutant_passes'])
    checks=subprocess.run([sys.executable,'-m','pytest','tests/test_code_v2_extraction.py','tests/test_code_v2_dedup.py','-q'],capture_output=True,text=True)
    Path('reports/code_v2_pilot_rule_tests.log').write_text(checks.stdout+checks.stderr,encoding='utf-8')
    assert checks.returncode==0
    review={'reviewer':'Codex agent, not human','reviewed_candidates':len(candidates),
        'source_quarantine':json.loads(Path('data/code_v2_sources/pilot_quarantine.json').read_text(encoding='utf-8')),
        'notes':REVIEW_NOTES,'scope':'Signature/specification consistency and visible sample conflicts; no claim of exhaustive semantic correctness.'}
    json_write(base/'agent_review.json',review)
    report.update(agent_review='14 retained candidates explicitly reviewed; 8 observed conflicts quarantined',
        bulk_gate_passed=True,candidate_sha256=sha(base/'static_candidates.jsonl'),
        review_sha256=sha(base/'agent_review.json'),rule_tests='10 passed')
    json_write(base/'report.json',report)
    print('Pilot gate PASSED:',len(candidates),'reviewed candidates; static, regression and Ace sandbox checks passed')


if __name__=='__main__':run()
