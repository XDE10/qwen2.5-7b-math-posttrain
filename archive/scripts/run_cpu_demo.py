"""Repeatable local-only synthesis/geometry smoke; no HTTP provider exists."""
import asyncio
import json
from pathlib import Path
from src.synthesis.async_client import AsyncSynthesisClient, BlindRequest, MockTransport
from src.evaluation.math_verifier import verify
from src.evaluation.geometry_perception_parser import canonicalize, perception_f1, parse_response


async def main():
    client = AsyncSynthesisClient('data/synthesis_mock/cache.sqlite',MockTransport(['#### 3','#### 2']),
                                  dry_run=False,model_id='offline-demo',requests_per_second=10000)
    request = BlindRequest('demo/train/1','Compute 1+1.','synthetic-v1')
    def verifier(text):
        result = verify(text,'2')
        return {'accepted':result['correct'],'math_verification':result}
    outcome = await client.generate(request,verifier)
    client.cache.close()
    text = '<perception>Perpendicular(DC,BA);Length(B,A)=5</perception><reasoning>Demo only.</reasoning><answer>2</answer>'
    parsed = parse_response(text)
    scores = perception_f1(parsed['canonical_perception'],'Length(A,B)=5;Perpendicular(AB,CD)',{'Length','Perpendicular'})
    result = {'provider':'local_mock','paid_api_calls':0,'gpu_operations':0,'synthesis':outcome,
              'geometry':{'canonical':parsed['canonical_perception'],'score':scores},
              'scope':'Synthetic demo, not real generated or visually verified training data'}
    Path('reports/cpu_demo.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'state':outcome['state'],'generation_attempts':len(outcome['attempts']),
                      'geometry_f1':scores['f1'],'paid_api_calls':0}))


if __name__=='__main__': asyncio.run(main())
