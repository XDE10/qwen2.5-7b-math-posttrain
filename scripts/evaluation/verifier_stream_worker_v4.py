"""Stream JSON requests; execute the original strict worker function unchanged."""
import json
import sys
from scripts.evaluation.math_verify_v3_strict_worker import check


def main():
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        try:result=check(request)
        except Exception as e:result={'correct':False,'status':str(e),'error_type':type(e).__name__}
        print(json.dumps({'id':request['id'],'result':result}),flush=True)


if __name__=='__main__':main()
