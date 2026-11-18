import os,sys,json,pathlib,traceback,time
ROOT=pathlib.Path('/content/p1');code=ROOT/'code/c7bcd9df7b7e'
os.environ.update(P1_LOCAL_ROOT=str(ROOT),P1_DRIVE_ROOT='/content/mydrive',P1_RCLONE_REMOTE='p1drive:',P1_CODE_SHA='c7bcd9df7b7e',P1_VALIDATION_ROOT='/content/validationdrive/runs/project1/v93/mmr_validation25',P1_VALIDATION_DRIVE_ROOT='/content/validationdrive',P1_VALIDATION_RCLONE_REMOTE='mydrive:',P1_VALIDATION_RCLONE_CONFIG='/content/validation_rclone.conf',P1_VALIDATION_BENCH_DIR='/content')
sys.path.insert(0,str(code));os.chdir(code)
from experiments.qfft_v93.common import write_json
from experiments.qfft_v93.colab import stages
stages.LOGS=pathlib.Path('/content/validationdrive/runs/project1/v93/mmr_validation25/logs')
(code/'experiments/qfft_v93/configs/grpo/mmr_validation25.json').write_bytes(pathlib.Path('/content/mmr_validation25.json').read_bytes())
# Keep the validation module external to the immutable training bundle.
import importlib.util
spec=importlib.util.spec_from_file_location('p1_validation','/content/p1_validation.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
status=ROOT/'validation_status.json';step=int(sys.argv[1]);ctx=None
write_json(status,{'state':'running','step':step,'pid':os.getpid(),'started':time.time()})
try:
 ctx=stages.Context(code,'c7bcd9df7b7e');module.evaluate(ctx,step);ctx.finish()
 write_json(status,{'state':'complete','step':step,'pid':os.getpid(),'ended':time.time()})
except BaseException as e:
 write_json(status,{'state':'failed','step':step,'pid':os.getpid(),'error':str(e),'ended':time.time()});traceback.print_exc();raise
