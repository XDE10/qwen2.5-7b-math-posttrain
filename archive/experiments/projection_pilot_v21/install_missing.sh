#!/usr/bin/env bash
set -euo pipefail
# Only run if vLLM is absent. Preserve the user's preinstalled core framework.
python3 - <<'PY'
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
lines=[]
for name in ['torch','transformers','peft','accelerate','torchao']:
    try:lines.append(f'{name}=={version(name)}')
    except PackageNotFoundError:pass
Path('/content/projection-core-constraints.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
if ! python3 -c 'import vllm' >/dev/null 2>&1; then
  env -u PIP_CONSTRAINT python3 -m pip install --only-binary=vllm -c /content/projection-core-constraints.txt vllm
fi
