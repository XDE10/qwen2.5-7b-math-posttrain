#!/usr/bin/env bash
# No model download or training. Run from the copied project root.
set -euo pipefail
: "${PROJECT_DRIVE_ROOT:?Set PROJECT_DRIVE_ROOT after mounting Google Drive in a notebook Python cell}"
if [[ ! -d /content/drive/MyDrive ]]; then
  printf '%s\n' 'Drive is not mounted. First run a notebook Python cell:' 'from google.colab import drive' "drive.mount('/content/drive')"
  exit 2
fi
python -c 'import sys; assert sys.version_info[:2] == (3,12), "Select Python 3.12; do not silently change the lock"'
venv_dir="${PROJECT_VENV_DIR:-/content/project1-env}"
if [[ ! -x "${venv_dir}/bin/python" ]]; then
  python -m venv "${venv_dir}"
fi
venv_python="${venv_dir}/bin/python"
mkdir -p "${PROJECT_DRIVE_ROOT}/environment"
meta="${PROJECT_DRIVE_ROOT}/environment"
"${venv_python}" -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee "${meta}/pip-torch.log"
"${venv_python}" -m pip install -r reports/handover_20261024_v2/colab/requirements-linux-cu128.txt 2>&1 | tee "${meta}/pip-install.log"
"${venv_python}" -m pip check 2>&1 | tee "${meta}/pip-check.txt"
"${venv_python}" -m pip freeze > "${meta}/pip-freeze-linux.txt"
"${venv_python}" -m torch.utils.collect_env > "${meta}/torch-collect-env.txt"
"${venv_python}" - <<'PY'
from importlib.metadata import version
import torch
assert version('transformers') == '4.57.6'
assert version('trl') == '1.16.0'
assert torch.__version__ == '2.11.0+cu128'
assert torch.cuda.is_available(), 'CUDA unavailable; do not start a model run'
print('Dependencies installed; GPU model execution remains unverified.')
PY
printf '%s\n' 'RESTART THE COLAB RUNTIME NOW, then remount Drive and rerun the verification cells.' "For shell commands: source ${venv_dir}/bin/activate"
