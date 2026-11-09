"""Fake google.colab.runtime for the mock VM: unassign() marks the VM dead and ends the kernel."""
import json
import os
from pathlib import Path


def unassign():
    vm = Path(os.environ['MOCK_VM_DIR'])
    state = json.loads((vm / 'state.json').read_text())
    state['alive'] = False
    state['stopped_by'] = 'runtime.unassign'
    (vm / 'state.json').write_text(json.dumps(state))
    os._exit(0)
