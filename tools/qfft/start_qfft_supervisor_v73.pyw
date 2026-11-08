"""Task Scheduler entry: independent process host, durable stdout/stderr, no retries."""
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / 'artifacts/qfft_all_supervisor_v71'
CONTROL.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    with (CONTROL / 'supervisor_v73_stdout.log').open('a', encoding='utf-8', buffering=1) as output:
        with (CONTROL / 'supervisor_v73_stderr.log').open('a', encoding='utf-8', buffering=1) as errors:
            sys.stdout, sys.stderr = output, errors
            import continue_qfft_all_v71 as supervisor
            # Scheduler runs a GUI Python; generation children use the normal CLI interpreter.
            sys.executable = str(Path(sys.executable).with_name('python.exe'))
            try:
                supervisor.main()
            except SystemExit:
                raise
            except Exception as error:
                supervisor.status('supervisor_error', error_type=type(error).__name__,
                                  saved_candidates=supervisor.saved_count())
                traceback.print_exc()
                raise
