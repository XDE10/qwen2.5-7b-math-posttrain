"""Download one pinned registry model to Drive (python -m experiments.qfft_v93.colab.fetch_model base)."""
import sys

from experiments.qfft_v93.models import download_to_drive

if __name__ == '__main__':
    for name in sys.argv[1:]:
        download_to_drive(name)
