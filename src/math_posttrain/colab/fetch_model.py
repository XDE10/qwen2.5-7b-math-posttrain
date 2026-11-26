"""Download one pinned registry model to Drive (python -m math_posttrain.colab.fetch_model base)."""
import sys

from math_posttrain.models import download_to_drive

if __name__ == '__main__':
    for name in sys.argv[1:]:
        download_to_drive(name)
