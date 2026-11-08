"""One command: sampler subprocess exits/releases GPU, then CPU approval checks."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='Other arguments are forwarded unchanged to sample_rl_candidates_v4')
    p.add_argument('--output-root',required=True)
    p.add_argument('--approval-output-root',required=True)
    a,remaining=p.parse_known_args()
    root=Path(a.output_root).resolve()
    subprocess.run([sys.executable,'-m','scripts.sample_rl_candidates_v4','--output-root',str(root),*remaining],check=True)
    if not (root/'sampling_complete.json').is_file():raise RuntimeError('Sampling paused/incomplete; persisted records retained; approval not executed')
    subprocess.run([sys.executable,'-m','scripts.approve_rl_pool_v4','--sampling-root',str(root),'--output-root',str(Path(a.approval_output_root).resolve())],check=True)


if __name__=='__main__':main()
