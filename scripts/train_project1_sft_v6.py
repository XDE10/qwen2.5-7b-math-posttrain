"""Versioned SFT entry; no model is loaded merely by importing it."""
from src.training.approved_hf_v6 import cli

if __name__ == '__main__':
    cli('sft', False)
