"""Quick one-shot or interactive inference against the published QFFT-R1 model (with optional MMR-50 LoRA).

Usage:
    math-posttrain infer --prompt "Solve x^2 - 5x + 6 = 0."
    math-posttrain infer --adapter mmr50_lora --prompt "Find the remainder when 2^100 is divided by 7."
"""
from __future__ import annotations

import argparse

from math_posttrain.formats import messages
from math_posttrain.models import resolve_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='qfft_r1', help='Model alias, HF repo ID, or local path (default: qfft_r1)')
    p.add_argument('--adapter', help='Optional LoRA adapter alias (e.g. mmr50_lora), HF repo ID, or local path')
    p.add_argument('--prompt', help='Math problem to solve (omit for interactive REPL)')
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--temperature', type=float, default=0.6)
    p.add_argument('--top-p', type=float, default=0.95)
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model(a.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map='auto')

    if a.adapter:
        from peft import PeftModel
        adapter_path = resolve_model(a.adapter)
        model = PeftModel.from_pretrained(model, str(adapter_path))

    def answer_one(question: str) -> str:
        prompt_text = tokenizer.apply_chat_template(
            messages(question), tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=a.max_new_tokens,
                do_sample=a.temperature > 0,
                temperature=a.temperature if a.temperature > 0 else None,
                top_p=a.top_p if a.temperature > 0 else None,
            )
        return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    if a.prompt:
        print(answer_one(a.prompt))
        return

    print('Interactive math inference (Ctrl+D / empty line to exit):')
    while True:
        try:
            q = input('\nProblem> ').strip()
        except EOFError:
            break
        if not q:
            break
        print('\n' + answer_one(q))


if __name__ == '__main__':
    main()
