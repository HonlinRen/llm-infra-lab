import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_once(model, tokenizer, prompt: str, max_new_tokens: int, use_cache: bool) -> dict:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
        )
    elapsed = time.perf_counter() - start
    generated_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]

    return {
        "use_cache": use_cache,
        "input_tokens": inputs["input_ids"].shape[-1],
        "generated_tokens": generated_tokens,
        "latency_seconds": elapsed,
        "tokens_per_second": generated_tokens / elapsed,
        "peak_cuda_memory_gb": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generation with and without KV Cache.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="用 5 点解释 prefill 和 decode 的区别")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    for use_cache in (True, False):
        metrics = run_once(model, tokenizer, args.prompt, args.max_new_tokens, use_cache)
        print(metrics)


if __name__ == "__main__":
    main()
