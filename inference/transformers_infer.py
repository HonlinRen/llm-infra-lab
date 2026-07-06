import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local inference with Transformers.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="解释一下 KV Cache 是什么")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    text = build_prompt(tokenizer, args.prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    elapsed = time.perf_counter() - start

    generated_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(answer)
    print("\n--- metrics ---")
    print(f"input_tokens={inputs['input_ids'].shape[-1]}")
    print(f"generated_tokens={generated_tokens}")
    print(f"latency_seconds={elapsed:.2f}")
    print(f"tokens_per_second={generated_tokens / elapsed:.2f}")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak_cuda_memory_gb={peak_gb:.2f}")


if __name__ == "__main__":
    main()
