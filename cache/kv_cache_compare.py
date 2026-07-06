import argparse
import gc
import statistics
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RunMetrics:
    use_cache: bool
    input_tokens: int
    generated_tokens: int
    latency_seconds: float
    tokens_per_second: float
    peak_cuda_memory_gb: float | None


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_once(model, tokenizer, prompt: str, max_new_tokens: int, use_cache: bool) -> RunMetrics:
    text = build_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    clear_cuda_memory()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            do_sample=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    input_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = outputs.shape[-1] - input_tokens
    peak_cuda_memory_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    )

    return RunMetrics(
        use_cache=use_cache,
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        latency_seconds=elapsed,
        tokens_per_second=generated_tokens / elapsed if elapsed > 0 else 0.0,
        peak_cuda_memory_gb=peak_cuda_memory_gb,
    )


def average_metrics(metrics: list[RunMetrics]) -> RunMetrics:
    first = metrics[0]
    memory_values = [m.peak_cuda_memory_gb for m in metrics if m.peak_cuda_memory_gb is not None]
    return RunMetrics(
        use_cache=first.use_cache,
        input_tokens=first.input_tokens,
        generated_tokens=round(statistics.mean(m.generated_tokens for m in metrics)),
        latency_seconds=statistics.mean(m.latency_seconds for m in metrics),
        tokens_per_second=statistics.mean(m.tokens_per_second for m in metrics),
        peak_cuda_memory_gb=statistics.mean(memory_values) if memory_values else None,
    )


def print_result(metrics: RunMetrics) -> None:
    peak_memory = (
        f"{metrics.peak_cuda_memory_gb:.2f} GB"
        if metrics.peak_cuda_memory_gb is not None
        else "N/A"
    )
    print(
        f"use_cache={str(metrics.use_cache):5} | "
        f"input_tokens={metrics.input_tokens:4} | "
        f"generated_tokens={metrics.generated_tokens:4} | "
        f"latency={metrics.latency_seconds:7.2f}s | "
        f"tokens/s={metrics.tokens_per_second:7.2f} | "
        f"peak_cuda_memory={peak_memory}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Transformers generation with KV Cache enabled and disabled."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--prompt",
        default="用 5 点解释大模型推理中的 KV Cache 是什么，以及为什么它能加速生成。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    for _ in range(args.warmup):
        run_once(model, tokenizer, args.prompt, min(args.max_new_tokens, 16), use_cache=True)

    results: dict[bool, list[RunMetrics]] = {}
    for use_cache in (True, False):
        results[use_cache] = [
            run_once(model, tokenizer, args.prompt, args.max_new_tokens, use_cache)
            for _ in range(args.runs)
        ]

    print("\nKV Cache comparison")
    print(f"model={args.model}")
    print(f"runs={args.runs}, max_new_tokens={args.max_new_tokens}\n")

    cache_on = average_metrics(results[True])
    cache_off = average_metrics(results[False])
    print_result(cache_on)
    print_result(cache_off)

    speedup = cache_on.tokens_per_second / cache_off.tokens_per_second
    saved_seconds = cache_off.latency_seconds - cache_on.latency_seconds
    print(f"\nuse_cache=True speedup: {speedup:.2f}x")
    print(f"latency saved per run: {saved_seconds:.2f}s")


if __name__ == "__main__":
    main()
