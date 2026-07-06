import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_PROMPT = (
    "Explain what KV Cache is in LLM inference. Include why it improves "
    "generation speed and what tradeoff it introduces."
)


@dataclass
class BenchmarkResult:
    mode: str
    model_name: str
    load_seconds: float
    cuda_allocated_after_load_gb: Optional[float]
    cuda_reserved_after_load_gb: Optional[float]
    cuda_peak_allocated_gb: Optional[float]
    input_tokens: int
    generated_tokens: int
    generation_seconds: float
    tokens_per_second: float
    answer: str


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def cuda_gb(value: int) -> float:
    return value / 1024**3


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_model(model_name: str, mode: str):
    if mode == "fp16":
        kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto",
            "trust_remote_code": True,
        }
    elif mode == "int4":
        if not torch.cuda.is_available():
            raise RuntimeError("INT4 bitsandbytes inference requires CUDA.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs = {
            "quantization_config": quant_config,
            "device_map": "auto",
            "trust_remote_code": True,
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def run_once(args, tokenizer, mode: str) -> BenchmarkResult:
    cleanup_cuda()

    load_start = time.perf_counter()
    model = load_model(args.model, mode)
    load_seconds = time.perf_counter() - load_start

    if torch.cuda.is_available():
        allocated_after_load = cuda_gb(torch.cuda.memory_allocated())
        reserved_after_load = cuda_gb(torch.cuda.memory_reserved())
    else:
        allocated_after_load = None
        reserved_after_load = None

    text = build_prompt(tokenizer, args.prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "use_cache": True,
    }
    if not args.do_sample:
        gen_kwargs.pop("temperature")

    gen_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**gen_kwargs)
    generation_seconds = time.perf_counter() - gen_start

    input_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = outputs.shape[-1] - input_tokens
    answer = tokenizer.decode(
        outputs[0][input_tokens:],
        skip_special_tokens=True,
    ).strip()

    if torch.cuda.is_available():
        peak_allocated = cuda_gb(torch.cuda.max_memory_allocated())
    else:
        peak_allocated = None

    result = BenchmarkResult(
        mode=mode,
        model_name=args.model,
        load_seconds=load_seconds,
        cuda_allocated_after_load_gb=allocated_after_load,
        cuda_reserved_after_load_gb=reserved_after_load,
        cuda_peak_allocated_gb=peak_allocated,
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        generation_seconds=generation_seconds,
        tokens_per_second=generated_tokens / max(generation_seconds, 1e-9),
        answer=answer,
    )

    del model
    cleanup_cuda()
    return result


def format_number(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def print_markdown_table(results: list[BenchmarkResult]) -> None:
    print("\n## Metrics")
    print(
        "| mode | load_s | mem_after_load_gb | reserved_after_load_gb | "
        "peak_mem_gb | input_tokens | output_tokens | gen_s | tokens/s |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for item in results:
        print(
            f"| {item.mode} "
            f"| {item.load_seconds:.2f} "
            f"| {format_number(item.cuda_allocated_after_load_gb)} "
            f"| {format_number(item.cuda_reserved_after_load_gb)} "
            f"| {format_number(item.cuda_peak_allocated_gb)} "
            f"| {item.input_tokens} "
            f"| {item.generated_tokens} "
            f"| {item.generation_seconds:.2f} "
            f"| {item.tokens_per_second:.2f} |"
        )


def print_quality_section(prompt: str, results: list[BenchmarkResult]) -> None:
    print("\n## Answer Quality Samples")
    print("Use the same prompt below to compare factuality, completeness, and fluency.")
    print(f"\nPrompt:\n{prompt}\n")
    for item in results:
        print(f"### {item.mode.upper()} answer")
        print(item.answer)
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FP16 and INT4 inference memory, load time, speed, and output quality."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--mode",
        choices=["both", "fp16", "int4"],
        default="both",
        help="Run both modes or only one mode.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. By default generation is deterministic for easier comparison.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--json-output",
        help="Optional path to save raw benchmark results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = ["fp16", "int4"] if args.mode == "both" else [args.mode]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    results = []
    for mode in modes:
        print(f"\nRunning {mode.upper()} benchmark...")
        results.append(run_once(args, tokenizer, mode))

    print_markdown_table(results)
    print_quality_section(args.prompt, results)

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as file:
            json.dump([asdict(item) for item in results], file, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON results to {args.json_output}")


if __name__ == "__main__":
    main()
