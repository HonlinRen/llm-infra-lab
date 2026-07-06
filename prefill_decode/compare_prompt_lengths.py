from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from prefill_decode_demo import (
    PrefillDecodeMetrics,
    load_model_and_tokenizer,
    run_prefill_decode,
)


PROMPT_UNIT = (
    "KV Cache stores the key and value tensors already computed by attention layers. "
    "During prefill, the model reads the whole prompt and builds this cache. "
    "During decode, the model only processes the newest token and reuses cached keys and values. "
    "This makes long generation faster because old prompt tokens do not need to be recomputed. "
)


def build_repeated_prompt(repeat: int) -> str:
    context = PROMPT_UNIT * repeat
    return (
        "Read the following notes, then explain the difference between prefill and decode.\n\n"
        f"{context}\n"
        "Answer in three concise bullet points."
    )


def print_comparison(rows: list[tuple[str, PrefillDecodeMetrics]]) -> None:
    print("\n--- prompt length comparison ---")
    print(
        "case   input_tokens  prefill_s  ttft_s  decode_ms/token  decode_tok/s  total_s  peak_mem_gb"
    )
    for label, metrics in rows:
        peak_memory = (
            f"{metrics.peak_cuda_memory_gb:.2f}"
            if metrics.peak_cuda_memory_gb is not None
            else "N/A"
        )
        print(
            f"{label:<6} "
            f"{metrics.input_tokens:>12} "
            f"{metrics.prefill_seconds:>10.4f} "
            f"{metrics.ttft_seconds:>7.4f} "
            f"{metrics.avg_decode_latency_ms:>15.2f} "
            f"{metrics.decode_tokens_per_second:>12.2f} "
            f"{metrics.total_latency_seconds:>8.4f} "
            f"{peak_memory:>11}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare prefill and decode metrics for short, medium, and long prompts."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--short-repeat",
        type=int,
        default=1,
        help="How many prompt units to use for the short case.",
    )
    parser.add_argument(
        "--medium-repeat",
        type=int,
        default=8,
        help="How many prompt units to use for the medium case.",
    )
    parser.add_argument(
        "--long-repeat",
        type=int,
        default=32,
        help="How many prompt units to use for the long case.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save all prompts, outputs, and metrics as JSON.",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    for name in ("short_repeat", "medium_repeat", "long_repeat"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model_and_tokenizer(args.model)

    cases = [
        ("short", args.short_repeat),
        ("medium", args.medium_repeat),
        ("long", args.long_repeat),
    ]

    rows: list[tuple[str, PrefillDecodeMetrics]] = []
    payload = []
    for label, repeat in cases:
        prompt = build_repeated_prompt(repeat)
        print(f"\nRunning {label} prompt: repeat={repeat}")
        generated_text, metrics = run_prefill_decode(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=True,
            temperature=0.0,
            top_p=1.0,
        )
        rows.append((label, metrics))
        payload.append(
            {
                "case": label,
                "repeat": repeat,
                "prompt": prompt,
                "generated_text": generated_text,
                "metrics": asdict(metrics),
            }
        )

    print_comparison(rows)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved_json={args.json_out}")


if __name__ == "__main__":
    main()
