from __future__ import annotations

"""对比短、中、长 prompt 下的 Prefill/Decode 指标。

这个脚本复用 `prefill_decode_demo.py` 中的拆解函数，但只加载一次模型，
然后连续运行 short / medium / long 三组 prompt。这样可以避免把模型加载时间
混入 TTFT，也能更直观看到：

1. prompt 越长，prefill/TTFT 通常越高。
2. prompt 越长，KV Cache 越大，峰值显存通常越高。
3. decode 单 token 延迟一般比 prefill 更稳定，但超长上下文下也会略受影响。
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from prefill_decode_demo import (
    PrefillDecodeMetrics,
    load_model_and_tokenizer,
    run_prefill_decode,
)


# 每重复一次 PROMPT_UNIT，就会向 prompt 里加入一段关于 KV Cache 的说明。
# 这样做的好处是内容语义稳定，不需要手工复制大段文本；
# 通过 --short-repeat / --medium-repeat / --long-repeat 就能控制输入长度。
PROMPT_UNIT = (
    "KV Cache stores the key and value tensors already computed by attention layers. "
    "During prefill, the model reads the whole prompt and builds this cache. "
    "During decode, the model only processes the newest token and reuses cached keys and values. "
    "This makes long generation faster because old prompt tokens do not need to be recomputed. "
)


def build_repeated_prompt(repeat: int) -> str:
    """构造指定长度的实验 prompt。

    repeat 越大，最终 input_tokens 越多，prefill 阶段要处理的上下文越长。
    注意不同 tokenizer 对同一文本的切分方式不同，所以 repeat 和 token 数
    不是严格线性关系，最终以输出表格中的 input_tokens 为准。
    """

    context = PROMPT_UNIT * repeat
    return (
        "Read the following notes, then explain the difference between prefill and decode.\n\n"
        f"{context}\n"
        "Answer in three concise bullet points."
    )


def print_comparison(rows: list[tuple[str, PrefillDecodeMetrics]]) -> None:
    """打印 short / medium / long 三组实验的核心指标对比表。

    这里重点关注三类指标：
    - input_tokens / prefill_s / ttft_s：观察输入长度对首 token 的影响。
    - decode_ms/token / decode_tok/s：观察逐 token 生成是否稳定。
    - peak_mem_gb：观察长 prompt 带来的 KV Cache 显存增长。
    """

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
    """解析对比实验参数。

    默认 repeat 配置比较温和，适合先快速跑通；如果想明显放大长 prompt 的
    prefill 成本，可以把 --long-repeat 调到 64 或更高。
    """

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
    """命令行入口：同一个模型连续跑三种 prompt 长度并汇总结果。"""

    args = parse_args()
    # 模型只加载一次。否则每组实验都重新加载模型，会把磁盘读取、权重搬运、
    # CUDA 初始化等开销混进对比结果，导致 prefill 指标不干净。
    tokenizer, model = load_model_and_tokenizer(args.model)

    # 三组 case 的标签固定，repeat 由命令行控制。输出表里 label 用来区分行。
    cases = [
        ("short", args.short_repeat),
        ("medium", args.medium_repeat),
        ("long", args.long_repeat),
    ]

    rows: list[tuple[str, PrefillDecodeMetrics]] = []
    # payload 用于保存完整 JSON，包括 prompt、生成文本和指标，方便后续复盘。
    payload = []
    for label, repeat in cases:
        prompt = build_repeated_prompt(repeat)
        print(f"\nRunning {label} prompt: repeat={repeat}")
        # 这里固定 temperature=0.0，用贪心解码降低随机性。
        # 如果开启采样，模型可能更早或更晚遇到 EOS，影响输出 token 数和耗时。
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

    # 三组都跑完后再打印汇总表，方便横向比较。
    print_comparison(rows)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        # ensure_ascii=False 保留中文和正常文本，打开 JSON 时更易读。
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved_json={args.json_out}")


if __name__ == "__main__":
    main()
