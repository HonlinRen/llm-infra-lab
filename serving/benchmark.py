import argparse
import csv
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class RequestResult:
    # 单个请求的结果。即使请求失败，也记录耗时和错误信息，便于写入 CSV 后复盘。
    ok: bool
    latency_seconds: float
    completion_tokens: int
    ttft_seconds: float = 0.0
    error: str = ""


def percentile(values: List[float], percent: float) -> float:
    """计算简单百分位数，用于 p50 / p95 latency。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent)
    return ordered[index]


def parse_concurrency(value: str) -> List[int]:
    """把 `1,4,8` 这样的命令行参数解析成整数列表。"""
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    if not result:
        raise argparse.ArgumentTypeError("at least one concurrency value is required")
    return result


def run_request(args: argparse.Namespace) -> RequestResult:
    # 每个 worker 线程都会调用这里，发起一次 OpenAI-compatible 请求。
    # 这里在函数内部创建 client，避免多线程共享同一个 http client 造成干扰。
    try:
        from openai import OpenAI
    except ImportError:
        return RequestResult(
            ok=False,
            latency_seconds=0.0,
            completion_tokens=0,
            error="Missing dependency: pip install openai",
        )

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    start = time.perf_counter()
    try:
        if args.stream:
            # 流式请求用于统计 TTFT。vLLM 生成首个 token 后会先返回一个 chunk。
            # benchmark 不打印正文，只统计时间和 usage，避免并发输出互相打乱。
            first_token_at = None
            completion_tokens = 0
            stream = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": args.prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            for event in stream:
                # 某些 streaming 事件可能只携带 usage，不携带 choices。
                if event.choices:
                    delta = event.choices[0].delta.content or ""
                    if delta and first_token_at is None:
                        first_token_at = time.perf_counter()
                # include_usage=True 时，最后一个事件会带 usage。
                usage = getattr(event, "usage", None)
                if usage is not None:
                    completion_tokens = usage.completion_tokens or 0

            end = time.perf_counter()
            return RequestResult(
                True,
                end - start,
                completion_tokens,
                (first_token_at or end) - start,
            )

        # 非流式请求只能统计总延迟，无法知道首 token 何时到达。
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        latency = time.perf_counter() - start
        completion_tokens = 0
        if response.usage is not None:
            completion_tokens = response.usage.completion_tokens or 0
        return RequestResult(True, latency, completion_tokens, 0.0)
    except Exception as exc:  # Keep the benchmark running so failures are visible in CSV.
        # benchmark 不因为一个请求失败就整体退出，而是把错误写到结果里。
        latency = time.perf_counter() - start
        return RequestResult(
            ok=False,
            latency_seconds=latency,
            completion_tokens=0,
            error=str(exc),
        )


def run_batch(args: argparse.Namespace, concurrency: int) -> dict:
    # 每个 concurrency 档位至少发 concurrency 个请求，否则线程池不能真正打满。
    total_requests = max(args.requests_per_level, concurrency)
    started = time.perf_counter()
    results: List[RequestResult] = []

    # ThreadPoolExecutor 用来模拟多个用户同时请求 vLLM server。
    # vLLM 的 continuous batching 优势主要在这种并发场景体现。
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_request, args) for _ in range(total_requests)]
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.perf_counter() - started
    # 只用成功请求计算 latency / throughput，失败请求单独计数。
    ok_results = [result for result in results if result.ok]
    latencies = [result.latency_seconds for result in ok_results]
    ttfts = [result.ttft_seconds for result in ok_results if result.ttft_seconds > 0]
    output_tokens = sum(result.completion_tokens for result in ok_results)

    return {
        "model": args.model,
        "concurrency": concurrency,
        "requests": total_requests,
        "success": len(ok_results),
        "failed": total_requests - len(ok_results),
        "avg_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "avg_ttft_s": statistics.mean(ttfts) if ttfts else 0.0,
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "wall_time_s": elapsed,
        # req/s 表示单位时间内完成了多少个成功请求。
        "req_per_s": len(ok_results) / elapsed if elapsed > 0 else 0.0,
        "output_tokens": output_tokens,
        # output_tokens_per_s 是服务化推理最重要的吞吐指标之一。
        "output_tokens_per_s": output_tokens / elapsed if elapsed > 0 else 0.0,
        "first_error": next((result.error for result in results if not result.ok), ""),
    }


def write_results(path: Path, rows: Iterable[dict]) -> None:
    # 将每个并发档位的统计结果写入 CSV，便于后续写报告或画图。
    rows = list(rows)
    fieldnames = [
        "model",
        "concurrency",
        "requests",
        "success",
        "failed",
        "avg_latency_s",
        "avg_ttft_s",
        "p50_latency_s",
        "p95_latency_s",
        "wall_time_s",
        "req_per_s",
        "output_tokens",
        "output_tokens_per_s",
        "first_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_row(row: dict) -> None:
    # 控制台输出一行摘要，方便边跑边观察吞吐变化。
    print(
        "concurrency={concurrency} "
        "success={success}/{requests} "
        "avg_latency={avg_latency_s:.3f}s "
        "avg_ttft={avg_ttft_s:.3f}s "
        "p95={p95_latency_s:.3f}s "
        "req/s={req_per_s:.3f} "
        "output_tokens/s={output_tokens_per_s:.3f}".format(**row)
    )
    if row["first_error"]:
        print(f"  first_error={row['first_error']}")


def main() -> None:
    # benchmark 的默认值对应本实验：1/4/8 并发，每档 8 个请求，模型与 server 一致。
    parser = argparse.ArgumentParser(
        description="Benchmark a vLLM OpenAI-compatible endpoint with concurrent requests."
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default="Explain transformer architecture.")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming responses to measure TTFT.",
    )
    parser.add_argument("--concurrency", type=parse_concurrency, default=[1, 4, 8])
    parser.add_argument("--requests-per-level", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results.csv"),
    )
    args = parser.parse_args()

    # 逐个并发档位运行，避免不同档位之间互相干扰。
    rows = []
    for concurrency in args.concurrency:
        row = run_batch(args, concurrency)
        rows.append(row)
        print_row(row)

    write_results(args.output, rows)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
