import argparse
import time


def main() -> None:
    # 这个脚本用于验证 vLLM OpenAI-compatible API 是否可用。
    # 它只发起一个请求，适合在 server 启动后做 smoke test。
    parser = argparse.ArgumentParser(
        description="Call a vLLM OpenAI-compatible chat completion endpoint."
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default="Explain KV Cache and PagedAttention.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens and print TTFT metrics.",
    )
    args = parser.parse_args()

    # 延迟导入 OpenAI SDK，这样 `python client.py --help` 不依赖 openai 包已经安装。
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc

    # vLLM 使用 OpenAI 协议，但本地服务不校验真实 API key，通常传 EMPTY 即可。
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    messages = [{"role": "user", "content": args.prompt}]

    start = time.perf_counter()
    if args.stream:
        # 流式模式可以记录 TTFT：从请求发出到收到第一个 token/chunk 的时间。
        # TTFT 能反映 prefill、调度和服务端首包响应开销。
        first_token_at = None
        chunks = []
        stream = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=True,
        )
        for event in stream:
            # OpenAI streaming 事件里，新增文本放在 delta.content 中。
            delta = event.choices[0].delta.content or ""
            if delta and first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(delta)
            print(delta, end="", flush=True)

        end = time.perf_counter()
        print("\n\n--- metrics ---")
        print(f"ttft_seconds={(first_token_at or end) - start:.3f}")
        print(f"latency_seconds={end - start:.3f}")
        print(f"output_chars={len(''.join(chunks))}")
        return

    # 非流式模式会等完整回答生成完后一次性返回，适合简单验证内容是否正确。
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    end = time.perf_counter()
    print(response.choices[0].message.content)
    print("\n--- metrics ---")
    print(f"latency_seconds={end - start:.3f}")
    # vLLM 会返回 usage 时，顺便打印 token 数，方便估算 tokens/s。
    if response.usage is not None:
        print(f"prompt_tokens={response.usage.prompt_tokens}")
        print(f"completion_tokens={response.usage.completion_tokens}")
        print(f"total_tokens={response.usage.total_tokens}")


if __name__ == "__main__":
    main()
