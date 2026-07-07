import argparse
import time


def main() -> None:
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

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    messages = [{"role": "user", "content": args.prompt}]

    start = time.perf_counter()
    if args.stream:
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
    if response.usage is not None:
        print(f"prompt_tokens={response.usage.prompt_tokens}")
        print(f"completion_tokens={response.usage.completion_tokens}")
        print(f"total_tokens={response.usage.total_tokens}")


if __name__ == "__main__":
    main()
