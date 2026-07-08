import argparse


def main() -> None:
    # 这是早期的最小客户端示例；推荐优先使用 client.py。
    # 保留它是为了演示 OpenAI SDK 调用 vLLM endpoint 的最短路径。
    parser = argparse.ArgumentParser(description="Call a vLLM OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="解释一下 PagedAttention 是什么")
    args = parser.parse_args()

    # 延迟导入，避免只查看 --help 时也要求安装 openai。
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc

    # 本地 vLLM 服务默认不校验真实 key，api_key 用 EMPTY 即可。
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0.2,
        max_tokens=256,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
