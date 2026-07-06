import argparse

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Call a vLLM OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="解释一下 PagedAttention 是什么")
    args = parser.parse_args()

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
