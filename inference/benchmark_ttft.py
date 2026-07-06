import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure TTFT and decode throughput.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="请详细解释 KV Cache、Prefill、Decode 和 tokens/s 的关系。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }

    start = time.perf_counter()
    worker = Thread(target=model.generate, kwargs=generation_kwargs)
    worker.start()

    first_token_time = None
    chunks = []
    for chunk in streamer:
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
        chunks.append(chunk)
        print(chunk, end="", flush=True)

    worker.join()
    end = time.perf_counter()
    output_text = "".join(chunks)
    output_tokens = len(tokenizer.encode(output_text))
    ttft = (first_token_time or end) - start
    decode_time = max(end - (first_token_time or start), 1e-9)

    print("\n\n--- metrics ---")
    print(f"input_tokens={inputs['input_ids'].shape[-1]}")
    print(f"output_tokens={output_tokens}")
    print(f"ttft_seconds={ttft:.2f}")
    print(f"decode_tokens_per_second={output_tokens / decode_time:.2f}")
    print(f"total_latency_seconds={end - start:.2f}")


if __name__ == "__main__":
    main()
