import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the base Qwen2.5 model locally.")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--question", default="保持健康的三个提示。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--greedy", action="store_true", help="Use deterministic greedy decoding.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    messages = [{"role": "user", "content": args.question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
        )

    answer_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()
