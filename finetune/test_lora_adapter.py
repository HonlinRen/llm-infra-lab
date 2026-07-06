import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./qwen-lora-adapter"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the base Qwen model with a LoRA adapter.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=ADAPTER_PATH)
    parser.add_argument("--question", default="保持健康的三个提示。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true", help="Use deterministic greedy decoding.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    messages = [{"role": "user", "content": args.question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.greedy,
    }
    if not args.greedy:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    answer_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()
