import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MERGED_MODEL_PATH = "./qwen-lora-merged"


def main() -> None:
    parser = argparse.ArgumentParser(description="测试合并后的完整 LoRA 模型。")
    parser.add_argument("--model-path", default=MERGED_MODEL_PATH)
    parser.add_argument("--question", default="保持健康的三个提示。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true", help="使用确定性的贪心解码，方便对比输出。")
    args = parser.parse_args()

    # merged 模型测试：这里加载的是 merge_lora.py 生成的完整模型目录。
    # LoRA 增量权重已经被写入基座模型权重，所以不再需要 PeftModel。
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 继续使用和 base / adapter 测试相同的 chat template，
    # 便于做三种模型形态的横向对比。
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
