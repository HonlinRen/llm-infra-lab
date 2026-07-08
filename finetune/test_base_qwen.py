import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description="测试原始 Qwen2.5 基座模型。")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--question", default="保持健康的三个提示。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--greedy", action="store_true", help="使用确定性的贪心解码，方便对比输出。")
    args = parser.parse_args()

    # baseline 测试：这里只加载原始模型，不加载任何 LoRA adapter。
    # 这个输出可以作为“微调前”的对照组。
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Qwen Instruct 模型期望 chat 格式输入。apply_chat_template 会把普通问题
    # 包装成模型熟悉的 user/assistant 对话格式。
    messages = [{"role": "user", "content": args.question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # --greedy 用于稳定对比；不加 --greedy 时会启用采样，回答会有随机性。
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
        )

    # generate 的结果包含“原始 prompt + 新生成 token”，这里把 prompt 部分切掉，
    # 只打印模型新生成的回答。
    answer_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()
