import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./qwen-lora-adapter"


def main() -> None:
    parser = argparse.ArgumentParser(description="测试“基座模型 + LoRA adapter”的推理效果。")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=ADAPTER_PATH)
    parser.add_argument("--question", default="保持健康的三个提示。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true", help="使用确定性的贪心解码，方便对比输出。")
    args = parser.parse_args()

    # adapter 推理：先加载 adapter 目录里的 tokenizer，再加载原始基座模型。
    # adapter 本身不会替代基座模型，它只是保存了 LoRA 训练得到的增量权重。
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # PeftModel 会把 LoRA adapter 挂到基座模型上。
    # 推理时等价于使用 W_base + LoRA_delta，但不会修改磁盘上的基座模型文件。
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # 和原始模型测试脚本保持同样的 prompt 构造方式，
    # 这样输出差异主要来自 LoRA 权重，而不是 prompt 格式差异。
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
        # temperature 和 top_p 只在采样模式下生效。
        # 如果要做稳定对比，建议加 --greedy 关闭采样。
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
