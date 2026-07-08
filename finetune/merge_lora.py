import argparse
from pathlib import Path


def parse_dtype(dtype: str):
    """把命令行传入的 dtype 名称转换成 torch dtype 对象。

    torch 放在函数内部导入，是为了让 `python merge_lora.py --help`
    不需要等待 PyTorch 和 Transformers 这些大依赖加载。
    """

    import torch

    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="把 LoRA adapter 合并进基座模型，生成可独立部署的完整模型。")

    # adapter 是基于这个模型训练出来的。合并时必须使用同一个基座模型，
    # 否则 LoRA 增量权重会加到错误的参数上。
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="./qwen-lora-adapter")
    parser.add_argument("--output-dir", default="./qwen-lora-merged")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="加载基座模型时使用的权重精度。",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='模型加载位置。默认 auto；如果想在 CPU 上合并，可以传 "cpu"。',
    )
    args = parser.parse_args()

    # 大模型相关依赖延迟导入，让 --help 更快，也让脚本作为 CLI 使用时更轻便。
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(args.adapter)
    output_dir = Path(args.output_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: {adapter_path}. "
            "请先运行 finetune/train_lora.py，或者用 --adapter 指定已有 adapter 目录。"
        )

    # 优先使用 adapter 目录中的 tokenizer 配置，因为训练时可能对 pad/eos 做过对齐。
    # 如果 adapter 里没有 tokenizer，再回退到基座模型的 tokenizer。
    tokenizer_source = args.adapter if (adapter_path / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 第 1 步：加载原始基座模型权重。
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=parse_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )

    # 第 2 步：把训练好的 LoRA adapter 挂到基座模型上。
    # 此时模型仍然是“基座权重 + LoRA 增量权重”的运行时组合。
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    # 第 3 步：把 LoRA 增量权重合并进基座权重。
    # 数学上可以理解为：W_merged = W_base + (lora_alpha / r) * B @ A。
    # merge_and_unload() 返回普通 Transformers 模型，部署时不再需要 PeftModel。
    print("Merging LoRA weights into the base model...")
    merged = model.merge_and_unload()
    merged.config.use_cache = True

    # 第 4 步：保存完整模型目录。
    # model.safetensors 中已经包含合并后的完整权重，可以直接用
    # AutoModelForCausalLM.from_pretrained(output_dir) 加载。
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    if getattr(merged, "generation_config", None) is not None:
        merged.generation_config.save_pretrained(str(output_dir))

    print(f"Merged model saved to: {output_dir}")
    print("现在可以直接用 AutoModelForCausalLM.from_pretrained() 加载这个完整模型。")


if __name__ == "__main__":
    main()
