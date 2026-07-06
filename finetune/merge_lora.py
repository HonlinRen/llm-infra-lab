import argparse
from pathlib import Path


def parse_dtype(dtype: str):
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
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the base model for standalone deployment."
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="./qwen-lora-adapter")
    parser.add_argument("--output-dir", default="./qwen-lora-merged")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Weight dtype used while loading the base model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Device map for loading the model. Use "cpu" to merge on CPU.',
    )
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(args.adapter)
    output_dir = Path(args.output_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: {adapter_path}. "
            "Run serving/qwen_llm.py first or pass --adapter."
        )

    tokenizer_source = args.adapter if (adapter_path / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=parse_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    print("Merging LoRA weights into the base model...")
    merged = model.merge_and_unload()
    merged.config.use_cache = True

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    if getattr(merged, "generation_config", None) is not None:
        merged.generation_config.save_pretrained(str(output_dir))

    print(f"Merged model saved to: {output_dir}")
    print("You can now load it directly with AutoModelForCausalLM.from_pretrained().")


if __name__ == "__main__":
    main()
