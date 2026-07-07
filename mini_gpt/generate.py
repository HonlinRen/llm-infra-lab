import argparse
from pathlib import Path

import torch

from model import GPTConfig, MiniGPT


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a mini GPT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/mini_gpt_shakespeare.pt"))
    parser.add_argument("--prompt", default="To be")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # checkpoint 里包含模型权重、模型结构配置、字符表。
    # map_location 让 checkpoint 可以在 cuda/cpu 之间切换加载。
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # 用训练时保存的 config 重建同样结构的 MiniGPT，再加载权重。
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config).to(args.device)
    model.load_state_dict(checkpoint["model"])

    # 生成时关闭 dropout，保证推理行为稳定。
    model.eval()

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    # 如果 prompt 里有训练 vocab 没见过的字符，就退化成换行符或 0。
    unk_id = stoi.get("\n", 0)

    # 把用户输入的 prompt 编码成 token ids，形状是 (1, prompt_len)。
    idx = torch.tensor([[stoi.get(ch, unk_id) for ch in args.prompt]], dtype=torch.long, device=args.device)

    # 自回归生成：每轮根据当前上下文采样一个 next token，并拼回 idx。
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature)[0].tolist()

    # 把 token ids 解码回字符串。
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
