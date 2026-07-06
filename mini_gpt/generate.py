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

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    unk_id = stoi.get("\n", 0)
    idx = torch.tensor([[stoi.get(ch, unk_id) for ch in args.prompt]], dtype=torch.long, device=args.device)

    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature)[0].tolist()
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
