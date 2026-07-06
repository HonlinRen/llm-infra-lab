import argparse

import torch

from model import MiniGPT


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from the tiny GPT checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/mini_gpt.pt")
    parser.add_argument("--prompt", default="AI")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MiniGPT(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    idx = torch.tensor([[stoi.get(ch, 0) for ch in args.prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens)[0].tolist()
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
