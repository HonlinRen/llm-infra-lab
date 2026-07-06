import argparse
from pathlib import Path

import torch

from model import GPTConfig, MiniGPT


DEFAULT_TEXT = "AI infra starts with understanding attention, loss, training, and inference.\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny character-level GPT.")
    parser.add_argument("--text-file", default="")
    parser.add_argument("--out", default="outputs/mini_gpt.pt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else DEFAULT_TEXT * 200
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig(vocab_size=len(chars), block_size=args.block_size)
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([data[i : i + args.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    for step in range(args.steps):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "stoi": stoi, "itos": itos}, output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
