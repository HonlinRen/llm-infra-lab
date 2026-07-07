import argparse
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import torch

from model import GPTConfig, MiniGPT


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DEFAULT_DATA_PATH = Path("data/tiny_shakespeare.txt")


def read_or_download_text(path: Path) -> str:
    # 第一次运行时下载 Tiny Shakespeare；后续直接读取本地文件，避免重复联网。
    if path.exists():
        return path.read_text(encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading Tiny Shakespeare to {path} ...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    # 字符级语言模型：vocab 是文本中出现过的所有字符。
    # stoi: string to int，用于编码；itos: int to string，用于解码生成结果。
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    # 把整份文本变成一长串 token id，形状是 (num_chars,)。
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data

    # 随机选择 batch_size 个起点，每个样本长度是 block_size。
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))

    # x 是输入序列，y 是目标序列；y 比 x 向右错开一位。
    # 例如 x="To be"，y="o be "，模型学习每个位置预测下一个字符。
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
    eval_iters: int,
) -> dict[str, float]:
    # 评估时不需要 dropout，也不需要计算梯度；@torch.no_grad() 会节省显存和时间。
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, train_data, val_data, batch_size, block_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    # 评估结束切回训练模式，恢复 dropout 等训练行为。
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny character-level Transformer on Tiny Shakespeare.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", type=Path, default=Path("outputs/mini_gpt_shakespeare.pt"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    args = parser.parse_args()

    # 固定随机种子，让同一套参数下的训练曲线更容易复现。
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

        # 允许 PyTorch 使用更快的矩阵乘法实现，适合 NVIDIA GPU 上的训练。
        torch.set_float32_matmul_precision("high")

    # 数据准备：文本 -> vocab -> token id -> train/val 切分。
    text = read_or_download_text(args.data)
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    # 90% 训练，10% 验证。验证集不参与参数更新，只用于观察泛化情况。
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # 根据命令行参数创建一个小 GPT。
    config = GPTConfig(
        vocab_size=len(stoi),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MiniGPT(config).to(args.device)

    # AdamW 是训练 Transformer 的常用优化器；weight decay 逻辑比 Adam 更适合权重衰减。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # AMP: 自动混合精度。开启 --amp 且设备是 cuda 时，用 fp16 加速部分计算。
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device == "cuda")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={args.device} vocab={len(stoi)} params={n_params:.2f}M chars={len(text):,}")

    start = time.time()
    for step in range(args.steps + 1):
        # 每隔 eval_interval 步，分别估计训练集和验证集 loss。
        # train_loss 下降说明模型在学；val_loss 下降说明不是只记住训练数据。
        if step % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                args.block_size,
                args.device,
                args.eval_iters,
            )
            elapsed = time.time() - start
            print(
                f"step={step:5d} train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} elapsed={elapsed:.1f}s"
            )

        xb, yb = get_batch("train", train_data, val_data, args.batch_size, args.block_size, args.device)

        # Forward pass computes logits and cross entropy loss. Backward pass fills
        # gradients, and optimizer.step updates model weights so loss can go down.
        # autocast 只影响 with 块中的前向计算；loss 仍然会被 scaler 安全地反传。
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and args.device == "cuda"):
            _, loss = model(xb, yb)

        # 标准训练三步：
        # 1. 清梯度；2. loss 反向传播算梯度；3. optimizer 根据梯度更新参数。
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # checkpoint 不只保存权重，还保存 config 和字符表；生成脚本需要这些信息还原模型。
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "stoi": stoi,
            "itos": itos,
            "source": str(args.data),
        },
        args.out,
    )
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
