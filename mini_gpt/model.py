from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    # MiniGPT 的所有结构超参数集中放在这里，训练和生成 checkpoint 都会保存这份配置。
    vocab_size: int
    block_size: int = 128  # 模型一次最多看多少个历史 token，也叫 context length。
    n_layer: int = 4  # Transformer Block 堆叠层数。
    n_head: int = 4  # 多头注意力的 head 数量。
    n_embd: int = 128  # 每个 token/position 向量的维度。
    dropout: float = 0.1  # 训练时随机丢弃部分连接，缓解过拟合。


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # embedding 维度必须能平均分给每个 head。
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        # 一次线性变换同时生成 Q/K/V，比分开写三个 Linear 更紧凑。
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # 下三角 causal mask:
        # mask[i, j] = 1 表示位置 i 可以看位置 j；j > i 的未来位置会被挡住。
        # register_buffer 会让 mask 跟着模型保存/移动到 GPU，但它不是可训练参数。
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape

        # One linear layer computes Q/K/V together, then we split into heads:
        # q,k,v: (batch, n_head, tokens, head_size)
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(batch, tokens, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch, tokens, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch, tokens, self.n_head, self.head_size).transpose(1, 2)

        # Scaled dot-product self-attention. The causal mask makes token t only
        # attend to positions <= t, so the model cannot peek at future answers.
        scores = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        scores = scores.masked_fill(self.mask[:, :, :tokens, :tokens] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # weights @ v 得到每个位置聚合后的上下文表示，再把多头拼回 channels 维度。
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, tokens, channels)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # GPT 常用 Pre-LN 结构：先 LayerNorm，再进入 attention/MLP。
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)

        # MLP 给每个 token 的表示做非线性变换；4*n_embd 是 Transformer 常见扩展比例。
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接：保留原始信息，同时叠加 attention/MLP 学到的新信息。
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # token_embedding: 把字符 id 映射成向量。
        # position_embedding: 把绝对位置 0..block_size-1 映射成向量。
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Decoder-only Transformer 主体。
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)

        # 输出层把隐藏向量投影回 vocab 维度，得到每个字符的 logits。
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx: (batch, tokens)，里面是字符 id。
        batch, tokens = idx.shape
        if tokens > self.config.block_size:
            raise ValueError(f"sequence length {tokens} exceeds block_size {self.config.block_size}")

        # Learned position encoding: each absolute position gets its own vector.
        positions = torch.arange(tokens, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.head(self.ln(x))

        loss = None
        if targets is not None:
            # Cross entropy compares predicted next-token logits with targets.
            # Shape is flattened from (batch, tokens, vocab) to (batch*tokens, vocab).
            loss = F.cross_entropy(logits.view(batch * tokens, -1), targets.view(batch * tokens))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # 生成越来越长时，只保留最后 block_size 个 token 作为上下文。
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)

            # 只用最后一个位置的 logits 来采样“下一个 token”。
            # temperature 越低越保守，越高越随机。
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            # 把新 token 拼回序列，进入下一轮自回归生成。
            idx = torch.cat((idx, next_id), dim=1)
        return idx
