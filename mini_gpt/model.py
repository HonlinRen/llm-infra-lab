from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
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

        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, tokens, channels)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
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
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
