# Mini GPT: Tiny Shakespeare

This folder trains a tiny character-level Transformer from scratch. It is for
understanding the mechanics, not for producing a useful model.

## What to Observe

- Position encoding: `token_embedding + position_embedding` in `model.py`.
- Self-attention: Q/K/V projection, attention scores, softmax, weighted sum.
- Causal mask: lower-triangular mask prevents looking at future characters.
- Cross entropy loss: next-character prediction loss in `MiniGPT.forward`.
- Backpropagation: `loss.backward()` computes gradients in `train.py`.
- Loss decreasing: printed `train_loss` and `val_loss` during training.

## Train

From the project root:

```bash
python mini_gpt/train.py --steps 3000 --amp
```

The script automatically downloads Tiny Shakespeare to:

```text
data/tiny_shakespeare.txt
```

RTX 4060 8GB should handle the default config comfortably:

```text
block_size=128 batch_size=64 n_layer=4 n_head=4 n_embd=128
```

If you hit CUDA memory pressure, reduce batch size:

```bash
python mini_gpt/train.py --batch-size 32 --steps 3000 --amp
```

For a very quick smoke test:

```bash
python mini_gpt/train.py --steps 20 --eval-interval 10 --eval-iters 2 --batch-size 8
```

## Generate

After training:

```bash
python mini_gpt/generate.py --checkpoint outputs/mini_gpt_shakespeare.pt --prompt "To be" --max-new-tokens 400
```

Temperature controls randomness:

```bash
python mini_gpt/generate.py --temperature 0.7
```

## Larger but Still Small

When the basic version runs, try a slightly larger model:

```bash
python mini_gpt/train.py --n-layer 6 --n-head 6 --n-embd 192 --batch-size 48 --steps 5000 --amp
```

