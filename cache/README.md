# KV Cache 开关对比实验

这个实验用 Hugging Face Transformers 对比 `model.generate()` 中 `use_cache=True` 和
`use_cache=False` 的生成性能差异。

KV Cache 会在 decode 阶段缓存历史 token 的 Key/Value。开启后，每一步生成新 token 时只需要
计算新 token 的注意力；关闭后，每一步都会重复计算完整历史上下文，所以通常会明显变慢。

## 运行

```bash
python cache/kv_cache_compare.py --model Qwen/Qwen2.5-0.5B-Instruct
```

如果本机显存较小，可以先减少生成长度：

```bash
python cache/kv_cache_compare.py --max-new-tokens 64
```

也可以换成本地模型目录：

```bash
python cache/kv_cache_compare.py --model models/SmolLM2-1.7B-Instruct
```

## 关键代码

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    use_cache=True,
)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    use_cache=False,
)
```

预期现象：

- `use_cache=True`：生成更快，tokens/s 更高。
- `use_cache=False`：会重复计算历史 token，生成速度更慢。

