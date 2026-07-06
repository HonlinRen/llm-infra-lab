# GGUF / llama.cpp 实验

目标：了解 GGUF 量化模型在本地 CPU/GPU 混合推理中的使用方式，并和 Transformers 路线做对比。

## 推荐模型

```text
Qwen2.5-1.5B-Instruct-GGUF
Qwen2.5-7B-Instruct-GGUF
```

优先选择 Q4_K_M 或 Q5_K_M 量化版本，通常能在体积、速度和质量之间取得比较好的平衡。

## 观察内容

- 模型文件体积。
- 加载速度。
- prompt tokens/s 与 generation tokens/s。
- CPU 与 GPU offload 层数变化对速度的影响。

## 对比结论模板

```text
Transformers 更适合和微调、Python 生态、实验代码结合。
GGUF / llama.cpp 更适合轻量部署、低显存推理和本地工具集成。
```
