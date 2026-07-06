# llm-infra-lab
一个面向大模型基础设施学习的实验项目，涵盖本地推理、LoRA 微调、KV Cache 对比、Prefill/Decode 性能分析和推理服务化部署。

## 目录结构

```text
inference/      Transformers 推理、KV Cache 对比、TTFT benchmark
finetune/       LoRA 微调数据、训练脚本、adapter 合并脚本
serving/        vLLM 服务启动说明和 OpenAI 兼容客户端
quantization/   INT4、GGUF、llama.cpp 量化实验说明
mini_gpt/       从零实现的字符级 Mini GPT
Plan.md         学习路线、项目规划和模块说明
```

## 快速开始

```bash
python inference/transformers_infer.py --model Qwen/Qwen2.5-0.5B-Instruct
python inference/kv_cache_test.py --model Qwen/Qwen2.5-0.5B-Instruct
python inference/benchmark_ttft.py --model Qwen/Qwen2.5-0.5B-Instruct
```

## 学习目标

1. 跑通本地小模型推理，理解显存和 token 生成速度。
2. 对比 KV Cache 开关，理解 decode 阶段为什么需要缓存。
3. 记录 TTFT、tokens/s、总延迟，区分 prefill 和 decode。
4. 使用 LoRA 微调小模型，理解 adapter、低秩更新和低显存训练。
5. 用 vLLM 暴露 OpenAI 兼容接口，理解服务化推理。
6. 从零实现 Mini GPT，理解 attention、causal mask、loss 和生成流程。
