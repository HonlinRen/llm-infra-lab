# llm-infra-lab 实验计划

这份计划已经和 `README.md` 合并为同一套项目逻辑。以后：

- `README.md` 作为主实验手册，记录目录结构、安装方式、运行命令、实验产物和复现实验流程。
- `Plan.md` 保留精简路线图，用来提醒每个阶段要学什么、先做什么、做到什么程度。

## 硬件定位

当前机器：

```text
32GB RAM + i5-12400 + RTX 4060 8GB
```

适合做：

```text
小模型本地推理
Qwen2.5-0.5B / 1.5B LoRA 微调
KV Cache / Prefill / Decode 性能实验
FP16 / INT4 量化对比
vLLM 小模型服务化
Mini GPT 从零训练
```

不适合做：

```text
从零训练大模型
长时间大规模 7B 全量微调
重型 TensorRT-LLM 工程化调优
```

## 当前项目模块

```text
inference/        数据准备
finetune/         Qwen LoRA 微调、adapter 测试、merge 测试
cache/            KV Cache 开关对比
prefill_decode/   Prefill/Decode 显式拆分和 prompt 长度对比
quantization/     FP16 vs INT4、GGUF 学习记录
serving/          vLLM OpenAI-compatible 服务和并发 benchmark
mini_gpt/         从零实现字符级 GPT
learn_html/       学习笔记
```

## 推荐推进顺序

### 1. 环境与数据

```bash
pip install -r requirements.txt
python inference/datasource.py
```

目标：确认 Python 环境、依赖安装、Hugging Face 数据下载都正常。

### 2. Base Model 验证

```bash
python finetune/test_base_qwen.py --question "什么是 KV Cache？" --greedy
```

目标：确认 Qwen2.5-0.5B 能在本地加载并完成一次问答。

### 3. LoRA 微调链路

```bash
python finetune/train_lora.py
python finetune/test_lora_adapter.py --greedy
python finetune/merge_lora.py
python finetune/test_merged_lora.py --greedy
```

目标：理解 base model、LoRA adapter、merged model 的关系。

### 4. KV Cache

```bash
python cache/kv_cache_compare.py --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 128 --runs 3
```

目标：用实际指标解释为什么开启 KV Cache 后 decode 更快。

### 5. Prefill / Decode

```bash
python prefill_decode/prefill_decode_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --json-out outputs/prefill_decode_once.json
python prefill_decode/compare_prompt_lengths.py --model Qwen/Qwen2.5-0.5B-Instruct --json-out outputs/prompt_length_compare.json
```

目标：区分 TTFT、prefill latency、decode latency、tokens/s。

### 6. 量化

```bash
python quantization/int4_compare.py --model Qwen/Qwen2.5-0.5B-Instruct --mode both --json-output quantization/int4_compare_result_0.5B.json
```

目标：理解 FP16 和 INT4 在显存、速度和质量上的取舍。

### 7. vLLM 服务化

在 WSL2/Linux 中运行：

```bash
python serving/vllm_server.py --model Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.8 --max-model-len 4096 --hf-endpoint https://hf-mirror.com --enforce-eager
python serving/client.py --model Qwen/Qwen2.5-1.5B-Instruct --stream
python serving/benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --concurrency 1,4,8 --requests-per-level 8 --stream
```

目标：理解 OpenAI-compatible API、PagedAttention、Continuous Batching 和并发吞吐。

### 8. Mini GPT

```bash
python mini_gpt/train.py --steps 20 --eval-interval 10 --eval-iters 2 --batch-size 8
python mini_gpt/train.py --steps 3000 --amp
python mini_gpt/generate.py --checkpoint outputs/mini_gpt_shakespeare.pt --prompt "To be"
```

目标：从代码层理解 attention、causal mask、loss、反向传播和自回归生成。

## 阶段完成标准

```text
能跑通：脚本可以执行，产物能生成。
能观察：知道该看哪些指标和日志。
能解释：可以用自己的话说明现象背后的原因。
能复现：README.md 里的命令可以重新跑出结果。
```

详细命令、目录说明和实验产物以 `README.md` 为准。
