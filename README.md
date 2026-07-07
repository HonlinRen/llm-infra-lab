# llm-infra-lab

一个面向大模型基础设施学习的实验项目。项目目标不是从零训练大模型，而是用 RTX 4060 8GB 这类消费级显卡，把 LLM 推理、KV Cache、Prefill/Decode、LoRA 微调、INT4 量化、vLLM 服务化和 Mini GPT 原理实现逐个跑通。

## 环境建议

本项目默认硬件假设：

```text
CPU: i5-12400
RAM: 32GB
GPU: RTX 4060 8GB
OS: Windows + PowerShell；vLLM 建议使用 WSL2/Linux
```

推荐先创建虚拟环境：

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

注意：`requirements.txt` 里包含 `vllm`，但官方 vLLM 不支持原生 Windows。Windows 本地实验可以先安装除 vLLM 外的依赖；vLLM 服务化实验建议在 WSL2 Ubuntu 中单独安装。

## 当前目录结构

```text
llm-infra-lab/
├── README.md                         # 项目总入口：实验顺序、运行命令、产物说明
├── Plan.md                           # 精简学习路线与模块地图
├── requirements.txt                  # Python 依赖
│
├── inference/
│   └── datasource.py                 # 下载并缓存 alpaca-gpt4-data-zh 的 500 条样本
│
├── finetune/
│   ├── train_lora.py                 # Qwen2.5-0.5B LoRA 微调
│   ├── merge_lora.py                 # 将 LoRA adapter 合并为独立模型
│   ├── test_base_qwen.py             # 测试原始 base model
│   ├── test_lora_adapter.py          # 测试 base model + LoRA adapter
│   └── test_merged_lora.py           # 测试合并后的模型
│
├── cache/
│   ├── README.md                     # KV Cache 实验说明
│   └── kv_cache_compare.py           # use_cache=True/False 性能对比
│
├── prefill_decode/
│   ├── README.md                     # Prefill/Decode 拆解说明
│   ├── prefill_decode_demo.py        # 显式拆分 prefill 和逐 token decode
│   └── compare_prompt_lengths.py     # 对比短/中/长 prompt 下的 TTFT 和 decode 指标
│
├── quantization/
│   ├── int4_compare.py               # FP16 vs INT4 显存、加载、速度、质量对比
│   ├── int4_infer.md                 # INT4 量化实验说明
│   ├── gguf_test.md                  # GGUF / llama.cpp 路线说明
│   └── *_result*.json                # 量化实验输出结果
│
├── serving/
│   ├── README.md                     # vLLM 服务化实验说明
│   ├── vllm_server.py                # 启动 vLLM OpenAI-compatible server
│   ├── client.py                     # 单请求客户端，支持 stream/TTFT
│   ├── benchmark.py                  # 并发 benchmark，输出 CSV
│   ├── openai_client.py              # 简化版 OpenAI-compatible 客户端
│   └── results.csv                   # benchmark 结果
│
├── mini_gpt/
│   ├── README.md                     # Mini GPT 原理实验说明
│   ├── model.py                      # 从零实现 GPTConfig、Attention、Block、MiniGPT
│   ├── train.py                      # 训练 Tiny Shakespeare 字符级模型
│   └── generate.py                   # 从 checkpoint 生成文本
│
├── learn_html/                       # 各模块学习笔记 HTML
├── data/                             # 数据缓存，运行后生成
├── outputs/                          # 训练 checkpoint，运行后生成
├── models/                           # 本地模型或 ONNX 模型
├── qwen-lora-adapter/                # LoRA adapter，训练后生成
└── qwen-lora-merged/                 # 合并后的 LoRA 模型，运行后生成
```

## 推荐实验顺序

### 1. 准备微调数据

下载中文 Alpaca 数据集前 500 条，并缓存到 `data/alpaca_500`：

```bash
python inference/datasource.py
```

这个步骤会访问 Hugging Face。如果网络慢，可以设置镜像：

```bash
$env:HF_ENDPOINT="https://hf-mirror.com"
python inference/datasource.py
```

### 2. 测试原始 Qwen 模型

先确认模型下载、CUDA、Transformers 都能正常工作：

```bash
python finetune/test_base_qwen.py --question "什么是 KV Cache？" --greedy
```

默认模型是：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

### 3. LoRA 微调

训练 Qwen2.5-0.5B 的 LoRA adapter：

```bash
python finetune/train_lora.py
```

主要产物：

```text
qwen-lora-adapter/
qwen-lora/
data/alpaca_500/
```

测试 LoRA adapter：

```bash
python finetune/test_lora_adapter.py --question "保持健康的三个提示。" --greedy
```

将 adapter 合并为独立模型：

```bash
python finetune/merge_lora.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter ./qwen-lora-adapter --output-dir ./qwen-lora-merged
```

测试合并后的模型：

```bash
python finetune/test_merged_lora.py --model-path ./qwen-lora-merged --question "保持健康的三个提示。" --greedy
```

### 4. KV Cache 对比

对比 `use_cache=True` 和 `use_cache=False`：

```bash
python cache/kv_cache_compare.py --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 128 --runs 3
```

显存紧张时减少生成长度：

```bash
python cache/kv_cache_compare.py --max-new-tokens 64 --runs 2
```

重点观察：

```text
latency_seconds
tokens_per_second
peak_cuda_memory_gb
use_cache=True speedup
```

### 5. Prefill / Decode 拆分

单条 prompt 拆分：

```bash
python prefill_decode/prefill_decode_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 128 --json-out outputs/prefill_decode_once.json
```

比较短、中、长 prompt：

```bash
python prefill_decode/compare_prompt_lengths.py --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 128 --json-out outputs/prompt_length_compare.json
```

重点观察：

```text
input_tokens
prefill_seconds
ttft_seconds
avg_decode_latency_ms
decode_tokens_per_second
total_latency_seconds
```

### 6. FP16 vs INT4 量化

对比 FP16 和 INT4：

```bash
python quantization/int4_compare.py --model Qwen/Qwen2.5-0.5B-Instruct --mode both --max-new-tokens 128 --json-output quantization/int4_compare_result_0.5B.json
```

如果只想跑 INT4：

```bash
python quantization/int4_compare.py --model Qwen/Qwen2.5-0.5B-Instruct --mode int4 --max-new-tokens 128
```

重点观察：

```text
load_seconds
cuda_allocated_after_load_gb
cuda_peak_allocated_gb
generation_seconds
tokens_per_second
answer
```

### 7. vLLM 服务化实验

官方 vLLM 建议在 WSL2/Linux 中运行。进入 WSL 后：

```bash
cd /mnt/c/Users/OnlyRen/Project/LLM/llm-infra-lab
uv venv .venv-wsl --python 3.12 --seed
source .venv-wsl/bin/activate
uv pip install vllm openai
```

启动服务：

```bash
python serving/vllm_server.py --model Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.8 --max-model-len 4096 --hf-endpoint https://hf-mirror.com --enforce-eager
```

单请求验证：

```bash
python serving/client.py --model Qwen/Qwen2.5-1.5B-Instruct --stream
```

并发 benchmark：

```bash
python serving/benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --concurrency 1,4,8 --requests-per-level 8 --stream
```

结果会写入：

```text
serving/results.csv
```

重点观察：

```text
avg_latency_s
avg_ttft_s
p50_latency_s
p95_latency_s
req_per_s
output_tokens_per_s
```

### 8. Mini GPT 从零训练

快速 smoke test：

```bash
python mini_gpt/train.py --steps 20 --eval-interval 10 --eval-iters 2 --batch-size 8
```

正式一点的训练：

```bash
python mini_gpt/train.py --steps 3000 --amp
```

生成文本：

```bash
python mini_gpt/generate.py --checkpoint outputs/mini_gpt_shakespeare.pt --prompt "To be" --max-new-tokens 400
```

重点观察：

```text
train_loss
val_loss
attention mask
next-token prediction
autoregressive generation
```

## 实验产物建议

这些目录通常是运行后生成的产物，不建议提交大文件：

```text
data/
outputs/
models/
qwen-lora/
qwen-lora-adapter/
qwen-lora-merged/
serving/*.log
serving/*.pid
```

小型结果文件可以保留，例如：

```text
serving/results.csv
quantization/*result*.json
outputs/*.json
```

## 模块学习目标

```text
finetune：理解 LoRA、adapter、base model、merge_and_unload。
cache：理解 KV Cache 为什么减少 decode 阶段重复计算。
prefill_decode：理解 TTFT、Prefill、Decode、TPOT、tokens/s。
quantization：理解 FP16/INT4 的显存、速度、质量取舍。
serving：理解 OpenAI-compatible API、vLLM、PagedAttention、Continuous Batching。
mini_gpt：理解 embedding、causal self-attention、causal mask、cross entropy、反向传播。
```

## 面试表达模板

我做过一个学习型 LLM Infra Lab，不是为了训练大模型，而是为了把推理、微调和部署链路跑通。在 RTX 4060 8GB 本地环境下，我使用 Qwen2.5-0.5B 做了 LoRA 微调，测试了 base model、LoRA adapter 和 merge 后模型的差异；同时对比了 KV Cache 开关、拆分了 prefill/decode 指标，记录 TTFT、decode tokens/s、总延迟和显存占用。

我还做了 FP16/INT4 量化对比，观察加载时间、显存峰值、生成速度和回答质量的变化；在 WSL2 中用 vLLM 启动 OpenAI 兼容服务，并通过并发 benchmark 观察 continuous batching 对吞吐的影响。为了理解底层训练机制，我也从零实现了一个字符级 Mini GPT，用 Tiny Shakespeare 观察 loss 下降和自回归生成流程。
