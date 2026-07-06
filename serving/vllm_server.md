# vLLM 服务化部署实验

目标：用 OpenAI 兼容接口部署一个小模型，理解 PagedAttention、KV Cache 管理和 Continuous Batching。

## 启动服务

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16 \
  --port 8000
```

8GB 显存建议先使用 0.5B 模型。如果显存不足，降低模型规模或减少最大上下文长度。

## 调用服务

```bash
python serving/openai_client.py \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

## 观察指标

- 首 token 延迟：请求发出到第一次返回 token 的时间。
- 吞吐：并发请求下每秒生成 token 数。
- 显存：模型权重、KV Cache、batch size 对显存的影响。
- 并发：多个请求同时进入时，Continuous Batching 如何提升 GPU 利用率。
