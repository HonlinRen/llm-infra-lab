# vLLM Serving Experiment

This directory contains a small vLLM serving experiment for validating:

- OpenAI-compatible API serving
- concurrent request throughput
- continuous batching behavior
- KV cache / PagedAttention memory observations

## Files

| File | Purpose |
| --- | --- |
| `vllm_server.py` | Starts `vllm serve` with repeatable defaults. |
| `client.py` | Sends one chat completion request to the OpenAI-compatible endpoint. |
| `benchmark.py` | Sends concurrent requests and writes latency/throughput results. |
| `results.csv` | Benchmark output file. |
| `vllm_server.md` | Experiment design notes. |

## Setup

Run the commands below inside WSL2 Ubuntu or another Linux environment. Official vLLM does not support native Windows PowerShell.

Check that WSL can see the NVIDIA GPU and that the driver supports CUDA 13:

```bash
nvidia-smi
```

For this vLLM install, PyTorch uses CUDA 13. The NVIDIA driver should report CUDA 13.x support. If it reports CUDA 12.x, update the Windows NVIDIA driver first, then restart WSL with `wsl --shutdown`.

```bash
cd /mnt/c/Users/OnlyRen/Project/LLM/llm-infra-lab
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv-wsl --python 3.12 --seed
source .venv-wsl/bin/activate
uv pip install vllm openai
```

Start the vLLM server:

```bash
python serving/vllm_server.py --model Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.8 --max-model-len 4096 --hf-endpoint https://hf-mirror.com --enforce-eager
```

This is the recommended command for an RTX 4060 8GB machine. It keeps vLLM's GPU memory target at 80% and limits context length so KV cache allocation is easier to fit.

The wrapper also sets `HF_HUB_DISABLE_XET=1` so model weights download through the regular Hugging Face path instead of Xet/CAS.
It sets `VLLM_USE_FLASHINFER_SAMPLER=0` to avoid FlashInfer JIT compilation issues in WSL.
It also prepends the virtual environment's CUDA `bin` directory to `PATH` so PyTorch/vLLM can find `nvcc` during startup profiling.
`--enforce-eager` avoids Torch compile/CUDA graph startup issues that are common in WSL learning environments.

Keep this server process running while using the client and benchmark commands below.

## API Check

```bash
python serving/client.py --model Qwen/Qwen2.5-1.5B-Instruct --stream
```

## Concurrency Benchmark

```bash
python serving/benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --concurrency 1,4,8 --requests-per-level 8 --stream
```

The benchmark records:

- average latency
- average TTFT when `--stream` is enabled
- p50 / p95 latency
- request throughput
- output token throughput
- failures and first error, if any

Results are written to `serving/results.csv`.

## GPU Memory Observation

Watch GPU memory in another terminal while the benchmark is running:

```bash
nvidia-smi
```

The expected learning outcome is not a surprising single-request speedup. The useful signal is that higher concurrency can increase aggregate throughput because vLLM keeps the GPU busier with continuous batching and manages KV cache blocks efficiently through PagedAttention.
