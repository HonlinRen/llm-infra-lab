import argparse
import gc
import statistics
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RunMetrics:
    """保存单次生成实验的性能指标。

    这个 dataclass 只是一个轻量容器，方便后面把每轮实验的结果汇总、求平均、打印。
    """

    # 当前生成是否开启 KV Cache。True 表示复用历史 token 的 Key/Value。
    use_cache: bool
    # prompt 被 tokenizer 编码后的 token 数，也就是 prefill 阶段需要处理的长度。
    input_tokens: int
    # 本次 generate 实际生成的新 token 数。
    generated_tokens: int
    # 从调用 model.generate 到生成结束的总耗时，单位是秒。
    latency_seconds: float
    # 生成吞吐：每秒生成多少个 token，数值越大说明 decode 越快。
    tokens_per_second: float
    # CUDA 峰值显存。如果在 CPU 上运行，则为 None。
    peak_cuda_memory_gb: float | None


def build_prompt(tokenizer, prompt: str) -> str:
    """把普通用户输入转换成模型更熟悉的 chat prompt。

    Instruct / Chat 模型通常在训练时使用固定的对话模板，例如：
    user: ...
    assistant:

    tokenizer.apply_chat_template 会把用户问题包装成模型期望的格式。
    如果某个 tokenizer 不支持 chat template，或者模板应用失败，就直接使用原始 prompt。
    """

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


def clear_cuda_memory() -> None:
    """清理 Python 和 CUDA 显存统计，让每轮实验的显存记录更干净。

    注意：empty_cache 不会释放模型本身占用的显存，它只是把 PyTorch 缓存分配器中
    暂时不用的显存还给 CUDA，便于观察本轮 generate 的额外峰值。
    """

    # 先触发 Python 垃圾回收，尽量清掉上一轮生成产生的临时对象。
    gc.collect()
    if torch.cuda.is_available():
        # 清理 PyTorch CUDA caching allocator 中暂时不用的缓存。
        torch.cuda.empty_cache()
        # 重置峰值显存计数，后面 max_memory_allocated 才能反映本轮实验峰值。
        torch.cuda.reset_peak_memory_stats()


def run_once(model, tokenizer, prompt: str, max_new_tokens: int, use_cache: bool) -> RunMetrics:
    """执行一次 generate，并返回本次生成的性能指标。

    这是整个实验最核心的函数。唯一被刻意切换的变量是 use_cache：
    - use_cache=True：decode 阶段复用历史 token 的 Key/Value，通常更快。
    - use_cache=False：每一步生成都会重复计算历史上下文，通常更慢。
    """

    # 1. 构造模型输入文本。如果模型是 chat/instruct 模型，会优先使用聊天模板。
    text = build_prompt(tokenizer, prompt)
    # 2. tokenizer 把文本变成 input_ids / attention_mask，并移动到模型所在设备。
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 3. 清理显存统计，开始计时。
    clear_cuda_memory()
    start = time.perf_counter()
    with torch.inference_mode():
        # 4. 执行自回归生成。
        #
        # max_new_tokens 控制最多生成多少个新 token。
        # use_cache 是本实验的核心开关：
        #   True  -> 缓存每层 attention 的历史 Key/Value；
        #   False -> 不缓存，每个 decode step 都重新处理完整历史序列。
        # do_sample=False 表示使用确定性生成，减少随机采样对耗时的干扰。
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            do_sample=False,
        )
    if torch.cuda.is_available():
        # CUDA kernel 默认异步执行。同步后再停止计时，才能得到真实 GPU 耗时。
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # outputs 包含 prompt + 新生成 token，所以需要减去输入长度。
    input_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = outputs.shape[-1] - input_tokens
    # 记录本轮 generate 期间的 CUDA 峰值显存。CPU 运行时该指标不可用。
    peak_cuda_memory_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    )

    return RunMetrics(
        use_cache=use_cache,
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        latency_seconds=elapsed,
        tokens_per_second=generated_tokens / elapsed if elapsed > 0 else 0.0,
        peak_cuda_memory_gb=peak_cuda_memory_gb,
    )


def average_metrics(metrics: list[RunMetrics]) -> RunMetrics:
    """把多轮实验结果求平均，降低单次运行抖动带来的影响。"""

    # 每轮实验的 use_cache 和 input_tokens 理论上都一样，直接取第一轮即可。
    first = metrics[0]
    # CPU 运行时 peak_cuda_memory_gb 为 None，因此只对非 None 显存值求平均。
    memory_values = [m.peak_cuda_memory_gb for m in metrics if m.peak_cuda_memory_gb is not None]
    return RunMetrics(
        use_cache=first.use_cache,
        input_tokens=first.input_tokens,
        generated_tokens=round(statistics.mean(m.generated_tokens for m in metrics)),
        latency_seconds=statistics.mean(m.latency_seconds for m in metrics),
        tokens_per_second=statistics.mean(m.tokens_per_second for m in metrics),
        peak_cuda_memory_gb=statistics.mean(memory_values) if memory_values else None,
    )


def print_result(metrics: RunMetrics) -> None:
    """以一行表格风格打印平均指标，方便对比 cache 开关差异。"""

    peak_memory = (
        f"{metrics.peak_cuda_memory_gb:.2f} GB"
        if metrics.peak_cuda_memory_gb is not None
        else "N/A"
    )
    print(
        f"use_cache={str(metrics.use_cache):5} | "
        f"input_tokens={metrics.input_tokens:4} | "
        f"generated_tokens={metrics.generated_tokens:4} | "
        f"latency={metrics.latency_seconds:7.2f}s | "
        f"tokens/s={metrics.tokens_per_second:7.2f} | "
        f"peak_cuda_memory={peak_memory}"
    )


def main() -> None:
    """命令行入口：加载模型，分别测试 use_cache=True 和 use_cache=False。"""

    parser = argparse.ArgumentParser(
        description="Compare Transformers generation with KV Cache enabled and disabled."
    )
    # Hugging Face 模型名或本地模型目录，例如：
    #   Qwen/Qwen2.5-0.5B-Instruct
    #   HuggingFaceTB/SmolLM2-1.7B-Instruct
    #   models/SmolLM2-1.7B-Instruct
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    # 用于生成的提示词。prompt 越长，关闭 KV Cache 时重复计算历史上下文的代价越明显。
    parser.add_argument(
        "--prompt",
        default="用 5 点解释大模型推理中的 KV Cache 是什么，以及为什么它能加速生成。",
    )
    # 生成 token 数。生成越长，decode 步数越多，KV Cache 的收益通常越明显。
    parser.add_argument("--max-new-tokens", type=int, default=256)
    # 每个配置重复运行几轮，最后输出平均值。
    parser.add_argument("--runs", type=int, default=3)
    # 正式计时前先跑几轮短生成，让 CUDA kernel、内存分配等进入较稳定状态。
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    # 加载 tokenizer。trust_remote_code=True 允许部分模型使用仓库中的自定义 tokenizer 逻辑。
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # 加载 causal language model。
    #
    # GPU 上使用 float16 可以降低显存并提高速度；CPU 上使用 float32 兼容性更好。
    # device_map="auto" 会让 Transformers/Accelerate 自动把模型放到可用设备上。
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    # 切换到推理模式，关闭 dropout 等训练行为。
    model.eval()

    # warmup 不参与最终统计，只是为了减少首次运行的初始化开销影响。
    for _ in range(args.warmup):
        run_once(model, tokenizer, args.prompt, min(args.max_new_tokens, 16), use_cache=True)

    # 分别测试开启和关闭 KV Cache。为了公平对比，除了 use_cache 外其余参数保持一致。
    results: dict[bool, list[RunMetrics]] = {}
    for use_cache in (True, False):
        results[use_cache] = [
            run_once(model, tokenizer, args.prompt, args.max_new_tokens, use_cache)
            for _ in range(args.runs)
        ]

    print("\nKV Cache comparison")
    print(f"model={args.model}")
    print(f"runs={args.runs}, max_new_tokens={args.max_new_tokens}\n")

    # 对多轮结果求平均后打印。
    cache_on = average_metrics(results[True])
    cache_off = average_metrics(results[False])
    print_result(cache_on)
    print_result(cache_off)

    # speedup 使用 tokens/s 计算，表示开启 KV Cache 后吞吐提升多少倍。
    speedup = cache_on.tokens_per_second / cache_off.tokens_per_second
    # saved_seconds 使用 latency 计算，表示每轮平均节省多少秒。
    saved_seconds = cache_off.latency_seconds - cache_on.latency_seconds
    print(f"\nuse_cache=True speedup: {speedup:.2f}x")
    print(f"latency saved per run: {saved_seconds:.2f}s")


if __name__ == "__main__":
    main()
