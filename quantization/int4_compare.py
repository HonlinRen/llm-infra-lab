import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# 默认测试 prompt。
# 这里选择 KV Cache 作为问题，是因为它能同时检验模型是否理解 LLM 推理中的
# 生成速度、显存占用和工程 tradeoff。FP16 和 INT4 使用同一个 prompt，
# 才能比较生成速度和回答质量。
DEFAULT_PROMPT = (
    "Explain what KV Cache is in LLM inference. Include why it improves "
    "generation speed and what tradeoff it introduces."
)


@dataclass
class BenchmarkResult:
    """单次 benchmark 的结构化结果。

    用 dataclass 的好处：
    1. 字段集中，便于确认实验记录了哪些指标。
    2. 可以通过 asdict() 很方便地保存成 JSON。
    3. 后续如果要追加 TTFT、batch size、GPU 名称等指标，也容易扩展。
    """

    # 当前实验模式：fp16 或 int4。
    mode: str
    # 模型名称或本地模型路径。
    model_name: str
    # AutoModelForCausalLM.from_pretrained 的耗时，单位秒。
    load_seconds: float
    # 模型加载后，PyTorch 当前已经实际分配的 CUDA 显存。
    cuda_allocated_after_load_gb: Optional[float]
    # 模型加载后，PyTorch CUDA caching allocator 预留的显存。
    # reserved 通常大于 allocated，因为 PyTorch 会缓存显存块以便复用。
    cuda_reserved_after_load_gb: Optional[float]
    # 本轮测试过程中的峰值 allocated 显存。
    cuda_peak_allocated_gb: Optional[float]
    # prompt 编码后的输入 token 数。
    input_tokens: int
    # generate 新生成的 token 数，不包含 prompt token。
    generated_tokens: int
    # model.generate 的总耗时，单位秒。
    generation_seconds: float
    # 生成吞吐：generated_tokens / generation_seconds。
    tokens_per_second: float
    # 模型生成的回答文本，用于人工比较质量。
    answer: str


def build_prompt(tokenizer, prompt: str) -> str:
    """把普通用户问题包装成 chat model 期望的 prompt 格式。

    Instruct/Chat 模型通常有自己的 chat template，例如 Qwen、SmolLM2、
    Llama chat 模型都会在 tokenizer 中提供 apply_chat_template。
    如果直接喂裸 prompt，模型可能少了 system/user/assistant 等特殊标记，
    输出质量和真实聊天推理不一致。
    """

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def cuda_gb(value: int) -> float:
    """把 CUDA 显存字节数转换为 GB，便于打印和写入实验报告。"""

    return value / 1024**3


def cleanup_cuda() -> None:
    """尽量清理上一轮模型占用，降低 FP16 和 INT4 互相影响。

    注意：
    - gc.collect() 触发 Python 对象回收，让已经 del 的模型尽快释放引用。
    - torch.cuda.empty_cache() 释放 PyTorch caching allocator 中未使用的缓存块。
    - reset_peak_memory_stats() 重置峰值显存统计，否则下一轮会继承上一轮峰值。

    这不能保证系统级显存完全回到初始状态，但足以让两种模式的测量更公平。
    """

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_model(model_name: str, mode: str):
    """根据模式加载 FP16 或 INT4 模型。

    FP16 模式：
    - 有 CUDA 时使用 torch.float16，方便使用 GPU 半精度推理。
    - 没有 CUDA 时退回 float32，避免 CPU 上很多 FP16 op 支持不完整。

    INT4 模式：
    - 使用 BitsAndBytesConfig(load_in_4bit=True) 启用 4bit 权重量化。
    - bnb_4bit_quant_type="nf4" 是 LLM 量化常用格式，通常比普通 int4
      更适合近似正态分布的权重。
    - bnb_4bit_compute_dtype=torch.float16 表示权重低比特存储，但计算中
      常以 FP16 参与矩阵乘法。
    """

    if mode == "fp16":
        kwargs = {
            # transformers 新版本提示 torch_dtype 已逐步迁移到 dtype，
            # 但当前版本仍兼容；这里保持项目已有写法。
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            # device_map="auto" 让 accelerate 自动把模型放到可用设备。
            "device_map": "auto",
            # 部分模型仓库需要自定义代码；常见开源 LLM 加载时保留该参数更稳。
            "trust_remote_code": True,
        }
    elif mode == "int4":
        if not torch.cuda.is_available():
            raise RuntimeError("INT4 bitsandbytes inference requires CUDA.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs = {
            "quantization_config": quant_config,
            "device_map": "auto",
            "trust_remote_code": True,
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def run_once(args, tokenizer, mode: str) -> BenchmarkResult:
    """运行一次完整 benchmark。

    一轮包含：
    1. 清理 CUDA 状态。
    2. 加载模型并记录加载耗时。
    3. 记录加载后显存。
    4. 使用同一个 prompt 生成文本。
    5. 统计生成耗时、输出 token 数和 tokens/s。
    6. 保存回答文本，用于人工比较回答质量。
    7. 删除模型并再次清理 CUDA，方便下一轮测试。
    """

    cleanup_cuda()

    # perf_counter 精度高，适合测量短时间间隔。
    load_start = time.perf_counter()
    model = load_model(args.model, mode)
    load_seconds = time.perf_counter() - load_start

    if torch.cuda.is_available():
        # allocated 是当前实际被 tensor 占用的显存。
        allocated_after_load = cuda_gb(torch.cuda.memory_allocated())
        # reserved 是 PyTorch allocator 向 CUDA 申请并保留的显存。
        reserved_after_load = cuda_gb(torch.cuda.memory_reserved())
    else:
        allocated_after_load = None
        reserved_after_load = None

    text = build_prompt(tokenizer, args.prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 这里把采样参数也放进去，便于通过命令行切换确定性/随机生成。
    # 默认 do_sample=False，便于 FP16 与 INT4 对比时减少随机性干扰。
    gen_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "use_cache": True,
    }
    if not args.do_sample:
        # transformers 在 do_sample=False 时传 temperature 可能触发 warning，
        # 因此确定性生成时移除 temperature。
        gen_kwargs.pop("temperature")

    gen_start = time.perf_counter()
    # inference_mode 比 no_grad 更彻底，会关闭 autograd 相关开销。
    with torch.inference_mode():
        outputs = model.generate(**gen_kwargs)
    generation_seconds = time.perf_counter() - gen_start

    input_tokens = inputs["input_ids"].shape[-1]
    # outputs 包含 prompt + 新生成内容，所以要减掉输入长度。
    generated_tokens = outputs.shape[-1] - input_tokens
    # 只 decode 新生成部分，避免回答质量对比里重复打印 prompt。
    answer = tokenizer.decode(
        outputs[0][input_tokens:],
        skip_special_tokens=True,
    ).strip()

    if torch.cuda.is_available():
        peak_allocated = cuda_gb(torch.cuda.max_memory_allocated())
    else:
        peak_allocated = None

    result = BenchmarkResult(
        mode=mode,
        model_name=args.model,
        load_seconds=load_seconds,
        cuda_allocated_after_load_gb=allocated_after_load,
        cuda_reserved_after_load_gb=reserved_after_load,
        cuda_peak_allocated_gb=peak_allocated,
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        generation_seconds=generation_seconds,
        tokens_per_second=generated_tokens / max(generation_seconds, 1e-9),
        answer=answer,
    )

    # 删除模型对象后清理 CUDA，避免 both 模式下 FP16 的显存影响 INT4。
    del model
    cleanup_cuda()
    return result


def format_number(value: Optional[float], digits: int = 2) -> str:
    """格式化可选浮点数。

    CPU 环境没有 CUDA 显存指标，此时值为 None，打印 N/A。
    """

    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def print_markdown_table(results: list[BenchmarkResult]) -> None:
    """把核心指标打印成 Markdown 表格，方便复制进笔记或 README。"""

    print("\n## Metrics")
    print(
        "| mode | load_s | mem_after_load_gb | reserved_after_load_gb | "
        "peak_mem_gb | input_tokens | output_tokens | gen_s | tokens/s |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for item in results:
        print(
            f"| {item.mode} "
            f"| {item.load_seconds:.2f} "
            f"| {format_number(item.cuda_allocated_after_load_gb)} "
            f"| {format_number(item.cuda_reserved_after_load_gb)} "
            f"| {format_number(item.cuda_peak_allocated_gb)} "
            f"| {item.input_tokens} "
            f"| {item.generated_tokens} "
            f"| {item.generation_seconds:.2f} "
            f"| {item.tokens_per_second:.2f} |"
        )


def print_quality_section(prompt: str, results: list[BenchmarkResult]) -> None:
    """打印 FP16/INT4 的回答样例，供人工评估质量。

    回答质量无法只靠 tokens/s 判断。需要人工看：
    - 是否事实准确。
    - 是否完整覆盖 prompt 要求。
    - 是否有乱码、重复、格式崩坏。
    - 量化后是否更容易出现概念混淆或遗漏。
    """

    print("\n## Answer Quality Samples")
    print("Use the same prompt below to compare factuality, completeness, and fluency.")
    print(f"\nPrompt:\n{prompt}\n")
    for item in results:
        print(f"### {item.mode.upper()} answer")
        print(item.answer)
        print()


def parse_args():
    """解析命令行参数。

    常用命令：
    python quantization/int4_compare.py --mode both
    python quantization/int4_compare.py --mode int4
    python quantization/int4_compare.py --model ./models/SmolLM2-1.7B-Instruct
    """

    parser = argparse.ArgumentParser(
        description="Compare FP16 and INT4 inference memory, load time, speed, and output quality."
    )
    # 默认模型来自原始 int4_infer.md。下载不稳定时可以换成本地路径或小模型。
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--mode",
        choices=["both", "fp16", "int4"],
        default="both",
        help="Run both modes or only one mode.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. By default generation is deterministic for easier comparison.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--json-output",
        help="Optional path to save raw benchmark results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # both 模式按 FP16 -> INT4 顺序执行；每轮结束会清理 CUDA。
    modes = ["fp16", "int4"] if args.mode == "both" else [args.mode]

    # tokenizer 两种模式共用即可，因为量化只影响模型权重，不影响分词。
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    results = []
    for mode in modes:
        print(f"\nRunning {mode.upper()} benchmark...")
        results.append(run_once(args, tokenizer, mode))

    print_markdown_table(results)
    print_quality_section(args.prompt, results)

    if args.json_output:
        # ensure_ascii=False 保留中文，便于后续直接阅读 JSON。
        with open(args.json_output, "w", encoding="utf-8") as file:
            json.dump([asdict(item) for item in results], file, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON results to {args.json_output}")


if __name__ == "__main__":
    main()
