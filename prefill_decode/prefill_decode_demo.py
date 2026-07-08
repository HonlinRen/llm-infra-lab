from __future__ import annotations

"""显式拆解 LLM 推理中的 Prefill 和 Decode 阶段。

这个脚本的目的不是追求最高性能，而是把 Transformers 默认封装在
`model.generate()` 里的流程拆开，便于观察：

1. Prefill：一次性处理完整 prompt，建立 KV Cache，并得到第一个新 token。
2. Decode：后续每一步只输入上一个 token，复用并更新 KV Cache。

这样可以单独统计 TTFT、prefill 耗时、decode 单 token 延迟、decode 吞吐
和峰值显存，帮助理解为什么长 prompt 会主要拉高首 token 时间。
"""

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

torch = None
AutoModelForCausalLM = None
AutoTokenizer = None


@dataclass
class PrefillDecodeMetrics:
    """一次 prefill/decode 拆解实验的指标结果。

    注意这里的 `generated_tokens` 和 `decode_tokens` 不是同一个概念：
    - `generated_tokens`：最终生成的新 token 总数。
    - `decode_tokens`：进入逐 token decode 循环的 token 数。

    第一个新 token 是 prefill 后直接从最后一个位置的 logits 里选出来的，
    因此当 `max_new_tokens=128` 时，常见结果是 generated_tokens=128，
    decode_tokens=127。
    """

    model: str
    # prompt 经过 tokenizer 和 chat template 后的输入 token 数。
    input_tokens: int
    # 最终新生成 token 总数，包含 prefill 阶段选出的第一个 token。
    generated_tokens: int
    # 完整 prompt 的一次 forward 耗时。实验里它也近似等于 TTFT。
    prefill_seconds: float
    # Time To First Token。这里用 prefill_seconds 表示首 token 时间。
    ttft_seconds: float
    # 逐 token decode 循环的总耗时，不包含 prefill。
    decode_seconds: float
    # decode 循环实际跑了多少步。
    decode_tokens: int
    # decode 阶段平均每个 token 的延迟，单位毫秒。
    avg_decode_latency_ms: float
    # decode 阶段吞吐，不包含 prefill。
    decode_tokens_per_second: float
    # prefill + decode 的总耗时。
    total_latency_seconds: float
    # 端到端吞吐，包含 prefill 成本。
    end_to_end_tokens_per_second: float
    # CUDA 峰值显存；CPU 运行时为 None。
    peak_cuda_memory_gb: float | None


def build_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    """按模型习惯构造 prompt 文本。

    Instruct 模型通常需要 chat template，例如 Qwen 会把普通用户问题包装成
    system/user/assistant 格式。这里默认启用 template，让实验更接近真实聊天推理。
    如果某个 tokenizer 不支持 chat template，就回退到原始 prompt。
    """

    if not use_chat_template:
        return prompt

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


def ensure_runtime_dependencies() -> None:
    """延迟导入 torch 和 transformers。

    这样做有两个好处：
    1. 用户即使用没有安装 torch 的全局 Python 运行 `--help`，也能看到参数说明。
    2. 真正执行模型推理时才加载重依赖，错误信息更清晰。
    """

    global AutoModelForCausalLM, AutoTokenizer, torch

    if torch is not None:
        return

    try:
        import torch as torch_module
        from transformers import AutoModelForCausalLM as model_cls
        from transformers import AutoTokenizer as tokenizer_cls
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing runtime dependency. Install project requirements first, for example: "
            "pip install -r requirements.txt"
        ) from exc

    torch = torch_module
    AutoModelForCausalLM = model_cls
    AutoTokenizer = tokenizer_cls


def model_input_device(model: torch.nn.Module) -> torch.device:
    """获取模型参数所在设备，用来把输入 tensor 移到同一设备。"""

    return next(model.parameters()).device


def sync_if_cuda() -> None:
    """CUDA 操作默认异步，计时前后同步才能得到真实 GPU 耗时。"""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cuda_memory() -> None:
    """清理 Python 和 CUDA 缓存，并重置峰值显存统计。

    这不会卸载模型，只是尽量减少上一次实验留下的显存统计干扰。
    """

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def normalize_eos_ids(tokenizer: Any) -> set[int]:
    """把 tokenizer 的 EOS token id 统一成 set，方便后面判断是否停止。"""

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return set(eos_token_id)


def pick_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """从最后一位 logits 中选择下一个 token。

    - temperature <= 0：贪心解码，直接取概率最大的 token，结果稳定可复现。
    - temperature > 0：按概率采样，可选 top-p 截断，适合观察随机生成。

    性能实验默认使用贪心解码，避免采样随机性影响输出长度和耗时。
    """

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return torch.gather(sorted_indices, dim=-1, index=sampled)

    return torch.multinomial(probs, num_samples=1)


def run_prefill_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool,
    temperature: float,
    top_p: float,
) -> tuple[str, PrefillDecodeMetrics]:
    """执行一次显式 prefill + decode 推理，并返回生成文本和指标。

    这个函数是整份实验的核心：
    1. tokenization：把 prompt 转成 input_ids。
    2. prefill：完整 prompt 一次 forward，拿到 past_key_values。
    3. first token：从 prefill 的最后一位 logits 里选出第一个新 token。
    4. decode loop：每步只喂上一个 token，同时传入 past_key_values。
    5. metrics：拆开统计 prefill 和 decode 指标。
    """

    ensure_runtime_dependencies()

    text = build_prompt(tokenizer, prompt, use_chat_template)
    inputs = tokenizer(text, return_tensors="pt").to(model_input_device(model))
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    eos_ids = normalize_eos_ids(tokenizer)

    clear_cuda_memory()

    # =========================
    # 1. Prefill 阶段
    # =========================
    # 一次性把完整 prompt 喂给模型。模型会为 prompt 中所有 token 计算每层
    # attention 的 Key/Value，并通过 `past_key_values` 返回。这个缓存就是
    # 后续 decode 阶段能避免重复计算历史上下文的关键。
    sync_if_cuda()
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )
    sync_if_cuda()
    prefill_seconds = time.perf_counter() - prefill_start

    past_key_values = outputs.past_key_values
    # prefill forward 的 logits 覆盖了 prompt 中每个位置。最后一个位置的 logits
    # 表示“基于完整 prompt，下一 token 是什么”，所以第一个新 token 来自这里。
    next_token = pick_next_token(outputs.logits[:, -1, :], temperature, top_p)
    generated_ids: list[int] = [int(next_token.item())]

    if attention_mask is not None:
        # 第一个新 token 已经被加入 generated_ids。后续 decode 时 attention_mask
        # 也要同步扩展一位，否则部分模型会认为新 token 不可见。
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, device=attention_mask.device)],
            dim=-1,
        )

    decode_latencies: list[float] = []
    should_stop = bool(eos_ids and generated_ids[-1] in eos_ids)

    # =========================
    # 2. Decode 阶段
    # =========================
    # 从第二个新 token 开始，每一步只输入“上一步生成的 token”。
    # `past_key_values` 里已经有 prompt 和历史生成 token 的 K/V，所以模型不需要
    # 重新计算完整上下文，这就是 KV Cache 加速 decode 的本质。
    while len(generated_ids) < max_new_tokens and not should_stop:
        decode_inputs: dict[str, Any] = {
            "input_ids": next_token,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
        }
        if attention_mask is not None:
            decode_inputs["attention_mask"] = attention_mask

        sync_if_cuda()
        decode_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**decode_inputs)
        sync_if_cuda()
        decode_latencies.append(time.perf_counter() - decode_start)

        # outputs.past_key_values 已经追加了当前输入 token 的 K/V。
        # 下一轮 decode 会继续复用这个更新后的缓存。
        past_key_values = outputs.past_key_values
        next_token = pick_next_token(outputs.logits[:, -1, :], temperature, top_p)
        token_id = int(next_token.item())
        generated_ids.append(token_id)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, device=attention_mask.device)],
                dim=-1,
            )

        should_stop = bool(eos_ids and token_id in eos_ids)

    decode_seconds = sum(decode_latencies)
    decode_tokens = len(decode_latencies)
    generated_tokens = len(generated_ids)
    # 这里的 total_latency_seconds 是拆解后的 prefill + decode 计时之和。
    # 不包含模型加载、tokenizer 加载和 prompt 构造耗时。
    total_latency_seconds = prefill_seconds + decode_seconds
    peak_cuda_memory_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    )

    metrics = PrefillDecodeMetrics(
        model=model_name,
        input_tokens=input_ids.shape[-1],
        generated_tokens=generated_tokens,
        prefill_seconds=prefill_seconds,
        ttft_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        decode_tokens=decode_tokens,
        avg_decode_latency_ms=(decode_seconds / decode_tokens * 1000 if decode_tokens else 0.0),
        decode_tokens_per_second=(decode_tokens / decode_seconds if decode_seconds else 0.0),
        total_latency_seconds=total_latency_seconds,
        end_to_end_tokens_per_second=(
            generated_tokens / total_latency_seconds if total_latency_seconds else 0.0
        ),
        peak_cuda_memory_gb=peak_cuda_memory_gb,
    )

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text, metrics


def load_model_and_tokenizer(model_name: str) -> tuple[Any, torch.nn.Module]:
    """加载 tokenizer 和 causal LM 模型。

    device_map="auto" 会让 Transformers/Accelerate 自动选择设备。
    CUDA 可用时使用 float16，CPU 时使用 float32，方便在不同机器上运行。
    """

    ensure_runtime_dependencies()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def print_metrics(metrics: PrefillDecodeMetrics) -> None:
    """用稳定的 key=value 格式打印指标，方便复制到实验笔记。"""

    peak_memory = (
        f"{metrics.peak_cuda_memory_gb:.2f}"
        if metrics.peak_cuda_memory_gb is not None
        else "N/A"
    )

    print("\n--- prefill/decode metrics ---")
    print(f"model={metrics.model}")
    print(f"input_tokens={metrics.input_tokens}")
    print(f"generated_tokens={metrics.generated_tokens}")
    print(f"prefill_seconds={metrics.prefill_seconds:.4f}")
    print(f"ttft_seconds={metrics.ttft_seconds:.4f}")
    print(f"decode_seconds={metrics.decode_seconds:.4f}")
    print(f"decode_tokens={metrics.decode_tokens}")
    print(f"avg_decode_latency_ms={metrics.avg_decode_latency_ms:.2f}")
    print(f"decode_tokens_per_second={metrics.decode_tokens_per_second:.2f}")
    print(f"total_latency_seconds={metrics.total_latency_seconds:.4f}")
    print(f"end_to_end_tokens_per_second={metrics.end_to_end_tokens_per_second:.2f}")
    print(f"peak_cuda_memory_gb={peak_memory}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数，并做基础合法性校验。"""

    parser = argparse.ArgumentParser(
        description="Run LLM inference as explicit prefill and one-token decode steps."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--prompt",
        default="Explain KV Cache, prefill, and decode in simple terms.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use the raw prompt instead of tokenizer.apply_chat_template.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 means greedy decoding. Use values like 0.7 to sample.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save metrics and generated text as JSON.",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    return args


def main() -> None:
    """命令行入口：加载模型、运行拆解实验、打印并可选保存 JSON。"""

    args = parse_args()
    tokenizer, model = load_model_and_tokenizer(args.model)

    output_text, metrics = run_prefill_decode(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=not args.no_chat_template,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(output_text)
    print_metrics(metrics)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        # JSON 里同时保存 prompt、输出文本和指标，方便后续做实验复盘或画图。
        payload = {
            "prompt": args.prompt,
            "generated_text": output_text,
            "metrics": asdict(metrics),
        }
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved_json={args.json_out}")


if __name__ == "__main__":
    main()
