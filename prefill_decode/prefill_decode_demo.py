from __future__ import annotations

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
    model: str
    input_tokens: int
    generated_tokens: int
    prefill_seconds: float
    ttft_seconds: float
    decode_seconds: float
    decode_tokens: int
    avg_decode_latency_ms: float
    decode_tokens_per_second: float
    total_latency_seconds: float
    end_to_end_tokens_per_second: float
    peak_cuda_memory_gb: float | None


def build_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
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
    return next(model.parameters()).device


def sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def normalize_eos_ids(tokenizer: Any) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return set(eos_token_id)


def pick_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
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
    ensure_runtime_dependencies()

    text = build_prompt(tokenizer, prompt, use_chat_template)
    inputs = tokenizer(text, return_tensors="pt").to(model_input_device(model))
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    eos_ids = normalize_eos_ids(tokenizer)

    clear_cuda_memory()

    # Prefill: one full forward pass over the prompt. This builds the KV cache.
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
    next_token = pick_next_token(outputs.logits[:, -1, :], temperature, top_p)
    generated_ids: list[int] = [int(next_token.item())]

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, device=attention_mask.device)],
            dim=-1,
        )

    decode_latencies: list[float] = []
    should_stop = bool(eos_ids and generated_ids[-1] in eos_ids)

    # Decode: feed exactly one token each step, reusing the KV cache from prefill.
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
        payload = {
            "prompt": args.prompt,
            "generated_text": output_text,
            "metrics": asdict(metrics),
        }
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved_json={args.json_out}")


if __name__ == "__main__":
    main()
