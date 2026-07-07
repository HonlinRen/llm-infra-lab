import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import List


def build_command(args: argparse.Namespace) -> List[str]:
    command = [
        "vllm",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]

    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.extra_args:
        command.extend(args.extra_args)

    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start a vLLM OpenAI-compatible API server."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="half", choices=["auto", "half", "float16", "bfloat16", "float32"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional context length cap, useful on small GPUs.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile/CUDA graph capture. Slower, but more stable on WSL.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Optional Hugging Face endpoint mirror, for example https://hf-mirror.com.",
    )
    parser.add_argument(
        "--allow-windows",
        action="store_true",
        help="Try to launch on Windows anyway, for community-built vLLM wheels.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed directly to vllm serve.",
    )
    args = parser.parse_args()

    if os.name == "nt" and not args.allow_windows:
        print(
            "Official vLLM does not support native Windows. Run this project in WSL2/Linux instead.\n\n"
            "Recommended path:\n"
            "  1. Open Ubuntu/WSL terminal.\n"
            "  2. cd /mnt/c/Users/OnlyRen/Project/LLM/llm-infra-lab\n"
            "  3. Create a Linux virtual environment and install vLLM there.\n"
            "  4. Re-run this command inside WSL.\n\n"
            "If you intentionally installed a community Windows vLLM build, pass --allow-windows.",
            file=sys.stderr,
        )
        return 1

    if shutil.which("vllm") is None:
        print(
            "vllm command was not found. Install it first with: pip install vllm",
            file=sys.stderr,
        )
        return 1

    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]

    command = build_command(args)
    env = os.environ.copy()
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    cuda_bin = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cu13"
        / "bin"
    )
    if cuda_bin.exists():
        env["PATH"] = f"{cuda_bin}{os.pathsep}{env.get('PATH', '')}"

    print("Starting vLLM server:")
    if args.hf_endpoint:
        print(f"HF_ENDPOINT={args.hf_endpoint}")
    print(f"HF_HUB_DISABLE_XET={env['HF_HUB_DISABLE_XET']}")
    print(f"VLLM_USE_FLASHINFER_SAMPLER={env['VLLM_USE_FLASHINFER_SAMPLER']}")
    if cuda_bin.exists():
        print(f"CUDA_BIN={cuda_bin}")
    print(" ".join(command))
    print()
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
