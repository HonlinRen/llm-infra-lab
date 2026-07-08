import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import List


def build_command(args: argparse.Namespace) -> List[str]:
    """把 Python 脚本参数转换成最终要执行的 `vllm serve ...` 命令。"""
    # 这里不直接 import vLLM，而是调用 vLLM CLI。
    # 好处是：启动行为和命令行完全一致，也方便把额外参数继续透传给 vLLM。
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

    # 8GB 显存机器通常需要限制最大上下文长度，否则 KV Cache 预留会比较激进。
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    # 某些模型仓库需要执行自定义建模代码时才需要打开；Qwen2.5 通常不需要。
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    # WSL 学习环境里 Torch compile / CUDA Graph 可能遇到编译器或 JIT 兼容问题。
    # enforce eager 会更慢一些，但启动稳定性更好，适合先跑通实验。
    if args.enforce_eager:
        command.append("--enforce-eager")
    # 允许用户在 `--` 后继续追加 vLLM 原生参数，例如：
    # python serving/vllm_server.py -- --generation-config vllm
    if args.extra_args:
        command.extend(args.extra_args)

    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start a vLLM OpenAI-compatible API server."
    )
    # vLLM 对外暴露 OpenAI-compatible API，模型名必须和 client/benchmark 里的 model 一致。
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="half", choices=["auto", "half", "float16", "bfloat16", "float32"])
    # vLLM 会按该比例规划可用显存，剩余空间留给系统、桌面、临时 kernel 等。
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

    # 官方 vLLM 不支持原生 Windows。这个保护可以避免用户在 PowerShell 里看到很长的
    # C++/CUDA 扩展 import 错误，直接提示切到 WSL/Linux。
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

    # 依赖安装成功后，虚拟环境中应该能找到 `vllm` 命令。
    if shutil.which("vllm") is None:
        print(
            "vllm command was not found. Install it first with: pip install vllm",
            file=sys.stderr,
        )
        return 1

    # argparse.REMAINDER 会把分隔符 `--` 本身也收进来，这里手动去掉。
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]

    command = build_command(args)
    env = os.environ.copy()
    # 国内网络访问 Hugging Face 主站可能失败，通过 HF_ENDPOINT 指向镜像站。
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint
    # 禁用 Hugging Face Hub 的 Xet/CAS 下载路径，避免部分镜像/网络环境出现 401。
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    # FlashInfer sampler 在 WSL + pip CUDA 包组合里可能触发 JIT 编译失败。
    # 对学习实验来说，禁用后使用 PyTorch/native sampler 更稳。
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # vLLM / PyTorch 有时需要调用 pip 安装的 CUDA nvcc。
    # uv venv 的 Python 可执行文件可能是链接，这里用 sys.prefix 定位当前虚拟环境。
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

    # 打印最终环境和命令，方便复现实验，也方便从日志中定位是否使用了镜像/稳定性参数。
    print("Starting vLLM server:")
    if args.hf_endpoint:
        print(f"HF_ENDPOINT={args.hf_endpoint}")
    print(f"HF_HUB_DISABLE_XET={env['HF_HUB_DISABLE_XET']}")
    print(f"VLLM_USE_FLASHINFER_SAMPLER={env['VLLM_USE_FLASHINFER_SAMPLER']}")
    if cuda_bin.exists():
        print(f"CUDA_BIN={cuda_bin}")
    print(" ".join(command))
    print()
    # subprocess.call 会阻塞当前进程；vLLM server 正常运行时不会主动退出。
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
