"""Hello, GPU: what did Slurm give me, and how fast is it?

Prints the resources of the current job, then times a large matrix multiplication
on the CPU and on the GPU in three number formats:

  fp32  32-bit floats, the "classic" format
  tf32  fp32 storage, but multiplied on the Tensor Cores with a shorter mantissa
  bf16  16-bit "brain float", the standard format for training and running LLMs
"""
import argparse
import os
import socket
import time

import torch


def benchmark(fn, repeats):
    """Average seconds per call, after one warm-up call."""
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # GPU calls are asynchronous: wait for them to finish
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def matmul_tflops(n, device, dtype, repeats):
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    seconds = benchmark(lambda: a @ b, repeats)
    return 2 * n**3 / seconds / 1e12  # an n x n matmul is 2 n^3 floating-point operations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=8192, help="matrix size n (default: 8192)")
    args = parser.parse_args()

    print(f"Host:            {socket.gethostname()}")
    print(f"Slurm job:       {os.environ.get('SLURM_JOB_ID', '(not in a Slurm job)')}")
    print(f"CPU cores:       {len(os.sched_getaffinity(0))} available to this job "
          f"(the node has {os.cpu_count()})")
    print(f"PyTorch:         {torch.__version__}, built for CUDA {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("No GPU visible! Did you request one with --gres=gpu:<type>:1 ?")
        return
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:             {props.name}")
    print(f"GPU memory:      {props.total_memory / 2**30:.1f} GiB")
    print(f"Multiprocessors: {props.multi_processor_count} SMs "
          f"(a full A100 has 108; a MIG slice gets a fraction of them)")

    n = args.size
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    print(f"\nMatrix multiply, {n} x {n}:")
    cpu = matmul_tflops(n // 2, "cpu", torch.float32, repeats=2)
    print(f"  CPU  fp32   {cpu * 1000:8.1f} GFLOP/s   (n={n // 2}, to keep it short)")

    torch.backends.cuda.matmul.allow_tf32 = False
    fp32 = matmul_tflops(n, "cuda", torch.float32, repeats=10)
    print(f"  GPU  fp32   {fp32 * 1000:8.1f} GFLOP/s   {fp32 / cpu:6.0f}x the CPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    tf32 = matmul_tflops(n, "cuda", torch.float32, repeats=10)
    print(f"  GPU  tf32   {tf32 * 1000:8.1f} GFLOP/s   {tf32 / cpu:6.0f}x the CPU")
    bf16 = matmul_tflops(n, "cuda", torch.bfloat16, repeats=10)
    print(f"  GPU  bf16   {bf16 * 1000:8.1f} GFLOP/s   {bf16 / cpu:6.0f}x the CPU")

    print(f"\nPeak GPU memory used: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
