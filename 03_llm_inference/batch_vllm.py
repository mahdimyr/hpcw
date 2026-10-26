"""Offline batch inference with vLLM: hundreds of prompts in one go.

vLLM keeps the GPU busy by batching many requests together (continuous batching)
and managing the attention cache in pages (PagedAttention). Compare the tokens/s
printed here with chat.py, which handles one request at a time.

    python batch_vllm.py                               # 256 prompts with Qwen3-1.7B
    python batch_vllm.py --model Qwen/Qwen3.5-2B --num-prompts 1000
"""
import argparse
import json
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams

TOPICS = ["black holes", "photosynthesis", "the French Revolution", "prime numbers", "volcanoes",
          "machine learning", "coral reefs", "the printing press", "quantum computing", "honey bees",
          "plate tectonics", "the immune system", "jazz", "the Silk Road", "superconductors", "glaciers"]
TASKS = ["Explain {} in two sentences.", "Write a haiku about {}.", "Give one surprising fact about {}.",
         "Write a multiple-choice quiz question about {}."]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=4096, help="longest prompt + answer, in tokens")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                   help="fraction of GPU memory vLLM may take (weights + KV cache)")
    p.add_argument("--output", default="outputs/vllm.jsonl")
    args = p.parse_args()

    prompts = [TASKS[i % len(TASKS)].format(TOPICS[(i // len(TASKS)) % len(TOPICS)])
               for i in range(args.num_prompts)]

    t0 = time.perf_counter()
    # Startup is slow the first time (torch.compile + CUDA graph capture, ~2-3 min);
    # the compiled kernels are cached in $VLLM_CACHE_ROOT, so later runs start faster.
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    print(f"### Engine ready after {time.perf_counter() - t0:.0f}s")

    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_tokens)
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    t0 = time.perf_counter()
    # use_tqdm=False: we print our own summary, and vLLM's progress bar crashes when
    # TQDM_DISABLE is set (env.sh sets it in batch jobs to keep logs readable)
    outputs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": False},
                       use_tqdm=False)
    seconds = time.perf_counter() - t0

    n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"### {len(prompts)} prompts -> {n_tokens} generated tokens in {seconds:.1f}s "
          f"= {n_tokens / seconds:,.0f} tokens/s")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for prompt, out in zip(prompts, outputs):
            f.write(json.dumps({"prompt": prompt, "response": out.outputs[0].text}) + "\n")
    print(f"### All responses written to {args.output}. Three of them:")
    for prompt, out in list(zip(prompts, outputs))[1:12:5]:
        print(f"\n>>> {prompt}\n{out.outputs[0].text.strip()}")


if __name__ == "__main__":
    main()
