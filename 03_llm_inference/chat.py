"""Run an open-weight LLM on a GPU with Hugging Face transformers.

    python chat.py                                   # answer the prompts in prompts.txt
    python chat.py --model Qwen/Qwen3.5-2B --prompt "Why is the sky blue?"
    python chat.py --interactive                     # chat (inside an salloc session)
    python chat.py --think --prompt "Is 1001 prime?" # let Qwen "think" before answering

Generation here is one conversation at a time. Compare its tokens/s with
batch_vllm.py, which serves hundreds of requests at once.
"""
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--prompt", action="append", help="a prompt (repeatable); default: prompts.txt")
    p.add_argument("--prompts-file", default=Path(__file__).with_name("prompts.txt"))
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--think", action="store_true", help="enable the model's reasoning mode (Qwen3/3.5)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.model} ({n_params / 1e9:.2f}B parameters) in {time.perf_counter() - t0:.1f}s; "
          f"weights use {torch.cuda.memory_allocated() / 2**30:.2f} GiB of "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB GPU memory\n")

    def reply(messages):
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                               return_dict=True, enable_thinking=args.think).to("cuda")
        start = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                                temperature=args.temperature, top_p=0.9,
                                streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True))
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        seconds = time.perf_counter() - start
        print(f"[{len(new_tokens)} tokens in {seconds:.1f}s = {len(new_tokens) / seconds:.1f} tokens/s]\n")
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    if args.interactive:
        messages = []
        print("Type a message (empty line or Ctrl-D to quit).")
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                break
            if not text:
                break
            messages.append({"role": "user", "content": text})
            print("model> ", end="", flush=True)
            messages.append({"role": "assistant", "content": reply(messages)})
    else:
        prompts = args.prompt or [l.strip() for l in Path(args.prompts_file).read_text().splitlines() if l.strip()]
        for prompt in prompts:
            print(f">>> {prompt}")
            reply([{"role": "user", "content": prompt}])

    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
