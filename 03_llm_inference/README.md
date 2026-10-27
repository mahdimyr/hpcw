# 03 · Running large language models

Run open-weight LLMs on a GPU three ways: a simple script, a high-throughput
batch engine, and your own ChatGPT-style API server inside a Slurm job.

**You will learn:** what decides whether a model fits on a GPU, why serving
engines exist (one request at a time vs. hundreds), how to run a network service
as a batch job, and how to use pre-downloaded models offline.

**Models** (pre-downloaded, Apache-2.0 licensed):

| Model | Released | Weights (bf16) | Notes |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 2025 | 1.2 GB | most downloaded open LLM on Hugging Face |
| `Qwen/Qwen3-1.7B` | 2025 | 3.4 GB | **default** |
| `Qwen/Qwen3-4B` | 2025 | 8.0 GB | only just fits a 10 GB slice |
| `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B` | 2026 | 1.8 / 4.6 GB | hybrid linear attention, also understand images |

## 1. One request at a time: `chat.py`

```bash
cd 03_llm_inference
sbatch job_chat.sh                               # answers the questions in prompts.txt
sbatch job_chat.sh --model Qwen/Qwen3-4B         # a bigger model
```

```
Loaded Qwen/Qwen3-1.7B (1.72B parameters) in 1.9s; weights use 3.20 GiB of 9.5 GiB GPU memory
>>> Write a haiku about waiting in a job queue.
...
[63 tokens in 2.0s = 31.7 tokens/s]
```

Or chat with it live in an interactive session:

```bash
salloc --gres=gpu:1g.10gb:1 --cpus-per-task=2 --mem=16G --time=00:30:00
source ../setup/env.sh
python chat.py --interactive
python chat.py --think --prompt "How many r's are in strawberry?"   # reasoning mode
```

Rule of thumb: **bf16 weights take 2 bytes per parameter**, so a 4B model needs
8 GB before it has processed a single word.

## 2. Hundreds of requests at once: vLLM

[vLLM](https://docs.vllm.ai) is the most widely used open-source LLM serving engine.
It batches many requests together and manages the attention cache in pages
(PagedAttention), so the GPU stays busy.

```bash
sbatch job_vllm.sh                               # 256 prompts
```

```
### Engine ready after 124s
### 256 prompts -> 20700 generated tokens in 9.2s = 2,260 tokens/s
```

That is about 70 times the tokens/s of `chat.py` on the same GPU slice. Startup is
slow the first time (it compiles the model and captures CUDA graphs). Those are
cached in the repository's `work/cache/vllm`, so the next start is faster. vLLM lives in its
own Python environment (`source ../setup/env.sh vllm`) because it pins its own
PyTorch version.

## 3. Your own API server: `serve_vllm.sh`

Run an OpenAI-compatible server as a batch job, then send it requests from the
login node, a notebook, or any OpenAI client library:

```bash
sbatch serve_vllm.sh
tail -f vllm-serve-<id>.out          # wait for "Application startup complete" (~2-3 min)

source vllm-server.env
python3 client.py "What is a job scheduler on a supercomputer?"
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" $OPENAI_BASE_URL/models

scancel <id>                         # stop the server: it holds a GPU until you do
```

The server listens on a random port of the compute node, which anyone on the
cluster can reach. The script therefore generates a secret API key, and requests
without it get `401 Unauthorized`. The connection details are written to
`vllm-server.env`, a file only you can read.

## Things to try

1. **Will it fit?** Before running, compute the memory for the weights of each
   model in the table. Then check with `chat.py` (it prints the actual number).
   Which is the largest Qwen3 model that could fit a `1g.10gb` slice? A full A100 (80 GB)?
2. **Newest generation:** `sbatch job_chat.sh --model Qwen/Qwen3.5-2B`. Its first run
   compiles extra GPU kernels for the linear-attention layers, so it starts slowly.
3. **Thinking:** compare the answer to a tricky question with and without `--think`.
   How many more tokens does it generate, and how much longer does it take?
4. **Throughput vs. batch size:** run `job_vllm.sh --num-prompts 16`, `256` and `1024`.
   How do tokens/s change? Why?
5. **KV cache:** vLLM reports `Available KV cache memory` and `GPU KV cache size: N tokens`
   in its log: memory for the attention cache of all requests in flight. Run
   `sbatch job_vllm.sh --model Qwen/Qwen3-4B` and read the error. Why does a model that
   works in `chat.py` fail in vLLM on a 10 GB slice? Which GPU type would you request?
6. **Use your server:** point any tool that speaks the OpenAI API at your server,
   e.g. `pip install openai` in your own environment and use
   `OpenAI(base_url=..., api_key=...)`.
