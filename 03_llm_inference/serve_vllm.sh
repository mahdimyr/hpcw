#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Run your own ChatGPT-style API: an OpenAI-compatible LLM server inside a Slurm job.
# Once the log shows "Application startup complete" (2-3 min), from the login node:
#     source vllm-server.env
#     python3 client.py "What is a job scheduler?"
# Stop the server with scancel when you are done -- it holds the GPU until then.
# Choose another model with:  MODEL=Qwen/Qwen3.5-2B sbatch serve_vllm.sh
source ../setup/env.sh vllm || exit 1

MODEL=${MODEL:-Qwen/Qwen3-1.7B}
PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
# Anyone on the cluster can reach this port, so require a secret key.
API_KEY=$(python -c 'import secrets; print(secrets.token_hex(16))')

# Connection details, readable only by you.
(umask 077; echo "export OPENAI_BASE_URL=http://$(hostname):$PORT/v1 OPENAI_API_KEY=$API_KEY VLLM_MODEL=$MODEL" \
    > vllm-server.env)
echo "Serving $MODEL on http://$(hostname):$PORT/v1 -- connection details in $PWD/vllm-server.env"

vllm serve "$MODEL" --port "$PORT" --api-key "$API_KEY" --gpu-memory-utilization 0.85 --max-model-len 4096
