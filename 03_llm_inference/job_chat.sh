#!/bin/bash
#SBATCH --job-name=llm-chat
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out

# Submit from this directory. Extra arguments go to chat.py, e.g.
#     sbatch job_chat.sh --model Qwen/Qwen3.5-2B
source ../setup/env.sh || exit 1

python chat.py "$@"
