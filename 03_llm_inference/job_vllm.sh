#!/bin/bash
#SBATCH --job-name=llm-vllm
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out

# Submit from this directory. Extra arguments go to batch_vllm.py, e.g.
#     sbatch job_vllm.sh --num-prompts 1000
source ../setup/env.sh vllm || exit 1    # note: the separate vLLM environment

python batch_vllm.py --output "outputs/vllm-$SLURM_JOB_ID.jsonl" "$@"
