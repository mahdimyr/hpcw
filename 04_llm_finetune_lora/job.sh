#!/bin/bash
#SBATCH --job-name=lora-sql
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out

# Submit from this directory. Extra arguments go to finetune_lora.py, e.g.
#     sbatch job.sh --steps 500 --lora-rank 32
# The adapter is saved to outputs/<job-id> in this folder
source ../setup/env.sh || exit 1

python finetune_lora.py --out-dir "outputs/$SLURM_JOB_ID" "$@"
