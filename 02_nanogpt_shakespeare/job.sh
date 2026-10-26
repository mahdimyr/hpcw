#!/bin/bash
#SBATCH --job-name=nanogpt
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out

# Submit from this directory. Extra arguments go to train_gpt.py, e.g.
#     sbatch job.sh --n-layer 6 --n-embd 384 --iters 3000
# The model is saved to outputs/<job-name>/model.pt in this folder
source ../setup/env.sh || exit 1

python train_gpt.py --out-dir "outputs/$SLURM_JOB_NAME" "$@"
