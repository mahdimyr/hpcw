#!/bin/bash
#SBATCH --job-name=sdxl-turbo
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out

# Submit from this directory. Extra arguments go to generate.py, e.g.
#     sbatch job.sh --prompt "a moose in a spacesuit" --num-images 8
# Images are saved to outputs/<job-id> in this folder
source ../setup/env.sh || exit 1

python generate.py --out-dir "outputs/$SLURM_JOB_ID" "$@"
