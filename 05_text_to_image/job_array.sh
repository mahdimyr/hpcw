#!/bin/bash
#SBATCH --job-name=sdxl-array
#SBATCH --array=0-7%4               # one task per line of prompts.txt, at most 4 at a time
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%A_%a.out

# "Embarrassingly parallel" work as a job array: task N renders line N of
# prompts.txt. All tasks write to the same directory, outputs/array-<array job id>
source ../setup/env.sh || exit 1

python generate.py --line "$SLURM_ARRAY_TASK_ID" --seed "$SLURM_ARRAY_TASK_ID" --num-images 4 \
    --out-dir "outputs/array-$SLURM_ARRAY_JOB_ID"
