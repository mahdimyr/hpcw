#!/bin/bash
#SBATCH --job-name=cifar10-sweep
#SBATCH --array=0-3%2               # 4 tasks (indices 0..3), at most 2 running at a time
#SBATCH --gres=gpu:1g.10gb:1        # resources are per task
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%A_%a.out       # %A = array job id, %a = task index

# Hyperparameter sweep as a job array: every task trains the same model with a
# different learning rate. Afterwards compare them with
#     python3 summarize_sweep.py outputs/sweep-<array job id>
source ../setup/env.sh || exit 1

LEARNING_RATES=(0.05 0.1 0.2 0.4)
lr=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}
echo "Array task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID: lr=$lr"

cp "$HPCW_DATA/cifar10.npz" "$SLURM_TMPDIR/"
python train.py --data "$SLURM_TMPDIR/cifar10.npz" --epochs 5 --lr "$lr" \
    --out-dir "outputs/sweep-$SLURM_ARRAY_JOB_ID/lr-$lr"
