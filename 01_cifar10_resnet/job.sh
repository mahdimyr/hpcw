#!/bin/bash
#SBATCH --job-name=cifar10
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2           # also the number of DataLoader workers
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out

# Submit from this directory. Extra arguments go to train.py, e.g.
#     sbatch job.sh --epochs 20
#     sbatch --job-name=r18 job.sh --model resnet18
# Output goes to outputs/<job-name> in this folder. Resubmitting with the same job name
# resumes from the last checkpoint; use a new job name to start a fresh run.
source ../setup/env.sh || exit 1

# Stage the dataset on the node's local NVMe disk ($SLURM_TMPDIR, deleted when the
# job ends): fast to read, and keeps load off the shared file system.
cp "$HPCW_DATA/cifar10.npz" "$SLURM_TMPDIR/"

python train.py --data "$SLURM_TMPDIR/cifar10.npz" --out-dir "outputs/$SLURM_JOB_NAME" "$@"
