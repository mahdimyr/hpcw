#!/bin/bash
#SBATCH --job-name=gpu-hello
#SBATCH --gres=gpu:1g.10gb:1        # GPU type:count -- here one 10 GB MIG slice of an A100
#SBATCH --cpus-per-task=2           # CPU cores
#SBATCH --mem=8G                    # host (CPU) memory; the default is only 512 MB per core
#SBATCH --time=00:05:00             # wall-time limit (hh:mm:ss); shorter jobs start sooner
#SBATCH --output=%x-%j.out          # log file: <job-name>-<job-id>.out

# Submit from this directory:   cd 00_gpu_hello && sbatch job.sh
source ../setup/env.sh || exit 1

echo "Job $SLURM_JOB_ID running on $(hostname) at $(date)"
nvidia-smi -L        # the GPU(s) this job can see
python gpu_check.py
