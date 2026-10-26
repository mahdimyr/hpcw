# 00 · Hello, GPU

Your first GPU job: find out what Slurm gave you and how much faster a GPU is
than a CPU core.

**You will learn:** writing and submitting a batch script, requesting a GPU,
reading job output, interactive sessions, and number formats (fp32 / tf32 / bf16).

## Run it

```bash
cd 00_gpu_hello
sbatch job.sh            # prints "Submitted batch job <id>"
squeue --me              # PD = pending (waiting), R = running
cat gpu-hello-<id>.out   # the output, once the job has run
```

Typical output on a `1g.10gb` slice:

```
GPU:             NVIDIA A100 80GB PCIe MIG 1g.10gb
GPU memory:      9.5 GiB
Multiprocessors: 14 SMs (a full A100 has 108; a MIG slice gets a fraction of them)

Matrix multiply, 8192 x 8192:
  CPU  fp32      406.1 GFLOP/s   (n=4096, to keep it short)
  GPU  fp32     2458.3 GFLOP/s        6x the CPU
  GPU  tf32    18930.5 GFLOP/s       47x the CPU
  GPU  bf16    37758.3 GFLOP/s       93x the CPU
```

The same GPU is about 15x faster in bf16 than in plain fp32. Tensor Cores only
work on reduced-precision formats, which is why deep learning trains in bf16.

## Read the job script

Open [job.sh](job.sh). Every `#SBATCH` line is a resource request:

| Line | Meaning |
|---|---|
| `--gres=gpu:1g.10gb:1` | one GPU of type `1g.10gb` (a 1/7 slice of an A100). Plato **requires** a GPU type |
| `--cpus-per-task=2` | CPU cores |
| `--mem=8G` | CPU memory (RAM). Without it you get only 512 MB per core |
| `--time=00:05:00` | maximum run time. The job is killed when it runs out, and short jobs start sooner |
| `--output=%x-%j.out` | log file name: `%x` = job name, `%j` = job id |

There is no `--partition` or `--account`: on Plato your account decides where
the job runs, and specifying a partition is an error.

## Interactive session

For trying things out, ask for a shell on a GPU node:

```bash
salloc --gres=gpu:1g.10gb:1 --cpus-per-task=2 --mem=8G --time=00:30:00
# ... wait for the prompt: you are now on the GPU node (check with: hostname)
source ../setup/env.sh
nvidia-smi
python gpu_check.py --size 4096
exit                     # give the GPU back!
```

An idle `salloc` session still holds a GPU that someone else could use. Exit
as soon as you are done.

## Things to try

1. Run `sacct -j <id> --format=JobID,JobName,Elapsed,State,AllocTRES%60` and
   `seff <id>` after the job finishes. How much of the memory you asked for did it use?
2. Ask for a GPU type that doesn't exist (`--gres=gpu:h100:1`), or for no GPU at
   all. What happens?
3. Run on a full A100 (`sbatch --gres=gpu:a100:1 job.sh`) and compare the numbers
   with a slice. Is the A100 7 times faster?
4. Submit the job three times in a row (`for i in 1 2 3; do sbatch job.sh; done`).
   Did they run one after the other or at the same time? On the same GPU? (Compare the
   `MIG ... UUID` lines that `nvidia-smi -L` printed in each log.)
