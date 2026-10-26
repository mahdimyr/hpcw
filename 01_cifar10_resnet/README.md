# 01 · Image classification: ResNet on CIFAR-10

Train a convolutional network to recognize 10 kinds of objects in 32×32 images:
the classic first deep-learning workload, written the way a cluster job should be.

**You will learn:** a complete PyTorch training job, staging data to node-local
disk, checkpoint/restart, mixed precision, the CPU-GPU balance of data loading,
and hyperparameter sweeps with **job arrays**.

The model is **ResNet-9**, the fast design that won the Stanford DAWNBench
training-speed competition (~6.5M parameters). `--model resnet18` gives the
better-known ResNet-18 (~11M).

## Run it

```bash
cd 01_cifar10_resnet
sbatch job.sh                         # 10 epochs, ~1.5 minutes on a 1g.10gb slice
tail -f cifar10-<id>.out              # follow the log live (Ctrl-C to stop watching)
```

```
resnet9: 6.6M parameters
epoch   1/10  loss 1.729  test acc 54.18%   12.4s     4009 img/s
epoch   2/10  loss 1.197  test acc 67.03%    8.7s     5679 img/s
...
epoch  10/10  loss 0.666  test acc 91.62%    8.7s     5682 img/s
Final test accuracy 91.62%, training time 91s, peak GPU memory 1.45 GiB.
```

What [job.sh](job.sh) does that a laptop script doesn't:

- **Data staging:** it copies the dataset (one 180 MB file) to `$SLURM_TMPDIR`,
  the node's local NVMe disk, which is deleted when the job ends. Reading local
  disk is fast and keeps load off the shared file system. One big file rather
  than 60,000 small images matters a lot on cluster storage.
- **Checkpointing:** [train.py](train.py) saves a checkpoint every epoch to
  `outputs/<job-name>/`. Resubmit with the same job name and it
  continues where it stopped. Use a new `--job-name` for a fresh run.

## Job arrays: a hyperparameter sweep

[sweep.sh](sweep.sh) trains the same model with four learning rates. It is
**one** submission that becomes four tasks, each with its own `$SLURM_ARRAY_TASK_ID`:

```bash
sbatch sweep.sh                       # Submitted batch job 5991017
squeue --me                           # 5991017_[2-3] pending, 5991017_0 / _1 running
python3 summarize_sweep.py outputs/sweep-5991017
```

```
run                      model         lr  batch epochs  test acc    time
lr-0.4                   resnet9      0.4    512      5    88.20%     47s
lr-0.2                   resnet9      0.2    512      5    88.10%     47s
lr-0.1                   resnet9      0.1    512      5    87.31%     56s
lr-0.05                  resnet9     0.05    512      5    85.43%     53s
```

`--array=0-3%2` runs at most two tasks at a time, so the sweep doesn't take
every free GPU from your classmates.

You can make the summary run automatically once all tasks are done, with a
**job dependency**:

```bash
jid=$(sbatch --parsable sweep.sh)
sbatch --dependency=afterany:$jid --time=00:02:00 --mem=1G --output=summary-%j.out \
       --wrap="python3 summarize_sweep.py outputs/sweep-$jid"
```

## Things to try

1. **Precision:** `sbatch --job-name=fp32 job.sh --precision fp32`. How much slower is
   it per epoch? Is the accuracy any different?
2. **Kill and resume:** `sbatch --job-name=resume --time=00:01:00 job.sh --epochs 20`.
   After it is killed (`sacct -j <id>` shows `TIMEOUT`), submit the same command
   with a longer `--time` and watch it resume.
3. **Is the GPU waiting for data?** The images are augmented (random crops and flips)
   by CPU worker processes, one per `--cpus-per-task`. Compare `img/s` for
   `--workers 0`, the default, and `--gpu-aug` (augmentation done on the GPU instead).
   On a `1g.10gb` slice the GPU is the bottleneck either way (~5,700 img/s). On a full
   A100 with 4 cores, CPU augmentation reaches ~28,000 img/s and GPU augmentation
   ~37,000 img/s: a faster GPU needs more CPU cores to keep it fed.
4. **torch.compile:** add `--compile`. The first epoch is slower (it compiles), the
   rest faster. When is it worth it?
5. **Bigger model:** `--model resnet18`. More accuracy per epoch, but how much more
   time? Check `peak GPU memory`, and try `--batch-size 1024`.
6. **Your own sweep:** edit `sweep.sh` to sweep over batch size, or over a 2-D grid
   (hint: `i=$SLURM_ARRAY_TASK_ID; lr=${LRS[i % 4]}; bs=${BSS[i / 4]}`).
